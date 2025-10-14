from __future__ import annotations

from fastapi import APIRouter, HTTPException, Response, Query
from pydantic import BaseModel
from typing import Any, Dict, Optional, List
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import io
import time
import json
import uuid
import base64

from ...storage.minio_client import (
    list_negatives_objects,
    get_negative_image_bytes,
    remove_negative_image,
    get_training_image_bytes, 
    training_bucket_name, 
    get_client as get_minio_client, 
    ensure_training_bucket, 
    ensure_things_bucket, 
    get_things_image_bytes,
    get_object_bytes,
    put_image_bytes,
    things_bucket_name
)
from ...database.milvus import ensure_training_collection, update_embedding_labels, commit_embeddings_to_things, update_things_labels, delete_selected_things_from_milvus_and_minio
from ...database.mongo import insert_one as mongo_insert_one, find_many as mongo_find_many, find_one as mongo_find_one, delete_one as mongo_delete_one, delete_many as mongo_delete_many, drop_collection as mongo_drop_collection, update_one as mongo_update_one, is_connected as mongo_is_connected, connect as mongo_connect
from ...service.embedder_service import embed_image
from pymilvus import Collection, utility
from bson import ObjectId  # type: ignore

router = APIRouter()


# Pydantic Models
class UpdateLabelsRequest(BaseModel):
    embedding_ids: list[str]
    label: str

class CommitToThingsRequest(BaseModel):
    embedding_ids: list[str]

class CreateThingGroupRequest(BaseModel):
    embedding_ids: list[str]

class ThingGroupResponse(BaseModel):
    id: str
    name: str
    thumbnail_b64: Optional[str] = None
    count: int

class AddToGroupRequest(BaseModel):
    group_id: str
    embedding_ids: list[str]

class RemoveFromGroupRequest(BaseModel):
    group_id: str
    embedding_ids: list[str]

class EditCropRequest(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float

class RenameGroupRequest(BaseModel):
    name: str

class SetGroupThumbnailRequest(BaseModel):
    image_id: str


# Helper Functions
def _ensure_mongo_connection():
    try:
        if not mongo_is_connected():
            mongo_connect()
    except Exception:
        pass


def _get_things_by_ids(int_ids: list[int]) -> list[dict]:
    collection_name = "things"
    coll = Collection(collection_name)
    coll.load()
    try:
        expr = f"id in {int_ids}"
        res = coll.query(expr=expr, output_fields=["id", "payload", "embedding"], limit=len(int_ids))  # type: ignore
        items = []
        for r in res:
            pid = int(r.get("id"))
            payload = r.get("payload")
            try:
                payload_obj = json.loads(payload) if isinstance(payload, str) else payload
            except Exception:
                payload_obj = {"raw": payload}
            items.append({"id": pid, "payload": payload_obj, "embedding": r.get("embedding")})
        return items
    finally:
        coll.release()


@router.get("/negatives/list")
def list_negatives() -> dict:
    try:
        items = list_negatives_objects()
        return {"items": items}
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"Failed to list negatives: {ex}")


@router.get("/negatives/image/{object_name:path}")
def get_negative_image(object_name: str):
    try:
        data, content_type = get_negative_image_bytes(object_name)
        return Response(content=data, media_type=content_type or "image/jpeg")
    except Exception as ex:
        raise HTTPException(status_code=404, detail=f"Negative image not found: {ex}")


@router.delete("/negatives/{object_name:path}")
def delete_negative_image(object_name: str) -> dict:
    try:
        ok = remove_negative_image(object_name)
        # Best-effort: remove any metadata if stored (ignore errors and optional dependency)
        try:
            mongo_delete_one("negatives", {"object_name": object_name})
        except Exception:
            pass
        return {"deleted": bool(ok)}
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"Failed to delete negative: {ex}")


# ==================== THINGS ROUTES ====================

@router.post("/labels")
def update_things_labels_endpoint(req: UpdateLabelsRequest) -> Dict[str, Any]:
    """Update labels for multiple embeddings in the 'things' collection."""
    try:
        updated_count = update_things_labels(req.embedding_ids, req.label)
        return {"updated": updated_count}
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"Failed to update things labels: {ex}")


@router.delete("/selected")
def delete_selected_things(req: CommitToThingsRequest) -> Dict[str, Any]:
    """Delete selected embeddings from the 'things' collection and their images from MinIO."""
    try:
        deleted_count = delete_selected_things_from_milvus_and_minio(req.embedding_ids)
        return {"deleted": deleted_count}
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"Failed to delete selected things: {ex}")


@router.get("/search")
def search_things_embeddings(
    query: str = Query(..., description="Search query for label field"),
    limit: int = Query(200, ge=1, le=10000), 
    offset: int = Query(0, ge=0)
) -> Dict[str, Any]:
    """Search embeddings in the 'things' collection by label."""
    try:
        collection_name = "things"
        
        # Check if collection exists
        if not utility.has_collection(collection_name):
            return {"items": [], "total": 0}
        
        collection = Collection(collection_name)
        collection.load()
        
        try:
            # Prefer scalar group_id to select only ungrouped items
            try:
                has_group_field = any(getattr(f, 'name', '') == 'group_id' for f in collection.schema.fields)
            except Exception:
                has_group_field = False
            expr = 'group_id == "unknown"' if has_group_field else ""
            res = collection.query(expr=expr, output_fields=["id", "payload"], limit=16384)  # type: ignore
            items = []
            for r in res:
                pid = r.get("id")
                payload = r.get("payload")
                try:
                    payload_obj = json.loads(payload) if isinstance(payload, str) else payload
                except Exception:
                    payload_obj = {"raw": payload}
                
                # Check if label matches search query (case-insensitive)
                label = payload_obj.get("label", "") if isinstance(payload_obj, dict) else ""
                if query.lower() in label.lower():
                    # Convert Milvus ID to string to avoid JavaScript integer precision issues
                    items.append({"id": str(pid), "payload": payload_obj})
            
            # Sort by frame number (frame_index in payload) or by original_id if available
            def get_sort_key(item):
                payload = item.get("payload", {})
                if isinstance(payload, dict):
                    # Use frame_index if available, otherwise use original_id, otherwise use 0
                    return payload.get("frame_index", payload.get("original_id", 0))
                return 0
            
            items.sort(key=get_sort_key)
            
            # Apply pagination after sorting and filtering
            start_idx = offset
            end_idx = offset + limit
            paginated_items = items[start_idx:end_idx]
            
            return {"items": paginated_items, "total": len(items)}
        finally:
            collection.release()
            
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"Failed to search things embeddings: {ex}")


@router.get("")
def get_things_embeddings(limit: int = Query(200, ge=1, le=10000), offset: int = Query(0, ge=0)) -> Dict[str, Any]:
    """Get all embeddings from the 'things' collection."""
    try:
        collection_name = "things"
        
        # Check if collection exists
        if not utility.has_collection(collection_name):
            return {"items": [], "total": 0}
        
        collection = Collection(collection_name)
        collection.load()
        
        try:
            # Prefer scalar group_id to select only ungrouped items
            try:
                has_group_field = any(getattr(f, 'name', '') == 'group_id' for f in collection.schema.fields)
            except Exception:
                has_group_field = False
            expr = 'group_id == "unknown"' if has_group_field else ""
            res = collection.query(expr=expr, output_fields=["id", "payload"], limit=16384)  # type: ignore
            items = []
            for r in res:
                pid = r.get("id")
                payload = r.get("payload")
                try:
                    payload_obj = json.loads(payload) if isinstance(payload, str) else payload
                except Exception:
                    payload_obj = {"raw": payload}
                # Convert Milvus ID to string to avoid JavaScript integer precision issues
                items.append({"id": str(pid), "payload": payload_obj})
            
            # Sort by frame number (frame_index in payload) or by original_id if available
            def get_sort_key(item):
                payload = item.get("payload", {})
                if isinstance(payload, dict):
                    # Use frame_index if available, otherwise use original_id, otherwise use 0
                    return payload.get("frame_index", payload.get("original_id", 0))
                return 0
            
            items.sort(key=get_sort_key)
            
            # Apply pagination after sorting and filtering
            start_idx = offset
            end_idx = offset + limit
            paginated_items = items[start_idx:end_idx]
            
            return {"items": paginated_items, "total": len(items)}
        finally:
            collection.release()
            
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"Failed to get things embeddings: {ex}")


@router.get("/image/{filename}")
def get_things_image(filename: str) -> Response:
    """Get an image from the things collection by filename."""
    try:
        print(f"[DEBUG] Requesting things image: {filename}")
        
        # For things collection, images are stored with image_id as filename
        # We need to get the image from MinIO using the things bucket
        bucket_name = "things"  # Use a dedicated things bucket
        
        # Ensure the things bucket exists
        ensure_things_bucket()
        
        # Get image from MinIO
        image_data = get_things_image_bytes(filename)
        
        print(f"[DEBUG] Successfully retrieved things image: {filename} ({len(image_data)} bytes)")
        return Response(content=image_data, media_type="image/jpeg")
    except Exception as ex:
        print(f"[ERROR] Failed to get things image {filename}: {ex}")
        raise HTTPException(status_code=404, detail=f"Image not found: {ex}")


@router.post("/{embedding_id}/crop")
def edit_thing_crop(embedding_id: str, req: EditCropRequest) -> Dict[str, Any]:
    """Update a thing's crop and embedding given new bbox.

    - Loads the full frame ({image_id}_full.jpg) from MinIO
    - Crops to the new bbox, re-embeds, and upserts Milvus with same id
    - Overwrites the crop image in MinIO with same filename {image_id}.jpg
    - Updates payload.bbox to [x1,y1,x2,y2]
    """
    try:
        coll = Collection("things")
        coll.load()
        # Fetch existing row to get image_id and current payload
        res = coll.query(expr=f"id in [{int(embedding_id)}]", output_fields=["id", "payload", "embedding"], limit=1)  # type: ignore
        if not res:
            raise HTTPException(status_code=404, detail="Embedding not found")
        row = res[0]
        payload = row.get("payload")
        try:
            payload_obj = json.loads(payload) if isinstance(payload, str) else payload
        except Exception:
            payload_obj = {"raw": payload}
        if not isinstance(payload_obj, dict):
            payload_obj = {}
        image_id = payload_obj.get("image_id")
        if not image_id:
            raise HTTPException(status_code=400, detail="Missing image_id in payload")

        # Load full frame from MinIO
        full_name = f"{image_id}_full.jpg"
        data, _ = get_object_bytes(things_bucket_name(), full_name)
        pil = Image.open(io.BytesIO(data)).convert('RGB')

        # Crop to bbox
        x1, y1, x2, y2 = float(req.x1), float(req.y1), float(req.x2), float(req.y2)
        x1i, y1i, x2i, y2i = max(0, int(x1)), max(0, int(y1)), max(0, int(x2)), max(0, int(y2))
        x2i = max(x1i+1, x2i)
        y2i = max(y1i+1, y2i)
        crop = pil.crop((x1i, y1i, x2i, y2i))

        # Re-embed
        emb = embed_image(crop)
        emb_list = emb.reshape(1, -1).tolist()
        
        # Update payload
        payload_obj["bbox"] = [x1, y1, x2, y2]
        new_payload_json = json.dumps(payload_obj)

        # Upsert in Milvus with same id
        try:
            # If schema has group_id, include it as 4th column
            has_group = any(getattr(f, 'name', '') == 'group_id' for f in coll.schema.fields)
        except Exception:
            has_group = False
        if has_group:
            group_val = payload_obj.get("group_id") or "unknown"
            coll.upsert([[int(embedding_id)], emb_list, [new_payload_json], [group_val]])
        else:
            coll.upsert([[int(embedding_id)], emb_list, [new_payload_json]])
        coll.flush()

        # Overwrite crop image in MinIO
        bio = io.BytesIO()
        crop.save(bio, format='JPEG', quality=90)
        put_image_bytes(things_bucket_name(), f"{image_id}.jpg", bio.getvalue(), "image/jpeg")
        return {"updated": 1, "id": str(embedding_id), "bbox": [x1, y1, x2, y2]}
    except HTTPException:
        raise
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"Failed to edit crop: {ex}")


@router.delete("")
def delete_all_things() -> Dict[str, Any]:
    """Delete all things from the 'things' collection and bucket."""
    try:
        collection_name = "things"
        bucket_name = "things"
        
        # Check if collection exists
        if not utility.has_collection(collection_name):
            return {"message": "Things collection does not exist", "deleted_embeddings": 0, "deleted_images": 0}
        
        # Get collection info before deletion
        collection = Collection(collection_name)
        collection.load()
        total_count = collection.num_entities
        
        # Drop the collection
        utility.drop_collection(collection_name)
        
        # Delete all objects from the things bucket
        minio_client = get_minio_client()
        object_count = 0
        try:
            # List all objects in the things bucket
            objects = minio_client.list_objects(bucket_name, recursive=True)
            for obj in objects:
                try:
                    minio_client.remove_object(bucket_name, obj.object_name)
                    object_count += 1
                except Exception as _rm_ex:
                    print(f"[WARN] Failed to remove object {obj.object_name} from {bucket_name}: {_rm_ex}")
            # Remove the bucket itself
            try:
                minio_client.remove_bucket(bucket_name)
            except Exception as _rb_ex:
                print(f"[INFO] Could not remove bucket {bucket_name}: {_rb_ex}")
        except Exception as bucket_ex:
            # If bucket doesn't exist or is already empty, that's fine
            print(f"[INFO] Things bucket cleanup: {bucket_ex}")
        
        # Also delete all thing_groups from MongoDB
        try:
            _ensure_mongo_connection()
            groups_deleted = int(mongo_delete_many("thing_groups", {}))
            # Optionally drop the collection to clean indexes
            try:
                mongo_drop_collection("thing_groups")
            except Exception:
                pass
        except Exception as mg_ex:
            print(f"[INFO] Failed to delete thing_groups: {mg_ex}")
            groups_deleted = 0

        return {
            "message": f"Successfully deleted all things and groups",
            "deleted_embeddings": total_count,
            "deleted_images": object_count,
            "deleted_groups": groups_deleted,
        }
        
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"Failed to delete all things: {ex}")


# ==================== THING GROUPS ROUTES ====================

@router.post("/groups", response_model=ThingGroupResponse)
def create_thing_group(req: CreateThingGroupRequest) -> Dict[str, Any]:
    """
    Create a thing group document in MongoDB with fields: _id, name, thumbnail_b64, thing_ids, count.
    - name is the label of the first item
    - thumbnail_b64 is a 200px wide JPEG base64 of the first item's image
    - All items get the same label (name) in Milvus 'things' collection
    """
    try:
        if not req.embedding_ids:
            raise HTTPException(status_code=400, detail="No embedding_ids provided")

        # Convert to int ids and fetch items
        try:
            int_ids = [int(x) for x in req.embedding_ids]
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid embedding_ids")

        items = _get_things_by_ids(int_ids)
        if not items:
            raise HTTPException(status_code=404, detail="No matching things found")

        # Determine name from first label
        first_payload = items[0].get("payload", {}) if isinstance(items[0].get("payload"), dict) else {}
        group_name = first_payload.get("label") or "Group"

        # Build thumbnail from first item's image_id (200px width)
        image_id = first_payload.get("image_id")
        thumb_b64: Optional[str] = None
        if image_id:
            try:
                data, _ = get_object_bytes(things_bucket_name(), f"{image_id}.jpg")
                pil = Image.open(io.BytesIO(data)).convert('RGB')
                w, h = pil.size
                if w > 200:
                    new_w = 200
                    new_h = max(1, int(round(h * (new_w / float(max(1, w))))))
                    pil = pil.resize((new_w, new_h))
                bio = io.BytesIO()
                pil.save(bio, format='JPEG', quality=80)
                thumb_b64 = base64.b64encode(bio.getvalue()).decode('utf-8')
            except Exception:
                thumb_b64 = None

        # Upsert payloads in Milvus to include group_id and ensure label is set to group_name
        try:
            coll = Collection("things")
            coll.load()
            try:
                ids_list: list[int] = []
                embeddings_list: list[list[float]] = []
                payloads_list: list[str] = []
                for it in items:
                    pid = int(it["id"])
                    current_payload = it.get("payload", {})
                    if isinstance(current_payload, str):
                        try:
                            current_payload = json.loads(current_payload)
                        except Exception:
                            current_payload = {"raw": current_payload}
                    # Preserve image_id or generate if missing
                    if not isinstance(current_payload, dict):
                        current_payload = {}
                    if "image_id" not in current_payload or not current_payload.get("image_id"):
                        current_payload["image_id"] = str(uuid.uuid4())
                    # Set/overwrite label and group_id
                    current_payload["label"] = group_name
                    current_payload["group_id"] = "unknown"  # will set after we have oid
                    ids_list.append(pid)
                    embeddings_list.append(it.get("embedding", []))
                    payloads_list.append(json.dumps(current_payload))
            finally:
                # We will upsert after we know group_id (oid)
                coll.release()
        except Exception as ex:
            print(f"[ERROR] Failed preparing upsert data for group payload updates: {ex}")

        # Insert Mongo document
        _ensure_mongo_connection()
        doc = {
            "name": group_name,
            "thumbnail_b64": thumb_b64,
            "ts": int(time.time()*1000),
        }
        oid = mongo_insert_one("thing_groups", doc)

        # Now perform the upsert with group_id filled
        try:
            coll2 = Collection("things")
            coll2.load()
            try:
                # Rebuild payloads with group_id set to oid
                ids_list2: list[int] = []
                embeddings_list2: list[list[float]] = []
                payloads_list2: list[str] = []
                for it in items:
                    pid = int(it["id"])
                    current_payload = it.get("payload", {})
                    if isinstance(current_payload, str):
                        try:
                            current_payload = json.loads(current_payload)
                        except Exception:
                            current_payload = {"raw": current_payload}
                    if not isinstance(current_payload, dict):
                        current_payload = {}
                    # Ensure image_id exists
                    if "image_id" not in current_payload or not current_payload.get("image_id"):
                        current_payload["image_id"] = str(uuid.uuid4())
                    current_payload["label"] = group_name
                    current_payload["group_id"] = str(oid)
                    ids_list2.append(pid)
                    embeddings_list2.append(it.get("embedding", []))
                    payloads_list2.append(json.dumps(current_payload))
                if ids_list2:
                    # Include group_id column when available
                    try:
                        has_group_field = any(getattr(f, 'name', '') == 'group_id' for f in coll2.schema.fields)
                    except Exception:
                        has_group_field = False
                    if has_group_field:
                        group_ids2 = [str(oid)] * len(ids_list2)
                        coll2.upsert([ids_list2, embeddings_list2, payloads_list2, group_ids2])
                    else:
                        coll2.upsert([ids_list2, embeddings_list2, payloads_list2])
                    coll2.flush()
            finally:
                coll2.release()
        except Exception as ex:
            print(f"[ERROR] Failed to upsert group_id to things payloads: {ex}")

        return {
            "id": str(oid),
            "name": group_name,
            "thumbnail_b64": thumb_b64,
            "count": len(int_ids),
        }
    except HTTPException:
        raise
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"Failed to create group: {ex}")


@router.get("/groups", response_model=list[ThingGroupResponse])
def list_thing_groups() -> List[Dict[str, Any]]:
    try:
        _ensure_mongo_connection()
        groups = mongo_find_many("thing_groups", sort=[["ts", -1]], projection=None) or []
        out: List[Dict[str, Any]] = []
        for g in groups:
            out.append({
                "id": str(g.get("_id")),
                "name": g.get("name") or "Group",
                "thumbnail_b64": g.get("thumbnail_b64"),
                "count": int(g.get("count") or 0),
            })
        return out
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"Failed to list groups: {ex}")


@router.get("/groups/{group_id}")
def get_thing_group(group_id: str) -> Dict[str, Any]:
    try:
        _ensure_mongo_connection()
        g = mongo_find_one("thing_groups", {"_id": ObjectId(group_id)})
        if not g:
            raise HTTPException(status_code=404, detail="Group not found")
        # Fetch items from Milvus by group_id expression (preferred)
        coll = Collection("things")
        coll.load()
        try:
            expr = f"group_id == \"{group_id}\""
            res = coll.query(expr=expr, output_fields=["id", "payload"], limit=16384)  # type: ignore
            items = []
            for r in res or []:
                pid = r.get("id")
                payload = r.get("payload")
                try:
                    payload_obj = json.loads(payload) if isinstance(payload, str) else payload
                except Exception:
                    payload_obj = {"raw": payload}
                items.append({"id": str(pid), "payload": payload_obj})
        finally:
            coll.release()
        return {
            "id": str(g.get("_id")),
            "name": g.get("name") or "Group",
            "thumbnail_b64": g.get("thumbnail_b64"),
            "count": int(g.get("count") or 0),
            "items": items,
        }
    except HTTPException:
        raise


@router.post("/groups/{group_id}/thumbnail")
def set_group_thumbnail(group_id: str, req: SetGroupThumbnailRequest) -> Dict[str, Any]:
    try:
        _ensure_mongo_connection()
        g = mongo_find_one("thing_groups", {"_id": ObjectId(group_id)})
        if not g:
            raise HTTPException(status_code=404, detail="Group not found")
        # Load crop image and create 200px wide preview
        data, _ = get_object_bytes("things", f"{req.image_id}.jpg")
        pil = Image.open(io.BytesIO(data)).convert('RGB')
        w, h = pil.size
        if w > 200:
            new_w = 200
            new_h = max(1, int(round(h * (new_w / float(max(1, w))))))
            pil = pil.resize((new_w, new_h))
        bio = io.BytesIO()
        pil.save(bio, format='JPEG', quality=80)
        thumb_b64 = base64.b64encode(bio.getvalue()).decode('utf-8')
        mongo_update_one("thing_groups", {"_id": ObjectId(group_id)}, {"$set": {"thumbnail_b64": thumb_b64}})
        return {"updated": True, "thumbnail_b64": thumb_b64}
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"Failed to set group thumbnail: {ex}")


@router.post("/groups/add")
def add_to_thing_group(req: AddToGroupRequest) -> Dict[str, Any]:
    """Add selected things to an existing group: update Mongo doc and Milvus payloads (label + group_id)."""
    try:
        if not req.embedding_ids:
            raise HTTPException(status_code=400, detail="No embedding_ids provided")
        # Validate group exists
        _ensure_mongo_connection()
        g = mongo_find_one("thing_groups", {"_id": ObjectId(req.group_id)})
        if not g:
            raise HTTPException(status_code=404, detail="Group not found")
        group_name = g.get("name") or "Group"

        # Fetch items by ids
        try:
            int_ids = [int(x) for x in req.embedding_ids]
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid embedding_ids")
        items = _get_things_by_ids(int_ids)
        if not items:
            return {"added": 0}

        # Upsert Milvus payloads (set label and group_id)
        coll = Collection("things")
        coll.load()
        try:
            ids_list: list[int] = []
            embeddings_list: list[list[float]] = []
            payloads_list: list[str] = []
            for it in items:
                pid = int(it["id"])
                payload = it.get("payload", {})
                if isinstance(payload, str):
                    try:
                        payload = json.loads(payload)
                    except Exception:
                        payload = {"raw": payload}
                if not isinstance(payload, dict):
                    payload = {}
                payload["group_id"] = req.group_id
                payload["label"] = group_name
                ids_list.append(pid)
                embeddings_list.append(it.get("embedding", []))
                payloads_list.append(json.dumps(payload))
            if ids_list:
                try:
                    has_group_field = any(getattr(f, 'name', '') == 'group_id' for f in coll.schema.fields)
                except Exception:
                    has_group_field = False
                if has_group_field:
                    coll.upsert([ids_list, embeddings_list, payloads_list, [req.group_id] * len(ids_list)])
                else:
                    coll.upsert([ids_list, embeddings_list, payloads_list])
                coll.flush()
        finally:
            coll.release()

        # We no longer store thing_ids nor maintain count here; count can be derived by query when needed
        return {"added": len(int_ids), "group_id": req.group_id}
    except HTTPException:
        raise
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"Failed to add to group: {ex}")


@router.post("/groups/remove")
def remove_from_thing_group(req: RemoveFromGroupRequest) -> Dict[str, Any]:
    """Remove selected things from a group: set group_id to 'unknown' in Milvus and update Mongo doc."""
    try:
        if not req.embedding_ids:
            return {"removed": 0}
        _ensure_mongo_connection()
        g = mongo_find_one("thing_groups", {"_id": ObjectId(req.group_id)})
        if not g:
            raise HTTPException(status_code=404, detail="Group not found")

        # Fetch items by ids
        try:
            int_ids = [int(x) for x in req.embedding_ids]
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid embedding_ids")
        items = _get_things_by_ids(int_ids)
        if not items:
            return {"removed": 0}

        # Upsert Milvus payloads: remove group association (no group_id reliance)
        coll = Collection("things")
        coll.load()
        try:
            ids_list: list[int] = []
            embeddings_list: list[list[float]] = []
            payloads_list: list[str] = []
            for it in items:
                pid = int(it["id"])
                payload = it.get("payload", {})
                if isinstance(payload, str):
                    try:
                        payload = json.loads(payload)
                    except Exception:
                        payload = {"raw": payload}
                if not isinstance(payload, dict):
                    payload = {}
                # Remove any previous group_id if present
                try:
                    if isinstance(payload, dict) and "group_id" in payload:
                        payload.pop("group_id", None)
                except Exception:
                    pass
                ids_list.append(pid)
                embeddings_list.append(it.get("embedding", []))
                payloads_list.append(json.dumps(payload))
            if ids_list:
                # If the 'things' collection has a scalar group_id field, we must provide it as the 4th column
                try:
                    has_group_field = any(getattr(f, 'name', '') == 'group_id' for f in coll.schema.fields)
                except Exception:
                    has_group_field = False
                if has_group_field:
                    group_ids = ["unknown"] * len(ids_list)
                    coll.upsert([ids_list, embeddings_list, payloads_list, group_ids])
                else:
                    coll.upsert([ids_list, embeddings_list, payloads_list])
                coll.flush()
        finally:
            coll.release()

        # No longer store thing_ids/count on the group document
        return {"removed": len(int_ids)}
    except HTTPException:
        raise
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"Failed to remove from group: {ex}")


@router.post("/groups/{group_id}/rename")
def rename_thing_group(group_id: str, req: RenameGroupRequest) -> Dict[str, Any]:
    """Rename a group and update all embeddings in the group with the new label."""
    try:
        if not req.name or not req.name.strip():
            raise HTTPException(status_code=400, detail="Group name cannot be empty")
        
        new_name = req.name.strip()
        _ensure_mongo_connection()
        
        # Check if group exists
        g = mongo_find_one("thing_groups", {"_id": ObjectId(group_id)})
        if not g:
            raise HTTPException(status_code=404, detail="Group not found")
        
        # Update group name in MongoDB
        mongo_update_one("thing_groups", {"_id": ObjectId(group_id)}, {"$set": {"name": new_name}})
        
        # Get all embedding IDs in this group and update their labels
        coll = Collection("things")
        coll.load()
        
        try:
            # Query all items in this group to get their IDs
            expr = f"group_id == \"{group_id}\""
            res = coll.query(expr=expr, output_fields=["id"], limit=16384)
            
            if res:
                # Extract embedding IDs
                embedding_ids = [str(r["id"]) for r in res]
                
                # Use the existing update_things_labels function
                updated_count = update_things_labels(embedding_ids, new_name)
            else:
                updated_count = 0
                
        finally:
            coll.release()
        
        return {
            "id": group_id,
            "name": new_name,
            "updated": updated_count
        }
        
    except HTTPException:
        raise
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"Failed to rename group: {ex}")


@router.delete("/groups/{group_id}")
def delete_thing_group(group_id: str) -> Dict[str, Any]:
    """Delete a group and set all embeddings in that group to ungrouped (group_id = 'unknown')."""
    try:
        _ensure_mongo_connection()
        
        # Check if group exists
        g = mongo_find_one("thing_groups", {"_id": ObjectId(group_id)})
        if not g:
            raise HTTPException(status_code=404, detail="Group not found")
        
        # Get all embedding IDs in this group before deleting
        coll = Collection("things")
        coll.load()
        
        try:
            # Query all items in this group to get their IDs
            expr = f"group_id == \"{group_id}\""
            res = coll.query(expr=expr, output_fields=["id"], limit=16384)
            
            if res:
                # Extract embedding IDs
                embedding_ids = [str(r["id"]) for r in res]
                
                # Query to get current embeddings and payloads
                int_ids = [int(id_str) for id_str in embedding_ids]
                query_expr = f"id in {int_ids}"
                out_fields = ["id", "payload", "embedding"]
                
                # Include group_id when available
                try:
                    if any(getattr(f, 'name', '') == 'group_id' for f in coll.schema.fields):
                        out_fields.append("group_id")
                except Exception:
                    pass
                    
                results = coll.query(expr=query_expr, output_fields=out_fields, limit=len(int_ids))
                
                if results:
                    embeddings_list: List[List[float]] = []
                    payloads_list: List[str] = []
                    ids_list: List[int] = []
                    group_ids_list: List[str] = []
                    
                    for result in results:
                        current_payload = result.get("payload", {})
                        if isinstance(current_payload, str):
                            current_payload = json.loads(current_payload)
                        
                        # Keep the existing label, just update group_id to "unknown"
                        current_payload["group_id"] = "unknown"
                        
                        if "image_id" not in current_payload:
                            current_payload["image_id"] = str(uuid.uuid4())
                        
                        ids_list.append(result["id"])
                        embeddings_list.append(result["embedding"])
                        payloads_list.append(json.dumps(current_payload))
                        group_ids_list.append("unknown")
                    
                    # Perform upsert to update all embeddings
                    if any(getattr(f, 'name', '') == 'group_id' for f in coll.schema.fields):
                        coll.upsert([ids_list, embeddings_list, payloads_list, group_ids_list])
                    else:
                        coll.upsert([ids_list, embeddings_list, payloads_list])
                    coll.flush()
                    
                    updated_count = len(results)
                else:
                    updated_count = 0
            else:
                updated_count = 0
                
        finally:
            coll.release()
        
        # Delete the group document from MongoDB
        mongo_delete_one("thing_groups", {"_id": ObjectId(group_id)})
        
        return {
            "id": group_id,
            "deleted": True,
            "ungrouped_items": updated_count
        }
        
    except HTTPException:
        raise
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"Failed to delete group: {ex}")


