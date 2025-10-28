from fastapi import APIRouter, HTTPException, Query, Response, UploadFile, File, Form
from fastapi.responses import JSONResponse
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

from ...service.training_service import start_od_training, training_status, _STATE
from ...service.system_service import get_configuration
from ...service.embedder_service import embed_image
from ...database.training_repo import list_training_runs, delete_training_record, error_out_running_records, create_training_record, mark_training_status
from ...database.milvus import ensure_training_collection, update_embedding_labels, commit_embeddings_to_things, update_things_labels, delete_selected_things_from_milvus_and_minio, insert_training_embeddings, create_collection
from pymilvus import Collection, utility
from ...storage.minio_client import get_training_image_bytes, training_bucket_name, get_client as get_minio_client, ensure_training_bucket, ensure_things_bucket, get_things_image_bytes, put_training_image_bytes
from ...database.mongo import insert_one as mongo_insert_one, find_many as mongo_find_many, find_one as mongo_find_one, delete_one as mongo_delete_one, delete_many as mongo_delete_many, drop_collection as mongo_drop_collection, update_one as mongo_update_one
from bson import ObjectId  # type: ignore

router = APIRouter()


class StartTrainingRequest(BaseModel):
    mosaic: bool = False
    extra_args: Optional[Dict[str, Any]] = None
    path: str
    task_type: Optional[str] = None
    phrase: Optional[str] = None
    auto_discovery: Optional[bool] = None
    discovery_interval: Optional[int] = None


class UpdateLabelsRequest(BaseModel):
    embedding_ids: list[str]
    label: str

class CommitToThingsRequest(BaseModel):
    embedding_ids: list[str]




@router.post("")
def start_training(req: StartTrainingRequest) -> Dict[str, Any]:
    extra = dict(req.extra_args or {})
    if req.path:
        extra.update({"input": req.path})
    if req.task_type:
        extra.update({"task_type": req.task_type})
    if req.phrase:
        extra.update({"phrase": req.phrase})
    if req.auto_discovery is not None:
        extra.update({"auto_discovery": req.auto_discovery})
    if req.discovery_interval is not None:
        extra.update({"discovery_interval": req.discovery_interval})
    ok = start_od_training(mosaic=req.mosaic, extra_args=extra)
    if not ok:
        raise HTTPException(status_code=409, detail="A training job is already running")
    return {"status": "started"}


# -------- Manual annotation helpers --------
class ManualStartResponse(BaseModel):
    run_id: str


@router.post("/manual/start")
def manual_start(path: str = Query(..., description="File path relative to library root")) -> Dict[str, Any]:
    """Create a training record and collection for manual annotation of a video, return run_id."""
    cfg = get_configuration()
    root_str = cfg.get("library_path")
    if not root_str:
        raise HTTPException(status_code=400, detail="library_path is not configured")
    root = Path(root_str).expanduser().resolve()
    video_path = (root / path).resolve()
    # Enforce within root
    try:
        video_path.relative_to(root)
    except ValueError:
        raise HTTPException(status_code=400, detail="Path is outside of library root")
    if not video_path.exists() or not video_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    # Probe video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="Failed to open video")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    # Create training record
    try:
        # Use configured embedding dimension when creating the collection
        try:
            dinov3_dimension = int(cfg.get("dinov3_dimension", 768))
        except Exception:
            dinov3_dimension = 768
        rec_id = create_training_record(
            file_path=str(video_path),
            file_name=video_path.name,
            fps=fps,
            width=w,
            height=h,
            total_frames=total,
            training_params={"manual": True}
        )
        ensure_training_collection(str(rec_id), dim=dinov3_dimension)
        return {"run_id": str(rec_id), "fps": fps, "total_frames": total, "width": w, "height": h}
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"Failed to start manual run: {ex}")


@router.get("/manual/metadata")
def manual_metadata(path: str = Query(..., description="File path relative to library root")) -> Dict[str, Any]:
    """Return basic video metadata: fps, total frames, width, height."""
    cfg = get_configuration()
    root_str = cfg.get("library_path")
    if not root_str:
        raise HTTPException(status_code=400, detail="library_path is not configured")
    root = Path(root_str).expanduser().resolve()
    video_path = (root / path).resolve()
    try:
        video_path.relative_to(root)
    except ValueError:
        raise HTTPException(status_code=400, detail="Path is outside of library root")
    if not video_path.exists() or not video_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="Failed to open video")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    return {"fps": fps, "total_frames": total, "width": w, "height": h}


@router.post("/manual/crop")
def manual_crop(
    image: UploadFile = File(...),
    path: str = Form(...),
    frame_index: int = Form(...),
    bbox: str = Form(...),
    label: str = Form(...),
    run_id: str | None = Form(default=None),
) -> Dict[str, Any]:
    """Accept a user-labeled crop image, embed it, and insert into current manual run collection.

    A run is implicitly created if not present by calling manual_start.
    """
    try:

        cfg = get_configuration()
        root_str = cfg.get("library_path")
        if not root_str:
            raise HTTPException(status_code=400, detail="library_path is not configured")
        root = Path(root_str).expanduser().resolve()
        video_path = (root / path).resolve()
        try:
            video_path.relative_to(root)
        except ValueError:
            raise HTTPException(status_code=400, detail="Path is outside of library root")
        if not video_path.exists():
            raise HTTPException(status_code=404, detail="File not found")

        # Determine or create a manual run for this path
        if not run_id:
            rec_id = create_training_record(
                file_path=str(video_path),
                file_name=video_path.name,
                fps=None,
                width=0,
                height=0,
                total_frames=0,
                training_params={"manual": True}
            )
            run_id = str(rec_id)
            # Dimension will be validated below after computing embedding
        else:
            # Will validate dimension below
            pass

        # Read image bytes
        img_bytes = image.file.read()
        pil = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        # Embed
        emb = embed_image(pil)
        emb_list = emb.reshape(1, -1).tolist()
        emb_dim = len(emb_list[0]) if emb_list and len(emb_list[0]) > 0 else 0
        if emb_dim <= 0:
            raise HTTPException(status_code=500, detail="Invalid embedding vector")

        # Ensure collection exists and matches embedding dimension; recreate if mismatched
        try:
            coll_name = f"training_{run_id}"
            if utility.has_collection(coll_name):
                coll = Collection(coll_name)
                # find embedding field and its dim
                dim_found = None
                for field in coll.schema.fields:
                    if getattr(field, 'name', '') == 'embedding':
                        dim_found = (field.params or {}).get('dim')
                        break
                if dim_found and int(dim_found) != int(emb_dim):
                    # Drop and recreate with correct dim
                    utility.drop_collection(coll_name)
                    create_collection(coll_name, dim=emb_dim, description=f"Embeddings for training run {run_id}")
            else:
                # Create fresh
                create_collection(coll_name, dim=emb_dim, description=f"Embeddings for training run {run_id}")
        except Exception as mm_ex:
            # As a fallback try ensure with emb_dim
            try:
                ensure_training_collection(run_id, dim=emb_dim)
            except Exception:
                raise HTTPException(status_code=500, detail=f"Failed to ensure training collection: {mm_ex}")
        # Parse bbox JSON (pixels) for later editing as [x1, y1, x2, y2]
        bbox_list = None
        try:
            bbox_obj = json.loads(bbox) if bbox else None
            if isinstance(bbox_obj, dict):
                x1 = int(round(float(bbox_obj.get("x1", 0))))
                y1 = int(round(float(bbox_obj.get("y1", 0))))
                x2 = int(round(float(bbox_obj.get("x2", 0))))
                y2 = int(round(float(bbox_obj.get("y2", 0))))
                bbox_list = [x1, y1, x2, y2]
            elif isinstance(bbox_obj, (list, tuple)) and len(bbox_obj) == 4:
                bbox_list = [int(round(float(v))) for v in bbox_obj]
        except Exception:
            bbox_list = None

        payload = {
            "label": label,
            "frame_index": int(frame_index),
            "source_path": str(path),
            **({"bbox": bbox_list} if bbox_list else {}),
        }
        # Store image to MinIO and include image_id in payload
        image_id = f"{int(time.time()*1000)}"
        put_training_image_bytes(run_id, f"{image_id}.jpg", img_bytes, "image/jpeg")
        payload["image_id"] = image_id
        # Also save the full frame for later crop edits
        try:
            cap2 = cv2.VideoCapture(str(video_path))
            try:
                # Seek to requested frame index if supported
                try:
                    cap2.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
                except Exception:
                    pass
                ok2, frame_bgr = cap2.read()
                if ok2 and frame_bgr is not None:
                    ok_enc, enc = cv2.imencode('.jpg', frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                    if ok_enc:
                        full_bytes = bytes(enc)
                        put_training_image_bytes(run_id, f"{image_id}_full.jpg", full_bytes, "image/jpeg")
                        #payload["full_image_id"] = f"{image_id}_full"
            finally:
                cap2.release()
        except Exception:
            # Saving full frame is best-effort; continue if it fails
            pass
        ids = insert_training_embeddings(run_id, embeddings=emb_list, payloads=[json.dumps(payload)])
        return {"run_id": run_id, "inserted": len(ids), "embedding_ids": ids}
    except HTTPException:
        raise
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"Failed to accept manual crop: {ex}")


@router.post("/manual/complete")
def manual_complete(run_id: str = Query(...)) -> Dict[str, Any]:
    """Mark the manual training run as completed so it shows in results page like others."""
    try:
        try:
            rid = ObjectId(run_id)
        except Exception:
            rid = run_id
        mark_training_status(rid, "completed")
        return {"status": "ok"}
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"Failed to complete run: {ex}")


@router.get("/status")
def get_training_status() -> Dict[str, Any]:
    return training_status()


@router.post("/terminate")
def terminate_training() -> Dict[str, Any]:
    _STATE.request_stop()
    return {"status": "terminating"}


@router.get("/runs")
def get_training_runs(page: int = Query(1, ge=1), page_size: int = Query(20, ge=1, le=200)) -> Dict[str, Any]:
    return list_training_runs(page=page, page_size=page_size)


def cleanup_training_run(run_id: str) -> Dict[str, Any]:
    """Comprehensive cleanup of a training run: MongoDB document, Milvus collection, and MinIO bucket."""
    cleanup_results = {
        "mongodb_deleted": False,
        "milvus_collection_dropped": False,
        "minio_bucket_removed": False,
        "errors": []
    }
    
    try:
        # 1. Delete MongoDB document
        try:
            cleanup_results["mongodb_deleted"] = delete_training_record(run_id)
            print(f"[DEBUG] MongoDB cleanup for {run_id}: {cleanup_results['mongodb_deleted']}")
        except Exception as e:
            cleanup_results["errors"].append(f"MongoDB cleanup failed: {e}")
            print(f"[ERROR] MongoDB cleanup failed for {run_id}: {e}")
        
        # 2. Drop Milvus collection
        try:
            collection_name = f"training_{run_id}"
            if utility.has_collection(collection_name):
                collection = Collection(collection_name)
                collection.drop()
                cleanup_results["milvus_collection_dropped"] = True
                print(f"[DEBUG] Dropped Milvus collection: {collection_name}")
            else:
                print(f"[DEBUG] Milvus collection {collection_name} does not exist")
        except Exception as e:
            cleanup_results["errors"].append(f"Milvus collection cleanup failed: {e}")
            print(f"[ERROR] Milvus collection cleanup failed for {run_id}: {e}")
        
        # 3. Remove MinIO bucket
        try:
            bucket_name = training_bucket_name(run_id)
            minio_client = get_minio_client()
            
            if minio_client.bucket_exists(bucket_name):
                # List and remove all objects first
                objects = minio_client.list_objects(bucket_name, recursive=True)
                for obj in objects:
                    minio_client.remove_object(bucket_name, obj.object_name)
                
                # Remove the bucket
                minio_client.remove_bucket(bucket_name)
                cleanup_results["minio_bucket_removed"] = True
                print(f"[DEBUG] Removed MinIO bucket: {bucket_name}")
            else:
                print(f"[DEBUG] MinIO bucket {bucket_name} does not exist")
        except Exception as e:
            cleanup_results["errors"].append(f"MinIO bucket cleanup failed: {e}")
            print(f"[ERROR] MinIO bucket cleanup failed for {run_id}: {e}")
        
        return cleanup_results
        
    except Exception as e:
        cleanup_results["errors"].append(f"General cleanup error: {e}")
        print(f"[ERROR] General cleanup error for {run_id}: {e}")
        return cleanup_results


@router.delete("/runs/{run_id}")
def remove_training_run(run_id: str) -> Dict[str, Any]:
    """Delete a training run with comprehensive cleanup."""
    try:
        cleanup_results = cleanup_training_run(run_id)
        
        # Check if at least MongoDB deletion succeeded
        if not cleanup_results["mongodb_deleted"]:
            raise HTTPException(status_code=404, detail="Training run not found or could not be deleted")
        
        # Return cleanup results
        return {
            "deleted": True,
            "cleanup_results": cleanup_results
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete training run: {e}")


@router.post("/reconcile")
def reconcile_running_records() -> Dict[str, Any]:
    """Mark any 'running' records as 'error'. Useful when UI detects no active training."""
    try:
        count = error_out_running_records("Reconciled: no active training")
        return {"updated": int(count)}
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"Reconcile failed: {ex}")


@router.get("/runs/{run_id}/embeddings")
def list_run_embeddings(run_id: str, limit: int = Query(200, ge=1, le=10000), offset: int = Query(0, ge=0)) -> Dict[str, Any]:
    """List embeddings for a training run from Milvus, returning id and payload (parsed)."""
    try:
        name = f"training_{run_id}"
        # Ensure exists; will raise if cannot
        coll = Collection(name)
    except Exception:
        try:
            coll = ensure_training_collection(run_id, dim=768)
        except Exception as ex:
            raise HTTPException(status_code=404, detail=f"Collection for run {run_id} not found: {ex}")
    try:
        # Load the collection before querying
        coll.load()
        
        # Query all data first, then sort by frame number and apply pagination
        # This ensures consistent ordering even after updates
        res = coll.query(expr="", output_fields=["id", "payload"], limit=16384)  # type: ignore
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
        
        # Sort by frame number (frame_index in payload)
        def get_frame_index(item):
            payload = item.get("payload", {})
            if isinstance(payload, dict):
                return payload.get("frame_index", 0)
            return 0
        
        items.sort(key=get_frame_index)
        
        # Apply pagination after sorting
        start_idx = offset
        end_idx = offset + limit
        paginated_items = items[start_idx:end_idx]
        
        return {"items": paginated_items}
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"Query failed: {ex}")
    finally:
        # Release the collection after use
        try:
            coll.release()
        except Exception:
            pass  # Ignore release errors


@router.get("/runs/{run_id}/image/{image_id}")
def get_run_image(run_id: str, image_id: str):
    """Serve a stored crop image from MinIO by training run and image id (e.g., 123.jpg)."""
    try:
        # Backward compatible: if extension not provided, default to .jpg
        object_name = image_id if (image_id.lower().endswith('.jpg') or image_id.lower().endswith('.jpeg')) else f"{image_id}.jpg"
        data, content_type = get_training_image_bytes(run_id, object_name)
        return Response(content=data, media_type=content_type)
    except Exception as ex:
        raise HTTPException(status_code=404, detail=f"Image not found: {ex}")


@router.get("/runs/{run_id}/images")
def list_run_images(run_id: str) -> Dict[str, Any]:
    """List object names in the per-training bucket to help debug/image retrieval."""
    try:
        # Ensure bucket exists (no-op if already there)
        bucket = ensure_training_bucket(run_id)
        client = get_minio_client()
        objects = client.list_objects(bucket, recursive=True)
        names = [obj.object_name for obj in objects]
        return {"bucket": bucket, "objects": names}
    except Exception as ex:
        # Return empty list rather than error when bucket truly doesn't exist
        if "NoSuchBucket" in str(ex):
            return {"bucket": training_bucket_name(run_id), "objects": []}
        raise HTTPException(status_code=500, detail=f"List images failed: {ex}")


@router.post("/runs/{run_id}/embeddings/labels")
def update_embedding_labels_endpoint(run_id: str, req: UpdateLabelsRequest) -> Dict[str, Any]:
    """Update labels for multiple embeddings in Milvus."""
    try:
        updated_count = update_embedding_labels(run_id, req.embedding_ids, req.label)
        return {"updated": updated_count}
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"Failed to update labels: {ex}")

@router.post("/runs/{run_id}/embeddings/commit-to-things")
def commit_to_things_endpoint(run_id: str, req: CommitToThingsRequest) -> Dict[str, Any]:
    """Commit selected embeddings to the 'things' collection."""
    try:
        committed_count = commit_embeddings_to_things(run_id, req.embedding_ids)
        return {"committed": committed_count}
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"Failed to commit to things: {ex}")
