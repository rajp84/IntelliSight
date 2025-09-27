from fastapi import APIRouter, HTTPException, Query, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Any, Dict, Optional

from ...service.training_service import start_od_training, training_status, _STATE
from ...database.training_repo import list_training_runs, delete_training_record, error_out_running_records
from ...database.milvus import ensure_training_collection, update_embedding_labels, commit_embeddings_to_things, update_things_labels, delete_selected_things_from_milvus_and_minio
from pymilvus import Collection, utility
import json
from ...storage.minio_client import get_training_image_bytes, training_bucket_name, get_client as get_minio_client, ensure_training_bucket, ensure_things_bucket, get_things_image_bytes

router = APIRouter()


class StartTrainingRequest(BaseModel):
    mosaic: bool = False
    extra_args: Optional[Dict[str, Any]] = None
    path: str
    task_type: Optional[str] = None
    phrase: Optional[str] = None


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
    ok = start_od_training(mosaic=req.mosaic, extra_args=extra)
    if not ok:
        raise HTTPException(status_code=409, detail="A training job is already running")
    return {"status": "started"}


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
        data, content_type = get_training_image_bytes(run_id, image_id)
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

@router.post("/things/labels")
def update_things_labels_endpoint(req: UpdateLabelsRequest) -> Dict[str, Any]:
    """Update labels for multiple embeddings in the 'things' collection."""
    try:
        updated_count = update_things_labels(req.embedding_ids, req.label)
        return {"updated": updated_count}
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"Failed to update things labels: {ex}")

@router.delete("/things/selected")
def delete_selected_things(req: CommitToThingsRequest) -> Dict[str, Any]:
    """Delete selected embeddings from the 'things' collection and their images from MinIO."""
    try:
        deleted_count = delete_selected_things_from_milvus_and_minio(req.embedding_ids)
        return {"deleted": deleted_count}
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"Failed to delete selected things: {ex}")

@router.get("/things/search")
def search_things_embeddings(
    query: str = Query(..., description="Search query for label field"),
    limit: int = Query(200, ge=1, le=10000), 
    offset: int = Query(0, ge=0)
) -> Dict[str, Any]:
    """Search embeddings in the 'things' collection by label."""
    try:
        from pymilvus import Collection, utility
        
        collection_name = "things"
        
        # Check if collection exists
        if not utility.has_collection(collection_name):
            return {"items": [], "total": 0}
        
        collection = Collection(collection_name)
        collection.load()
        
        try:
            # Query all data first, then filter by label and apply pagination
            res = collection.query(expr="", output_fields=["id", "payload"], limit=16384)  # type: ignore
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
            
            # Apply pagination after sorting
            start_idx = offset
            end_idx = offset + limit
            paginated_items = items[start_idx:end_idx]
            
            return {"items": paginated_items, "total": len(items)}
        finally:
            collection.release()
            
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"Failed to search things embeddings: {ex}")

@router.get("/things")
def get_things_embeddings(limit: int = Query(200, ge=1, le=10000), offset: int = Query(0, ge=0)) -> Dict[str, Any]:
    """Get all embeddings from the 'things' collection."""
    try:
        from pymilvus import Collection, utility
        
        collection_name = "things"
        
        # Check if collection exists
        if not utility.has_collection(collection_name):
            return {"items": [], "total": 0}
        
        collection = Collection(collection_name)
        collection.load()
        
        try:
            # Query all data first, then sort by frame number and apply pagination
            res = collection.query(expr="", output_fields=["id", "payload"], limit=16384)  # type: ignore
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
            
            # Apply pagination after sorting
            start_idx = offset
            end_idx = offset + limit
            paginated_items = items[start_idx:end_idx]
            
            return {"items": paginated_items, "total": len(items)}
        finally:
            collection.release()
            
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"Failed to get things embeddings: {ex}")

@router.get("/things/image/{filename}")
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

@router.delete("/things")
def delete_all_things() -> Dict[str, Any]:
    """Delete all things from the 'things' collection and bucket."""
    try:
        from pymilvus import utility
        from ...storage.minio_client import get_client as get_minio_client
        
        collection_name = "things"
        bucket_name = "things"
        
        # Check if collection exists
        if not utility.has_collection(collection_name):
            return {"message": "Things collection does not exist", "deleted": 0}
        
        # Get collection info before deletion
        collection = Collection(collection_name)
        collection.load()
        total_count = collection.num_entities
        
        # Drop the collection
        utility.drop_collection(collection_name)
        
        # Delete all objects from the things bucket
        minio_client = get_minio_client()
        try:
            # List all objects in the things bucket
            objects = minio_client.list_objects(bucket_name, recursive=True)
            object_count = 0
            for obj in objects:
                minio_client.remove_object(bucket_name, obj.object_name)
                object_count += 1
            
            # Remove the bucket itself
            minio_client.remove_bucket(bucket_name)
        except Exception as bucket_ex:
            # If bucket doesn't exist or is already empty, that's fine
            print(f"[INFO] Things bucket cleanup: {bucket_ex}")
        
        return {
            "message": f"Successfully deleted all things",
            "deleted_embeddings": total_count,
            "deleted_images": object_count
        }
        
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"Failed to delete all things: {ex}")

