from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    MilvusException,
    connections,
    utility,
)

# Import MinIO functions for image copying
from app.storage.minio_client import ensure_things_bucket, put_image_bytes, get_object_bytes


def connect(alias: str = "default", host: Optional[str] = None, port: Optional[str] = None) -> None:
    """
    Establish a connection to Milvus.

    Uses MILVUS_HOST and MILVUS_PORT env vars if not provided, defaulting to localhost:19530.
    """
    host = host or os.getenv("MILVUS_HOST", "127.0.0.1")
    port = port or os.getenv("MILVUS_PORT", "19530")
    if not connections.has_connection(alias):
        connections.connect(alias=alias, host=host, port=port)


def is_connected(alias: str = "default") -> bool:
    return connections.has_connection(alias)


def disconnect(alias: str = "default") -> None:
    if connections.has_connection(alias):
        connections.disconnect(alias)


def list_collections() -> List[str]:
    return utility.list_collections()


def drop_collection(collection_name: str) -> None:
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)


def create_collection(
    collection_name: str,
    dim: int,
    index_params: Optional[Dict[str, Any]] = None,
    description: str = "",
    auto_id: bool = True,
    metric_type: str = "L2",
    index_type: str = "IVF_FLAT",
    shards_num: int = 2,
) -> Collection:
    """
    Create a simple collection with an auto-increment primary key `id` (int64),
    a vector field `embedding` with the provided dimension, and an optional
    string `payload` field for arbitrary metadata.
    """
    if utility.has_collection(collection_name):
        return Collection(collection_name)

    id_field = FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=auto_id)
    vector_field = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim)
    payload_field = FieldSchema(name="payload", dtype=DataType.VARCHAR, max_length=2048)

    schema = CollectionSchema(
        fields=[id_field, vector_field, payload_field],
        description=description,
    )

    collection = Collection(name=collection_name, schema=schema, shards_num=shards_num)

    # Create index on the vector field for efficient search
    index_params = index_params or {
        "index_type": index_type,
        "metric_type": metric_type,
        "params": {"nlist": 1024},
    }
    collection.create_index(field_name="embedding", index_params=index_params)

    # Load the collection into memory for search/insert
    collection.load()
    return collection


def insert_entries(
    collection_name: str,
    embeddings: Sequence[Sequence[float]],
    payloads: Optional[Sequence[str]] = None,
) -> List[int]:
    """
    Insert entries into a collection.

    If the primary key is auto_id=True, omit ids and Milvus assigns them.
    Returns the list of assigned ids.
    """
    if not connections.has_connection("default"):
        raise RuntimeError("Milvus connection not initialized. Ensure connection_manager.init_connections() has run.")
    collection = Collection(collection_name)
    num = len(embeddings)
    if payloads is None:
        payloads = [""] * num
    if len(payloads) != num:
        raise ValueError("payloads length must match embeddings length")

    # Data order must match fields in schema excluding auto-id
    mr = collection.insert([embeddings, payloads])
    collection.flush()
    return list(mr.primary_keys)


def upsert_entries(
    collection_name: str,
    ids: Sequence[int],
    embeddings: Sequence[Sequence[float]],
    payloads: Optional[Sequence[str]] = None,
) -> List[int]:
    """
    Upsert entries with explicit ids (requires auto_id=False in schema).
    """
    if not connections.has_connection("default"):
        raise RuntimeError("Milvus connection not initialized. Ensure connection_manager.init_connections() has run.")
    collection = Collection(collection_name)
    num = len(embeddings)
    if len(ids) != num:
        raise ValueError("ids length must match embeddings length")
    if payloads is None:
        payloads = [""] * num
    if len(payloads) != num:
        raise ValueError("payloads length must match embeddings length")

    mr = collection.upsert([list(ids), embeddings, payloads])
    collection.flush()
    return list(mr.primary_keys)


def delete_by_ids(collection_name: str, ids: Sequence[int]) -> int:
    if not connections.has_connection("default"):
        raise RuntimeError("Milvus connection not initialized. Ensure connection_manager.init_connections() has run.")
    collection = Collection(collection_name)
    expr = f"id in {list(ids)}"
    res = collection.delete(expr)
    collection.flush()
    return res.delete_count


def query_by_ids(collection_name: str, ids: Sequence[int], output_fields: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    if not connections.has_connection("default"):
        raise RuntimeError("Milvus connection not initialized. Ensure connection_manager.init_connections() has run.")
    collection = Collection(collection_name)
    expr = f"id in {list(ids)}"
    return collection.query(expr=expr, output_fields=output_fields or ["id", "payload"])


def search_embeddings(
    collection_name: str,
    query_embeddings: Sequence[Sequence[float]],
    top_k: int = 5,
    metric_type: str = "L2",
    nprobe: int = 16,
    output_fields: Optional[List[str]] = None,
):
    if not connections.has_connection("default"):
        raise RuntimeError("Milvus connection not initialized. Ensure connection_manager.init_connections() has run.")
    collection = Collection(collection_name)
    search_params = {"metric_type": metric_type, "params": {"nprobe": nprobe}}
    return collection.search(
        data=query_embeddings,
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        output_fields=output_fields or ["id", "payload"],
    )


__all__ = [
    "connect",
    "disconnect",
    "list_collections",
    "create_collection",
    "drop_collection",
    "insert_entries",
    "upsert_entries",
    "delete_by_ids",
    "query_by_ids",
    "search_embeddings",
]


def ensure_training_collection(training_id: str, dim: int = 768, *, metric: str = "L2") -> Collection:
    """Create (if needed) and return a Milvus collection for a specific training run.

    Collection name: training_<training_id>
    Schema: id (int64, auto), embedding (float_vector dim), payload (varchar)
    """
    if not connections.has_connection("default"):
        raise RuntimeError("Milvus connection not initialized. Ensure connection_manager.init_connections() has run.")
    name = f"training_{training_id}"
    return create_collection(
        collection_name=name,
        dim=dim,
        description=f"Embeddings for training run {training_id}",
        metric_type=metric,
        index_type="IVF_FLAT",
    )


def insert_training_embeddings(training_id: str, embeddings: Sequence[Sequence[float]], payloads: Optional[Sequence[str]] = None) -> List[int]:
    name = f"training_{training_id}"
    return insert_entries(name, embeddings, payloads)


def _update_labels_in_collection(collection_name: str, embedding_ids: List[str], new_label: str, use_upsert: bool = False) -> int:
    """
    Update labels for multiple embeddings in a Milvus collection.
    
    Args:
        collection_name: Name of the Milvus collection
        embedding_ids: List of Milvus IDs to update
        new_label: The new label to set
        use_upsert: If True, use upsert (for collections with explicit IDs). 
                   If False, use delete+insert (for collections with auto_id=True)
        
    Returns:
        Number of embeddings successfully updated
    """
    if not is_connected():
        raise RuntimeError("Milvus connection not initialized. Ensure connection_manager.init_connections() has run.")
    
    # Check if collection exists
    if not utility.has_collection(collection_name):
        raise ValueError(f"Collection {collection_name} does not exist")
    
    collection = Collection(collection_name)
    collection.load()
    
    try:
        # Convert string IDs to integers (Milvus IDs are integers)
        int_ids = [int(id_str) for id_str in embedding_ids]
        
        # Query to get current embeddings and payloads
        query_expr = f"id in {int_ids}"
        results = collection.query(
            expr=query_expr,
            output_fields=["id", "payload", "embedding"],
            limit=len(int_ids)
        )
        
        if not results:
            print(f"[DEBUG] No embeddings found for IDs: {int_ids}")
            return 0
        
        print(f"[DEBUG] Found {len(results)} embeddings to update in {collection_name}")
        
        # Prepare data for update
        import json
        import uuid
        
        if use_upsert:
            # Use upsert approach (for things collection)
            embeddings_list = []
            payloads_list = []
            ids_list = []
            
            for result in results:
                current_payload = result.get("payload", {})
                if isinstance(current_payload, str):
                    current_payload = json.loads(current_payload)
                
                # Update the label
                current_payload["label"] = new_label
                
                # Generate stable image_id if not present
                if "image_id" not in current_payload:
                    current_payload["image_id"] = str(uuid.uuid4())
                
                # Use the same ID for upsert
                ids_list.append(result["id"])
                embeddings_list.append(result["embedding"])
                payloads_list.append(json.dumps(current_payload))
            
            # Upsert the updated records
            collection.upsert([ids_list, embeddings_list, payloads_list])
            print(f"[DEBUG] Updated {len(results)} embeddings in {collection_name} with label: '{new_label}'")
            
        else:
            # Use delete+insert approach (for training collections with auto_id=True)
            updated_items = []
            
            for result in results:
                current_payload = result.get("payload", {})
                if isinstance(current_payload, str):
                    current_payload = json.loads(current_payload)
                
                print(f"[DEBUG] Original payload for ID {result['id']}: {current_payload}")
                
                # Update the label
                current_payload["label"] = new_label
                
                # Prepare the updated item
                updated_item = {
                    "id": result["id"],
                    "embedding": result["embedding"],
                    "payload": json.dumps(current_payload)
                }
                
                print(f"[DEBUG] Updated item for ID {result['id']}: {updated_item}")
                updated_items.append(updated_item)
            
            if updated_items:
                print(f"[DEBUG] Updating {len(updated_items)} embeddings with new label: {new_label}")
                
                # Extract IDs for deletion
                ids_to_delete = [item["id"] for item in updated_items]
                
                # Delete existing entries
                print(f"[DEBUG] Deleting {len(ids_to_delete)} existing entries")
                collection.delete(f"id in {ids_to_delete}")
                
                # Prepare insert data with stable image IDs
                embeddings_list = []
                payloads_list = []
                
                for item in updated_items:
                    # Parse existing payload
                    current_payload = item["payload"]
                    if isinstance(current_payload, str):
                        current_payload = json.loads(current_payload)
                    
                    # Generate or preserve stable image ID
                    if "image_id" not in current_payload:
                        current_payload["image_id"] = str(uuid.uuid4())
                        print(f"[DEBUG] Generated new image_id: {current_payload['image_id']}")
                    else:
                        print(f"[DEBUG] Preserving existing image_id: {current_payload['image_id']}")
                    
                    # Update the label
                    current_payload["label"] = new_label
                    
                    embeddings_list.append(item["embedding"])
                    payloads_list.append(json.dumps(current_payload))
                
                # Insert updated entries (Milvus will assign new IDs)
                print(f"[DEBUG] Inserting {len(embeddings_list)} updated entries")
                insert_result = collection.insert([embeddings_list, payloads_list])
                new_ids = list(insert_result.primary_keys)
                
                # Flush to ensure data is persisted
                collection.flush()
                
                # Verify the update by querying the new items
                print(f"[DEBUG] Verifying update by querying new items...")
                verify_results = collection.query(
                    expr=f"id in {new_ids}",
                    output_fields=["id", "payload"],
                    limit=len(new_ids)
                )
                print(f"[DEBUG] Verification query returned {len(verify_results)} items")
                for verify_result in verify_results:
                    verify_payload = verify_result.get("payload", {})
                    if isinstance(verify_payload, str):
                        verify_payload = json.loads(verify_payload)
                    print(f"[DEBUG] Verified payload for ID {verify_result['id']}: {verify_payload}")
                
                print(f"[DEBUG] Successfully updated {len(updated_items)} embeddings using delete+insert (new IDs: {new_ids})")
        
        return len(results)
        
    except Exception as e:
        print(f"[ERROR] Failed to update labels in {collection_name}: {e}")
        raise RuntimeError(f"Failed to update labels in {collection_name}: {e}")
    finally:
        collection.release()


def update_embedding_labels(training_id: str, embedding_ids: List[str], new_label: str) -> int:
    """
    Update the label field in the payload for multiple embeddings in a training collection.
    
    Args:
        training_id: The training run ID (used to construct collection name)
        embedding_ids: List of Milvus IDs to update
        new_label: The new label to set
        
    Returns:
        Number of embeddings successfully updated
    """
    collection_name = f"training_{training_id}"
    return _update_labels_in_collection(collection_name, embedding_ids, new_label, use_upsert=False)


def commit_embeddings_to_things(training_id: str, embedding_ids: List[str]) -> int:
    """
    Copy selected embeddings from a training collection to the 'things' collection.
    If the 'things' collection doesn't exist, create it.
    
    Args:
        training_id: The training run ID (used to construct source collection name)
        embedding_ids: List of Milvus IDs to copy
        
    Returns:
        Number of embeddings successfully committed
    """
    if not is_connected():
        raise RuntimeError("Milvus connection not initialized. Ensure connection_manager.init_connections() has run.")
    
    source_collection_name = f"training_{training_id}"
    things_collection_name = "things"
    
    # Check if source collection exists
    if not utility.has_collection(source_collection_name):
        raise ValueError(f"Source collection {source_collection_name} does not exist")
    
    source_collection = Collection(source_collection_name)
    source_collection.load()
    
    try:
        # Get the dimension from the source collection schema
        source_schema = source_collection.schema
        embedding_field = None
        for field in source_schema.fields:
            if field.name == "embedding":
                embedding_field = field
                break
        
        if not embedding_field:
            raise ValueError(f"Source collection {source_collection_name} does not have an embedding field")
        
        source_dim = embedding_field.params.get("dim")
        if not source_dim:
            raise ValueError(f"Source collection {source_collection_name} embedding field does not have dimension specified")
        
        print(f"[DEBUG] Source collection dimension: {source_dim}")
        
        # Convert string IDs to integers (Milvus IDs are integers)
        int_ids = [int(id_str) for id_str in embedding_ids]
        
        # Query to get embeddings and payloads for the specified IDs
        query_expr = f"id in {int_ids}"
        results = source_collection.query(
            expr=query_expr,
            output_fields=["id", "payload", "embedding"],
            limit=len(int_ids)
        )
        
        if not results:
            print(f"[DEBUG] No embeddings found for IDs: {int_ids}")
            return 0
        
        print(f"[DEBUG] Found {len(results)} embeddings to commit to things collection")
        
        # Ensure 'things' collection exists with matching dimension
        if not utility.has_collection(things_collection_name):
            print(f"[DEBUG] Creating 'things' collection with dimension {source_dim}")
            things_collection = create_collection(
                collection_name=things_collection_name,
                dim=source_dim,  # Use dimension from source collection
                description="User's personal things library"
            )
        else:
            # Check if existing things collection has the correct dimension
            things_collection = Collection(things_collection_name)
            things_schema = things_collection.schema
            things_embedding_field = None
            for field in things_schema.fields:
                if field.name == "embedding":
                    things_embedding_field = field
                    break
            
            if things_embedding_field:
                things_dim = things_embedding_field.params.get("dim")
                if things_dim != source_dim:
                    raise ValueError(
                        f"Things collection dimension mismatch: {things_dim} vs {source_dim}. "
                        f"Please use the 'Delete All' functionality on the Things page to clear the existing collection "
                        f"and try again. This ensures you don't lose any existing things data accidentally."
                    )
            else:
                raise ValueError(
                    f"Things collection does not have a proper embedding field. "
                    f"Please use the 'Delete All' functionality on the Things page to clear the existing collection "
                    f"and try again."
                )
        
        things_collection.load()
        
        # Prepare data for insertion into things collection
        import json
        import uuid
        
        embeddings_list = []
        payloads_list = []
        
        for result in results:
            current_payload = result.get("payload", {})
            if isinstance(current_payload, str):
                current_payload = json.loads(current_payload)
            
            # Add committed flag and preserve original data
            current_payload["committed"] = True
            current_payload["source_training_id"] = training_id
            current_payload["original_id"] = result["id"]
            
            # Generate new image_id for things collection
            current_payload["image_id"] = str(uuid.uuid4())
            
            embeddings_list.append(result["embedding"])
            payloads_list.append(json.dumps(current_payload))
            
            print(f"[DEBUG] Prepared embedding for things collection: {current_payload}")
        
        # Insert into things collection
        print(f"[DEBUG] Inserting {len(embeddings_list)} embeddings into things collection")
        insert_result = things_collection.insert([embeddings_list, payloads_list])
        things_collection.flush()
        
        # Copy images to things bucket
        print(f"[DEBUG] Copying images to things bucket")
        ensure_things_bucket()
        
        print(f"[DEBUG] Source bucket: training_{training_id}")
        print(f"[DEBUG] Target bucket: things")
        
        for i, (result, payload) in enumerate(zip(results, payloads_list)):
            try:
                print(f"[DEBUG] Processing image {i+1}/{len(results)} for result {result['id']}")
                
                # Get the original image from training bucket
                source_bucket = f"training-{training_id}"
                
                # Get the original image_id from the source payload
                original_payload = result.get("payload", {})
                if isinstance(original_payload, str):
                    original_payload = json.loads(original_payload)
                
                original_image_id = original_payload.get("image_id")
                if not original_image_id:
                    print(f"[WARNING] No image_id found for result {result['id']}, skipping image copy")
                    continue
                
                source_filename = f"{original_image_id}.jpg"  # Use original image_id
                print(f"[DEBUG] Source filename: {source_filename}")
                
                # Get image data from source bucket
                image_data, _ = get_object_bytes(source_bucket, source_filename)
                print(f"[DEBUG] Retrieved {len(image_data)} bytes from source")
                
                # Store in things bucket with new filename (same image_id)
                payload_data = json.loads(payload)
                things_filename = f"{payload_data['image_id']}.jpg"
                print(f"[DEBUG] Target filename: {things_filename}")
                
                put_image_bytes("things", things_filename, image_data, "image/jpeg")
                
                print(f"[DEBUG] Successfully copied image: {source_filename} -> {things_filename}")
            except Exception as img_ex:
                print(f"[ERROR] Failed to copy image for ID {result['id']}: {img_ex}")
                import traceback
                traceback.print_exc()
                # Continue with other images even if one fails
        
        # Update source collection to mark as committed
        print(f"[DEBUG] Updating source collection to mark items as committed")
        for result in results:
            current_payload = result.get("payload", {})
            if isinstance(current_payload, str):
                current_payload = json.loads(current_payload)
            
            current_payload["committed"] = True
            
            # Update the source collection
            source_collection.delete(f"id == {result['id']}")
            source_collection.insert([[result["embedding"]], [json.dumps(current_payload)]])
        
        source_collection.flush()
        
        committed_count = len(embeddings_list)
        print(f"[DEBUG] Successfully committed {committed_count} embeddings to things collection")
        
        return committed_count
        
    except Exception as e:
        print(f"[ERROR] Failed to commit embeddings to things: {e}")
        raise RuntimeError(f"Failed to commit embeddings to things: {e}")
    finally:
        source_collection.release()
        if 'things_collection' in locals():
            things_collection.release()


def update_things_labels(embedding_ids: List[str], new_label: str) -> int:
    """
    Update labels for multiple embeddings in the 'things' collection.
    
    Args:
        embedding_ids: List of Milvus IDs to update
        new_label: New label to set
        
    Returns:
        Number of embeddings successfully updated
    """
    return _update_labels_in_collection("things", embedding_ids, new_label, use_upsert=True)


def delete_selected_things_from_milvus_and_minio(embedding_ids: List[str]) -> int:
    """
    Delete selected embeddings from the 'things' collection and their images from MinIO.
    
    Args:
        embedding_ids: List of Milvus IDs to delete
        
    Returns:
        Number of embeddings successfully deleted
    """
    if not is_connected():
        raise RuntimeError("Milvus connection not initialized. Ensure connection_manager.init_connections() has run.")
    
    collection_name = "things"
    
    # Check if collection exists
    if not utility.has_collection(collection_name):
        raise ValueError(f"Collection {collection_name} does not exist")
    
    collection = Collection(collection_name)
    collection.load()
    
    try:
        # Convert string IDs to integers (Milvus IDs are integers)
        int_ids = [int(id_str) for id_str in embedding_ids]
        
        # Query to get payloads to extract image IDs for MinIO deletion
        query_expr = f"id in {int_ids}"
        results = collection.query(
            expr=query_expr,
            output_fields=["id", "payload"],
            limit=len(int_ids)
        )
        
        if not results:
            print(f"[DEBUG] No embeddings found for IDs: {int_ids}")
            return 0
        
        print(f"[DEBUG] Found {len(results)} embeddings to delete from {collection_name}")
        
        # Extract image IDs for MinIO deletion
        import json
        from app.storage.minio_client import get_client as get_minio_client, things_bucket_name
        
        image_ids_to_delete = []
        for result in results:
            current_payload = result.get("payload", {})
            if isinstance(current_payload, str):
                current_payload = json.loads(current_payload)
            
            # Get image_id from payload
            image_id = current_payload.get("image_id")
            if image_id:
                image_ids_to_delete.append(image_id)
                print(f"[DEBUG] Will delete image: {image_id}")
        
        # Delete from Milvus
        print(f"[DEBUG] Deleting {len(int_ids)} embeddings from Milvus")
        collection.delete(f"id in {int_ids}")
        collection.flush()
        
        # Delete images from MinIO
        if image_ids_to_delete:
            minio_client = get_minio_client()
            bucket_name = things_bucket_name()
            
            print(f"[DEBUG] Deleting {len(image_ids_to_delete)} images from MinIO bucket: {bucket_name}")
            for image_id in image_ids_to_delete:
                try:
                    minio_client.remove_object(bucket_name, f"{image_id}.jpg")
                    print(f"[DEBUG] Deleted image: {image_id}.jpg")
                except Exception as e:
                    print(f"[WARNING] Failed to delete image {image_id}.jpg from MinIO: {e}")
        
        print(f"[DEBUG] Successfully deleted {len(results)} embeddings and {len(image_ids_to_delete)} images")
        return len(results)
        
    except Exception as e:
        print(f"[ERROR] Failed to delete selected things: {e}")
        raise RuntimeError(f"Failed to delete selected things: {e}")
    finally:
        collection.release()


__all__ += [
    "ensure_training_collection",
    "insert_training_embeddings",
    "update_embedding_labels",
    "commit_embeddings_to_things",
    "update_things_labels",
    "delete_selected_things_from_milvus_and_minio",
]



