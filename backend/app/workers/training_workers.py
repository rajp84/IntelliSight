from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
import uuid
from io import BytesIO
from queue import Queue, Empty, Full
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from ..utils.media_utils import create_detection_mosaic, split_mosaic_detections
from ..service.detector_service import detect_batch, parse_detection_output, load_detector_model
from ..service.embedder_service import embed_images
from ..database.milvus import ensure_training_collection, insert_training_embeddings
from ..storage.minio_client import put_training_image_bytes, ensure_training_bucket
from ..service.system_service import get_configuration
from ..socket.socket_manager import broadcast
from ..utils.florence_utils import run_caption_task
import re


def extract_phrases_from_caption(caption_text: str, top_k: int = 5) -> str:
    """Extract phrases from Florence-2 caption text for auto discovery."""
    # Remove special tokens like <s>, </s>, <...>
    t = re.sub(r"<[^>]+>", " ", caption_text)
    t = t.replace("</s>", " ").replace("<s>", " ")
    # Split on commas/and/semicolons
    parts = re.split(r"[,;]|\band\b", t, flags=re.IGNORECASE)
    phrases = []
    seen = set()
    for p in parts:
        p = p.strip().lower()
        p = re.sub(r"[^a-z0-9\s-]", "", p)
        p = re.sub(r"\s+", " ", p)
        if len(p) < 2:
            continue
        if p in seen:
            continue
        seen.add(p)
        phrases.append(p)
        if len(phrases) >= top_k:
            break
    return ", ".join(phrases)


def embedding_worker(
    embed_queue: Queue,
    embed_stop: threading.Event,
    training_doc_id: Optional[str],
    embedding_batch_size: int,
    set_status_callback: Callable,
    emit_log_callback: Callable
) -> None:
    """
    Worker function to process embedding queue items.
    
    Args:
        embed_queue: Queue containing items to be embedded
        embed_stop: Event to signal worker to stop
        training_doc_id: Training document ID for database operations
        embedding_batch_size: Batch size for embedding processing
        set_status_callback: Callback to update training status
        emit_log_callback: Callback to emit log messages
    """
    try:
        # Init per-training bucket
        try:
            if training_doc_id is not None:
                ensure_training_bucket(str(training_doc_id))
        except Exception:
            pass

        collection_inited = False
        batch_imgs = []
        batch_meta = []

        while not embed_stop.is_set() or not embed_queue.empty() or batch_imgs:

            # Fill up to embedding_batch_size
            if len(batch_imgs) < embedding_batch_size:
                try:
                    item = embed_queue.get(timeout=0.1)
                    crop_bgr = item.get("crop")
                    if crop_bgr is None:
                        continue

                    # Convert BGR ndarray to PIL RGB
                    pil_img = Image.fromarray(crop_bgr[:, :, ::-1])
                    batch_imgs.append(pil_img)
                    batch_meta.append(item)
                    continue
                except Exception:
                    pass

            if not batch_imgs:
                continue

            # Update queue stats
            try:
                qe_now = embed_queue.qsize()
                qemax_now = getattr(embed_queue, 'maxsize', 0) or 0
                set_status_callback(queue_embed=qe_now, queue_embed_max=qemax_now)
            except Exception:
                pass

            # Get embeddings (DINOv3)
            try:
                embs = embed_images(batch_imgs, batch_size=embedding_batch_size, normalize=True)
            except Exception as _ex:
                try:
                    emit_log_callback(f"Embedder: embedding batch failed: {_ex}")
                except Exception:
                    pass

                batch_imgs.clear()
                batch_meta.clear()
                continue

            # try to init Milvus collection
            if not collection_inited:
                try:
                    if training_doc_id is not None:
                        ensure_training_collection(str(training_doc_id), dim=int(embs.shape[1]))
                        try:
                            emit_log_callback(f"Milvus: initialized training collection training_{training_doc_id} (dim {int(embs.shape[1])})")
                        except Exception:
                            pass
                    collection_inited = True
                except Exception:
                    pass

            # insert                                
            payloads = []
            for i, m in enumerate(batch_meta):
                payload = {
                    "label": m.get("label"),
                    "score": m.get("score"),
                    "bbox": m.get("bbox"),
                    "frame_index": m.get("frame_index"),
                    "track_id": m.get("track_id"),
                    "unique_id": m.get("unique_id"),
                    "image_id": str(uuid.uuid4()),
                }
                payloads.append(json.dumps(payload))
            
            try:
                if training_doc_id is not None:
                    ids = insert_training_embeddings(str(training_doc_id), embs.tolist(), payloads)
                    try:
                        emit_log_callback(f"Milvus: inserted {len(ids)} embeddings into training_{training_doc_id}")
                    except Exception:
                        pass
                else:
                    ids = []
            except Exception as _ex:
                try:
                    emit_log_callback(f"Milvus: insert failed: {_ex}")
                except Exception:
                    pass

                batch_imgs.clear()
                batch_meta.clear()
                continue

            # Upload images to bucket named by image_id from payload
            if ids:
                for idx, (mid, payload) in enumerate(zip(ids, payloads)):
                    try:
                        bio = BytesIO()
                        batch_imgs[idx].save(bio, format="JPEG", quality=90)
                        
                        # Parse payload to get image_id
                        payload_data = json.loads(payload)
                        image_id = payload_data.get("image_id")
                        
                        # Use image_id for filename
                        if image_id:
                            unique_filename = f"{image_id}.jpg"
                        else:
                            unique_filename = f"{training_doc_id}_{mid}.jpg"
                        
                        put_training_image_bytes(str(training_doc_id), unique_filename, bio.getvalue(), content_type="image/jpeg")
                    except Exception as _mx:
                        try:
                            emit_log_callback(f"Failed to upload image for id {mid}: {_mx}")
                        except Exception:
                            pass
                        
            batch_imgs.clear()
            batch_meta.clear()

            # Update queue stats
            try:
                qe_now2 = embed_queue.qsize()
                qemax_now2 = getattr(embed_queue, 'maxsize', 0) or 0
                set_status_callback(queue_embed=qe_now2, queue_embed_max=qemax_now2)
            except Exception:
                pass
    except Exception:
        # just keep going? (TODO: maybe we should stop the training)
        pass


def detection_worker(
    detect_queue: Queue,
    results_map: Dict[int, Tuple[List, List, List]],
    worker_stop: threading.Event,
    eof_event: threading.Event,
    florence_prompt: str,
    max_new_tokens: int,
    detection_debug: bool,
    should_stop_callback: Callable,
    emit_log_callback: Callable,
    auto_discovery: bool = False,
    discovery_interval: int = 90,
    florence_model: str = None,
    dtype_mode: str = None,
    hf_token: str = None
) -> None:
    """
    Worker function to process detection queue items.
    
    Args:
        detect_queue: Queue containing (frame_index, image) tuples
        results_map: Dictionary to store detection results
        worker_stop: Event to signal worker to stop
        eof_event: Event to signal end of video
        florence_prompt: Florence-2 prompt for detection
        max_new_tokens: Maximum new tokens for Florence-2
        detection_debug: Whether to enable debug logging
        should_stop_callback: Callback to check if training should stop
        emit_log_callback: Callback to emit log messages
    """
    try:
        staged: List[Tuple[int, Image.Image]] = []
        
        # Auto discovery state
        discover_phrase_str = ""
        last_discover_refresh_frame = -10**9
        # Use the configurable discovery_interval parameter
        
        while not worker_stop.is_set() or staged or not detect_queue.empty():
            # If termination requested, purge immediately
            if should_stop_callback():
                staged = []
                try:
                    while True:
                        try:
                            _ = detect_queue.get_nowait()
                            detect_queue.task_done()
                        except Empty:
                            break
                except Exception:
                    pass
                break
            
            # If end-of-video, flush any remaining frames without requiring full mosaic
            if eof_event.is_set():
                # clear any remaining queued items
                try:
                    while True:
                        try:
                            item = detect_queue.get_nowait()
                            staged.append(item)
                            detect_queue.task_done()
                        except Empty:
                            break
                except Exception:
                    pass

                if staged:
                    idxs = [i for (i, _) in staged]
                    imgs = [im for (_, im) in staged]
                    # Process in smaller batches to avoid ONNX batch size issues
                    batch_size = 1  # ONNX model expects batch size of 1
                    for i in range(0, len(imgs), batch_size):
                        batch_imgs = imgs[i:i + batch_size]
                        batch_idxs = idxs[i:i + batch_size]
                        outs = detect_batch(batch_imgs, [florence_prompt] * len(batch_imgs), max_new_tokens)
                        for j, od in enumerate(outs):
                            bxs, lbls, scs = parse_detection_output(od)
                            results_map[batch_idxs[j]] = (bxs, lbls, scs)
                    staged = []
                    continue
                else:
                    # Nothing left
                    break

            # Mosaic parameters
            cfg_vals_worker = get_configuration()
            cols = max(1, int(cfg_vals_worker.get("mosaic_cols", 3)))
            cols = min(cols, 5)
            need = cols * cols

            # auto calculate tile scale
            default_scale = 0.5 if cols == 2 else (1.0 / max(1, cols))
            tile_scale = float(cfg_vals_worker.get("mosaic_tile_scale", default_scale))
            tile_scale = max(0.1, min(tile_scale, 1.0))

            # Fill up to cols*cols frames
            while len(staged) < need and not worker_stop.is_set():
                try:
                    item = detect_queue.get(timeout=0.02)
                    staged.append(item)
                except Empty:
                    break

            # if we have enough frames, build mosaic
            if len(staged) >= need:
                group = staged[:need]
                staged = staged[need:]
                idxs = [i for (i, _) in group]
                imgs = [im for (_, im) in group]
                
                # Auto discovery: Run CAPTION every N frames to discover phrases
                if auto_discovery and (idxs[0] - last_discover_refresh_frame) >= discovery_interval:
                    try:
                        # Get the loaded model and processor for caption task
                        processor, model, device = load_detector_model(model_id=florence_model, dtype_mode=dtype_mode, hf_token=hf_token)
                        # Run CAPTION task on the first image in the batch (single frame, not mosaic)
                        caption_text = run_caption_task(model, processor, imgs[0], device, max_new_tokens=64)
                        if caption_text:
                            # Extract phrases from caption
                            discover_phrase_str = extract_phrases_from_caption(caption_text, top_k=5)
                            last_discover_refresh_frame = idxs[0]
                            emit_log_callback(f"[AUTO DISCOVERY] Caption: '{caption_text}'")
                            emit_log_callback(f"[AUTO DISCOVERY] Extracted phrases: '{discover_phrase_str}'")
                            
                            # Convert the frame image to base64 for UI display
                            import base64
                            import io
                            img_buffer = io.BytesIO()
                            imgs[0].save(img_buffer, format='JPEG', quality=85)
                            img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
                            
                            # Emit to UI via socket using emit_log with custom event
                            emit_log_callback(f"TRAINING_CAPTION:{caption_text}|{discover_phrase_str}|{idxs[0]}|{img_base64}", "training_caption")
                    except Exception as e:
                        emit_log_callback(f"[AUTO DISCOVERY] Failed: {e}")
                
                # Create mosaic using the utility function
                mosaic = create_detection_mosaic(imgs, cols, tile_scale)
                
                # Build the prompt for detection
                if auto_discovery and discover_phrase_str:
                    current_prompt = f"<CAPTION_TO_PHRASE_GROUNDING>{discover_phrase_str}"
                else:
                    current_prompt = florence_prompt
                
                # Run detection once on mosaic
                outs = detect_batch([mosaic], [current_prompt], max_new_tokens)
                od = outs[0]
                m_boxes, m_labels, m_scores = parse_detection_output(od)
                if detection_debug:
                    try:
                        boxes_dbg = [(round(b[0],1), round(b[1],1), round(b[2],1), round(b[3],1)) for b in m_boxes][:20]
                        emit_log_callback(f"[DEBUG] mosaic det: batch_idx=0 num_boxes={len(m_boxes)} labels_head={m_labels[:5]} boxes_head={boxes_dbg}")
                    except Exception:
                        emit_log_callback(f"[DEBUG] mosaic det: batch_idx=0 num_boxes={len(m_boxes)} (boxes debug unavailable)")
                
                # Split back to individual tiles and remap to original frame
                original_size = (imgs[0].width, imgs[0].height)
                per_frame_results = split_mosaic_detections(
                    m_boxes, m_labels, m_scores, cols, tile_scale, original_size
                )

                # Store results for each corresponding frame index
                for k in range(need):
                    results_map[idxs[k]] = per_frame_results[k]
                # Mark queue tasks done for the frames
                for _ in range(need):
                    detect_queue.task_done()
                continue

            # If terminating and leftovers remain, process them directly
            if worker_stop.is_set() and staged:
                idxs = [i for (i, _) in staged]
                imgs = [im for (_, im) in staged]
                # Process in smaller batches to avoid ONNX batch size issues
                batch_size = 1  # ONNX model expects batch size of 1
                for i in range(0, len(imgs), batch_size):
                    batch_imgs = imgs[i:i + batch_size]
                    batch_idxs = idxs[i:i + batch_size]
                    
                    # Build the prompt for detection
                    if auto_discovery and discover_phrase_str:
                        current_prompt = f"<CAPTION_TO_PHRASE_GROUNDING>{discover_phrase_str}"
                    else:
                        current_prompt = florence_prompt
                    
                    outs = detect_batch(batch_imgs, [current_prompt] * len(batch_imgs), max_new_tokens)
                    for j, od in enumerate(outs):
                        bxs, lbls, scs = parse_detection_output(od)
                        if detection_debug:
                            try:
                                boxes_dbg = [(round(b[0],1), round(b[1],1), round(b[2],1), round(b[3],1)) for b in bxs][:20]
                                emit_log_callback(f"[DEBUG] leftover det: batch_idx={j} frame_idx={batch_idxs[j]} num_boxes={len(bxs)} boxes_head={boxes_dbg}")
                            except Exception:
                                emit_log_callback(f"[DEBUG] leftover det: batch_idx={j} frame_idx={batch_idxs[j]} num_boxes={len(bxs)}")
                        results_map[batch_idxs[j]] = (bxs, lbls, scs)
                for _ in staged:
                    detect_queue.task_done()
                staged = []
    except Exception as e:
        emit_log_callback(f"Detection worker stopped: {e}")


def detection_prefetcher(
    video_path: str,
    detect_queue: Queue,
    worker_stop: threading.Event,
    total_frames: int,
    detect_every: int,
    target_size: Tuple[int, int],
    original_size: Tuple[int, int],
    emit_log_callback: Callable
) -> None:
    """
    Worker function to prefetch video frames for detection.
    
    Args:
        video_path: Path to the video file
        detect_queue: Queue to put frames into
        worker_stop: Event to signal worker to stop
        total_frames: Total number of frames in video
        detect_every: Process every Nth frame
        target_size: Target size for resizing frames
        original_size: Original video dimensions
        emit_log_callback: Callback to emit log messages
    """
    try:
        cap2 = cv2.VideoCapture(video_path)
        if not cap2.isOpened():
            emit_log_callback("Prefetch: failed to open video")
            return
        
        w, h = original_size
        target_w, target_h = target_size
        
        for idx in range(0, total_frames, detect_every):
            if worker_stop.is_set():
                break
            try:
                cap2.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok, fb = cap2.read()
                if not ok:
                    continue
                if (target_w, target_h) != (w, h):
                    fb = cv2.resize(fb, (target_w, target_h))
                pil = Image.fromarray(fb[:, :, ::-1])
                # Block if queue is full, but remain responsive to stop
                while not worker_stop.is_set():
                    try:
                        detect_queue.put((idx, pil), timeout=0.05)
                        break
                    except Full:
                        continue
            except Exception as e:
                emit_log_callback(f"Prefetch error at frame {idx}: {e}")
                continue
        cap2.release()
        emit_log_callback("Prefetch: completed")
    except Exception as e:
        emit_log_callback(f"Prefetch thread error: {e}")
