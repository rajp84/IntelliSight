from __future__ import annotations

import os
import io
import json
import uuid
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import asyncio

from PIL import Image

from .system_service import get_configuration
from ..database.training_repo import create_training_record, mark_training_status, update_training_record
from ..database.milvus import ensure_training_collection, insert_training_embeddings
from ..storage.minio_client import ensure_training_bucket, put_training_image_bytes
from .embedder_service import embed_images, preload_model
from ..socket.socket_manager import broadcast
from fastapi import HTTPException


logger = logging.getLogger(__name__)


def _read_yolo_names(dataset_root: Path) -> List[str]:
    # Try data.yaml under root or one level up
    candidates = [dataset_root / "data.yaml", dataset_root.parent / "data.yaml"]
    for fp in candidates:
        try:
            if fp.exists():
                txt = fp.read_text(encoding="utf-8", errors="ignore")
                # naive parse for names: [..]
                import re
                m = re.search(r"names\s*:\s*\[([^\]]+)\]", txt)
                if m:
                    inner = m.group(1)
                    parts = [p.strip().strip("'\"") for p in inner.split(",")]
                    return [p for p in parts if p]
                # fallback parse yaml if available
                try:
                    import yaml  # type: ignore
                    data = yaml.safe_load(txt)
                    names = data.get("names") if isinstance(data, dict) else None
                    if isinstance(names, dict):
                        # mapping {id: name}
                        out: List[str] = []
                        for k in sorted(names.keys(), key=lambda x: int(x) if str(x).isdigit() else x):
                            out.append(str(names[k]))
                        return out
                    if isinstance(names, list):
                        return [str(x) for x in names]
                except Exception:
                    pass
        except Exception:
            continue
    return []


def _iter_yolo_images(dataset_dir: Path, split: str) -> List[Tuple[Path, Optional[Path]]]:
    # Expect structure: <dataset_dir>/<split>/images/*.jpg and matching labels/*.txt
    imgs_dir = dataset_dir / split / "images"
    labels_dir = dataset_dir / split / "labels"
    items: List[Tuple[Path, Optional[Path]]] = []
    if not imgs_dir.exists():
        # fallback to direct images/labels under dataset_dir
        imgs_dir = dataset_dir / "images"
        labels_dir = dataset_dir / "labels"
    if not imgs_dir.exists():
        return []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
        for img_path in imgs_dir.glob(ext):
            lab = None
            if labels_dir.exists():
                cand = labels_dir / (img_path.stem + ".txt")
                if cand.exists():
                    lab = cand
            items.append((img_path, lab))
    return items


def _parse_yolo_label_file(label_path: Path, img_w: int, img_h: int) -> List[Tuple[int, float, float, float, float]]:
    # Returns list of (class_id, x1, y1, x2, y2) in pixel coords
    out: List[Tuple[int, float, float, float, float]] = []
    try:
        for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            cls_id = int(float(parts[0]))
            xc = float(parts[1]); yc = float(parts[2]); ww = float(parts[3]); hh = float(parts[4])
            # YOLO coords are normalized [0,1]
            x1 = (xc - ww / 2.0) * img_w
            y1 = (yc - hh / 2.0) * img_h
            x2 = (xc + ww / 2.0) * img_w
            y2 = (yc + hh / 2.0) * img_h
            # clamp
            x1 = max(0.0, min(img_w - 1.0, x1))
            y1 = max(0.0, min(img_h - 1.0, y1))
            x2 = max(0.0, min(img_w - 1.0, x2))
            y2 = max(0.0, min(img_h - 1.0, y2))
            if x2 > x1 and y2 > y1:
                out.append((cls_id, x1, y1, x2, y2))
    except Exception:
        pass
    return out


def import_dataset_as_training(dataset_path: str, *, split: str = "train", label_format: str = "yolov8") -> Dict[str, Any]:
    """
    Import a downloaded dataset folder as a training run.

    Steps:
    - Create Mongo training record
    - Create Milvus training collection
    - Create MinIO training bucket
    - For each image, upload full frame and crop(s) to MinIO, compute crop embeddings, insert into Milvus with bbox payload
    - Mark training run completed
    """
    try:
        root = Path(dataset_path)
        if not root.exists() or not root.is_dir():
            raise HTTPException(status_code=400, detail="Dataset path not found or not a directory")

        cfg = get_configuration()
        dinov3_model = cfg.get("dinov3_model", "facebook/dinov3-vitb16-pretrain-lvd1689m")
        dinov3_dimension = int(cfg.get("dinov3_dimension", 768))

        # Preload embedder (best effort)
        try:
            preload_model(model_id=dinov3_model)
        except Exception:
            pass

        # Build file listing
        images: List[Tuple[Path, Optional[Path]]] = []
        if (label_format or "").lower().startswith("yolo"):
            images = _iter_yolo_images(root, split)
            class_names = _read_yolo_names(root)
        else:
            images = _iter_yolo_images(root, split)
            class_names = []
        total_imgs = len(images)
        if total_imgs == 0:
            raise HTTPException(status_code=400, detail="No images found in dataset")

        # Create training record
        run_id = create_training_record(
            file_path=str(root),
            file_name=root.name,
            fps=None,
            width=0,
            height=0,
            total_frames=total_imgs,
            training_params={
                "source": "roboflow",
                "format": label_format,
                "split": split,
            },
            status="running",
        )

        # Initial broadcast
        try:
            asyncio.run(broadcast("dataset_import_status", {
                "training_id": str(run_id),
                "dataset_path": str(root),
                "processed": 0,
                "total": int(total_imgs),
                "message": "Import started"
            }))
        except Exception:
            pass

        # Ensure bucket and collection
        ensure_training_bucket(str(run_id))
        ensure_training_collection(str(run_id), dim=dinov3_dimension)

        # Iterate images and accumulate crops for batched embedding insert
        batch_crops: List[Image.Image] = []
        batch_payloads: List[Dict[str, Any]] = []
        batch_full_frames: List[bytes] = []
        processed = 0

        def flush_batch() -> None:
            nonlocal batch_crops, batch_payloads, batch_full_frames
            if not batch_crops:
                return
            try:
                embs = embed_images(batch_crops, batch_size=16, normalize=True)
            except Exception as e:
                logger.warning("Embedding batch failed: %s", e)
                batch_crops = []
                batch_payloads = []
                batch_full_frames = []
                return
            payload_json = [json.dumps(p) for p in batch_payloads]
            ids = insert_training_embeddings(str(run_id), embs.tolist(), payload_json)
            # Upload crops and full frames by payload image_id
            for idx, pay in enumerate(batch_payloads):
                try:
                    image_id = pay.get("image_id") or str(uuid.uuid4())
                    bio = io.BytesIO()
                    batch_crops[idx].save(bio, format="JPEG", quality=90)
                    put_training_image_bytes(str(run_id), f"{image_id}.jpg", bio.getvalue(), content_type="image/jpeg")
                    # Full frame
                    fb = batch_full_frames[idx]
                    if fb:
                        put_training_image_bytes(str(run_id), f"{image_id}_full.jpg", fb, content_type="image/jpeg")
                except Exception as upx:
                    logger.warning("Failed uploading crop/full: %s", upx)
                    continue
            batch_crops = []
            batch_payloads = []
            batch_full_frames = []
            # Batch-complete console + socket update will be handled by outer loop progress update

        for img_path, lbl_path in images:
            try:
                with Image.open(img_path) as im:
                    im_rgb = im.convert("RGB")
                    w, h = im_rgb.size
                    # Serialize full frame once
                    full_bio = io.BytesIO()
                    im_rgb.save(full_bio, format="JPEG", quality=90)
                    full_bytes = full_bio.getvalue()

                    boxes: List[Tuple[int, float, float, float, float]] = []
                    if lbl_path and lbl_path.exists():
                        boxes = _parse_yolo_label_file(lbl_path, w, h)
                    if not boxes:
                        # Skip images without boxes
                        processed += 1
                        if processed % 20 == 0:
                            try:
                                update_training_record(run_id, set_fields={"frames_processed": int(processed)})
                            except Exception:
                                pass
                        continue

                    for (cls_id, x1, y1, x2, y2) in boxes:
                        # Crop
                        crop = im_rgb.crop((int(x1), int(y1), int(x2), int(y2)))
                        label_txt = None
                        try:
                            if 0 <= int(cls_id) < len(class_names):
                                label_txt = class_names[int(cls_id)]
                        except Exception:
                            pass
                        label_txt = label_txt or f"class_{cls_id}"
                        image_id = str(uuid.uuid4())
                        payload = {
                            "label": label_txt,
                            "score": 1.0,
                            "bbox": [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))],
                            "frame_index": processed,
                            "track_id": None,
                            "unique_id": str(uuid.uuid4()),
                            "image_id": image_id,
                        }
                        batch_crops.append(crop)
                        batch_payloads.append(payload)
                        batch_full_frames.append(full_bytes)

                    # Periodic flush to control memory
                    if len(batch_crops) >= 64:
                        flush_batch()
                        # Broadcast after each batch flush
                        try:
                            asyncio.run(broadcast("dataset_import_status", {
                                "training_id": str(run_id),
                                "dataset_path": str(root),
                                "processed": int(processed),
                                "total": int(total_imgs),
                                "message": f"Processed {processed}/{total_imgs} images"
                            }))
                        except Exception:
                            pass

                    processed += 1
                    if processed % 20 == 0:
                        try:
                            update_training_record(run_id, set_fields={"frames_processed": int(processed)})
                        except Exception:
                            pass
                        # Log and broadcast periodic progress
                        logger.info("Dataset import progress: %s/%s", processed, total_imgs)
                        try:
                            asyncio.run(broadcast("dataset_import_status", {
                                "training_id": str(run_id),
                                "dataset_path": str(root),
                                "processed": int(processed),
                                "total": int(total_imgs),
                                "message": f"Processed {processed}/{total_imgs} images"
                            }))
                        except Exception:
                            pass
            except Exception as e:
                logger.warning("Failed processing %s: %s", img_path, e)
                continue

        # Final flush
        flush_batch()

        try:
            mark_training_status(run_id, "completed", frames_processed=int(processed))
        except Exception:
            pass

        # Final broadcast
        try:
            asyncio.run(broadcast("dataset_import_status", {
                "training_id": str(run_id),
                "dataset_path": str(root),
                "processed": int(processed),
                "total": int(total_imgs),
                "message": "Import completed"
            }))
        except Exception:
            pass

        return {"training_id": str(run_id), "processed": int(processed)}
    except HTTPException:
        raise
    except Exception as ex:
        logger.exception("Dataset import failed: %s", ex)
        raise HTTPException(status_code=500, detail=f"Failed to import dataset: {ex}")


