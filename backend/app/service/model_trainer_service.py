from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

import albumentations as A
import cv2
from PIL import Image as PILImage
from pymilvus import Collection, utility
from ultralytics import YOLO

from ..database.milvus import is_connected, search_embeddings
from ..database.mongo import find_many
from ..service.system_service import get_configuration
from ..socket.socket_manager import broadcast
from ..storage.minio_client import get_negative_image_bytes, get_object_bytes, list_negatives_objects

logger = logging.getLogger(__name__)
# Regex to strip ANSI escape sequences (colors, cursor controls like \x1b[K)
ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")


class _TrainerState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._proc: Optional[subprocess.Popen] = None
        self._stop_event = threading.Event()
        self._status: Dict[str, Any] = {
            "state": "idle",
            "progress": 0,
            "message": "",
            "epoch": 0,
            "epochs": 0,
            "started_at": None,
            "ended_at": None,
        }

    def is_running(self) -> bool:
        with self._lock:
            return self._proc is not None and self._proc.poll() is None

    def _set_status(self, **kwargs: Any) -> None:
        with self._lock:
            self._status.update(kwargs)

    def status(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._status)

    def request_stop(self) -> None:
        self._stop_event.set()
        # Update status to terminating and broadcast
        try:
            self._set_status(state="terminating", message="Terminate requested")
            asyncio.run(broadcast("model_trainer_status", self.status()))
        except RuntimeError:
            pass
        with self._lock:
            if self._proc and self._proc.poll() is None:
                try:
                    self._proc.send_signal(signal.SIGINT)
                except Exception:
                    pass

    def start(self, *, epochs: int = 100, yolo_model: str = "yolov8n.pt", synth_per_image: int | None = None) -> bool:
        if self.is_running():
            return False

        cfg = get_configuration()
        media_dir = Path(cfg.get("media_directory", "~/ai_media")).expanduser()
        media_dir.mkdir(parents=True, exist_ok=True)

        # Prepare YOLO dataset from things collection images
        work_dir = Path(tempfile.mkdtemp(prefix="yolo_train_"))
        images_dir = work_dir / "images" / "train"
        labels_dir = work_dir / "labels" / "train"
        images_val_dir = work_dir / "images" / "val"
        labels_val_dir = work_dir / "labels" / "val"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        images_val_dir.mkdir(parents=True, exist_ok=True)
        labels_val_dir.mkdir(parents=True, exist_ok=True)
        data_yaml = work_dir / "data.yaml"

        # Export all things images and labels to YOLO format
        try:
            if not is_connected():
                from ..database.connection_manager import init_connections
                asyncio.get_event_loop().run_until_complete(init_connections())
            if not utility.has_collection("things"):
                raise RuntimeError("No things collection to train from")
            coll = Collection("things")
            coll.load()
            # Only include grouped items (classes): group_id != "unknown"
            try:
                has_group_field = any(getattr(f, 'name', '') == 'group_id' for f in coll.schema.fields)
            except Exception:
                has_group_field = False
            expr = 'group_id != "unknown"' if has_group_field else ""
            rows = coll.query(expr=expr, output_fields=["id", "payload"], limit=16384)
            labels = {}
            class_to_id = {}
            next_cls = 0
            crop_index: list[tuple[Path, int]] = []  # (image_path, class_id)
            dataset_items: list[tuple[Path, int, tuple[float, float, float, float]]] = []  # (img_path, class_id, bbox_abs_pascal)
            for r in rows:
                payload = r.get("payload")
                if isinstance(payload, str):
                    payload = json.loads(payload)
                label = (payload or {}).get("label")
                image_id = (payload or {}).get("image_id")
                bbox = (payload or {}).get("bbox")  # expected [x1, y1, x2, y2]
                if not label or not image_id:
                    continue
                if label not in class_to_id:
                    class_to_id[label] = next_cls
                    next_cls += 1
                # Fetch full-frame if available; fallback to crop
                img_bytes = None
                img_name_saved = None
                try:
                    # Try full-frame first
                    img_bytes, _ = get_object_bytes("things", f"{image_id}_full.jpg")
                    img_name_saved = f"{image_id}_full.jpg"
                except Exception:
                    try:
                        # Fallback to crop
                        img_bytes, _ = get_object_bytes("things", f"{image_id}.jpg")
                        img_name_saved = f"{image_id}.jpg"
                    except Exception:
                        continue
                img_path = images_dir / img_name_saved
                with open(img_path, "wb") as f:
                    f.write(img_bytes)
                crop_index.append((img_path, class_to_id[label]))
                # YOLO label generation from bbox if present, else fallback to full image box
                with PILImage.open(img_path) as im:
                    w, h = im.size
                cls_id = class_to_id[label]
                # Label filename must match image base name
                base_name = Path(img_name_saved).stem
                label_path = labels_dir / f"{base_name}.txt"
                # Convert bbox to YOLO normalized format
                def _to_yolo_box(bb, iw, ih):
                    try:
                        x1, y1, x2, y2 = float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])
                        # clamp
                        x1 = max(0.0, min(iw - 1.0, x1))
                        y1 = max(0.0, min(ih - 1.0, y1))
                        x2 = max(0.0, min(iw - 1.0, x2))
                        y2 = max(0.0, min(ih - 1.0, y2))
                        if x2 <= x1 or y2 <= y1:
                            return 0.5, 0.5, 1.0, 1.0
                        bw = (x2 - x1)
                        bh = (y2 - y1)
                        cx = x1 + bw / 2.0
                        cy = y1 + bh / 2.0
                        # normalize
                        return cx / iw, cy / ih, bw / iw, bh / ih
                    except Exception:
                        return 0.5, 0.5, 1.0, 1.0
                if isinstance(bbox, (list, tuple)) and len(bbox) >= 4 and img_name_saved.endswith("_full.jpg"):
                    x_c, y_c, bw_n, bh_n = _to_yolo_box(bbox, w, h)
                else:
                    # No bbox or using crop; assume full image as object
                    x_c, y_c, bw_n, bh_n = 0.5, 0.5, 1.0, 1.0
                with open(label_path, "w") as lf:
                    lf.write(f"{cls_id} {x_c} {y_c} {bw_n} {bh_n}\n")
                # Track item for augmentation; store absolute Pascal VOC bbox
                if isinstance(bbox, (list, tuple)) and len(bbox) >= 4 and img_name_saved.endswith("_full.jpg"):
                    bbox_abs = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
                else:
                    bbox_abs = (0.0, 0.0, float(w), float(h))
                dataset_items.append((img_path, cls_id, bbox_abs))
            # Create validation split using only original images (no synthetic)
            try:
                cfg_vals = get_configuration()
                val_frac = float(cfg_vals.get("yolo_val_fraction", 0.2))
                val_frac = max(0.0, min(0.9, val_frac))
                items = list(dataset_items)
                random.shuffle(items)
                val_count = int(round(len(items) * val_frac)) if len(items) > 0 else 0
                if val_count == 0 and len(items) >= 5:
                    val_count = 1
                val_set = set()
                selected = items[:val_count]
                for (img_path, cls_id, bbox_abs) in selected:
                    try:
                        base = Path(img_path).stem
                        dst_img = images_val_dir / f"{base}.jpg"
                        dst_lbl = labels_val_dir / f"{base}.txt"
                        # Copy image
                        try:
                            shutil.copy2(img_path, dst_img)
                        except Exception:
                            continue
                        # Copy or regenerate label
                        src_lbl = labels_dir / f"{base}.txt"
                        if src_lbl.exists():
                            shutil.copy2(src_lbl, dst_lbl)
                        else:
                            with PILImage.open(dst_img) as imv:
                                vw, vh = imv.size
                            def _to_yolo_box2(bb, iw, ih):
                                try:
                                    x1, y1, x2, y2 = float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])
                                    x1 = max(0.0, min(iw - 1.0, x1)); y1 = max(0.0, min(ih - 1.0, y1))
                                    x2 = max(0.0, min(iw - 1.0, x2)); y2 = max(0.0, min(ih - 1.0, y2))
                                    if x2 <= x1 or y2 <= y1:
                                        return 0.5, 0.5, 1.0, 1.0
                                    bw = (x2 - x1); bh = (y2 - y1)
                                    cx = x1 + bw / 2.0; cy = y1 + bh / 2.0
                                    return cx / iw, cy / ih, bw / iw, bh / ih
                                except Exception:
                                    return 0.5, 0.5, 1.0, 1.0
                            x_c2, y_c2, bw_n2, bh_n2 = _to_yolo_box2(bbox_abs, vw, vh)
                            with open(dst_lbl, "w") as lf:
                                lf.write(f"{cls_id} {x_c2} {y_c2} {bw_n2} {bh_n2}\n")
                        # Remove from train to avoid leakage
                        try:
                            if Path(img_path).exists():
                                Path(img_path).unlink()
                        except Exception:
                            pass
                        try:
                            if src_lbl.exists():
                                src_lbl.unlink()
                        except Exception:
                            pass
                        val_set.add(base)
                    except Exception:
                        continue
                # Filter dataset_items to exclude validation items before augmentation
                if val_set:
                    dataset_items = [t for t in dataset_items if Path(t[0]).stem not in val_set]
                try:
                    asyncio.run(broadcast("model_trainer_log", {"message": f"[MODEL] Validation split: {len(val_set)} images moved to val (fraction={val_frac})"}))
                except RuntimeError:
                    pass
            except Exception:
                try:
                    asyncio.run(broadcast("model_trainer_log", {"message": "[MODEL] Validation split failed; using train as val"}))
                except RuntimeError:
                    pass
            # Include negatives bucket images as background (empty labels)
            negatives_count = 0
            try:
                neg_objects = list_negatives_objects()
                for obj in neg_objects or []:
                    try:
                        data, _ct = get_negative_image_bytes(obj)
                        # Sanitize filename and ensure .jpg extension
                        safe = str(obj).replace('/', '_')
                        base = Path(safe).stem
                        img_name = f"{base}.jpg"
                        img_path = images_dir / img_name
                        with open(img_path, "wb") as f:
                            f.write(data)
                        # Empty label marks as background image
                        label_path = labels_dir / f"{base}.txt"
                        with open(label_path, "w") as lf:
                            lf.write("")
                        negatives_count += 1
                    except Exception:
                        continue
            except Exception:
                negatives_count = 0

            # Before augmentation, compute and broadcast dataset stats
            try:
                # Count current training images on disk (after val split and negatives added)
                train_images_count = 0
                try:
                    train_images_count = len([p for p in images_dir.glob('*.jpg')])
                except Exception:
                    train_images_count = 0
                classes_count = len(class_to_id or {})
                # Synthetic images estimated based on effective per-image setting (UI override wins)
                try:
                    cfg_vals = get_configuration()
                    synth_cfg = int(cfg_vals.get("yolo_synth_per_image", 2))
                except Exception:
                    synth_cfg = 2
                spi_effective = int(synth_per_image) if (synth_per_image is not None) else synth_cfg
                synthetic_total = max(0, spi_effective) * len(dataset_items or [])
                # Attach to status and broadcast
                self._set_status(classes=classes_count, train_images=train_images_count, synthetic_images=synthetic_total, negatives=negatives_count)
                try:
                    asyncio.run(broadcast("model_trainer_status", self.status()))
                except RuntimeError:
                    pass
                try:
                    asyncio.run(broadcast("model_trainer_log", {"message": f"[MODEL] Dataset stats \u2192 classes: {classes_count}, train images: {train_images_count}, negatives: {negatives_count}, synthetic (est): {synthetic_total}"}))
                except RuntimeError:
                    pass
            except Exception:
                pass

            # Albumentations synthetic images
            try:
                cfg_vals = get_configuration()
                synth_cfg = int(cfg_vals.get("yolo_synth_per_image", 2))
                spi = int(synth_per_image) if (synth_per_image is not None) else synth_cfg
                if spi > 0 and dataset_items:
                    try:
                        asyncio.run(broadcast("model_trainer_log", {"message": f"[MODEL] Generating {synth_per_image} augmented images per sample using Albumentations"}))
                    except RuntimeError:
                        pass
                    aug = A.Compose([
                        A.HorizontalFlip(p=0.5),
                        A.VerticalFlip(p=0.1),
                        A.RandomBrightnessContrast(p=0.4),
                        A.HueSaturationValue(p=0.3),
                        A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
                        A.MotionBlur(blur_limit=5, p=0.2),
                        A.Affine(scale=(0.9, 1.1), translate_percent=(0.0, 0.05), rotate=(-10, 10), shear=(-5, 5), p=0.5),
                    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'], min_visibility=0.2))
                    for img_path, cls_id, bbox_abs in dataset_items:
                        try:
                            img_np = cv2.imread(str(img_path))
                            if img_np is None:
                                continue
                            # Ensure bbox within image bounds
                            ih, iw = img_np.shape[:2]
                            x1 = max(0.0, min(iw - 1.0, float(bbox_abs[0])))
                            y1 = max(0.0, min(ih - 1.0, float(bbox_abs[1])))
                            x2 = max(0.0, min(iw - 1.0, float(bbox_abs[2])))
                            y2 = max(0.0, min(ih - 1.0, float(bbox_abs[3])))
                            if x2 <= x1 or y2 <= y1:
                                x1, y1, x2, y2 = 0.0, 0.0, float(iw), float(ih)
                            
                            for k in range(spi):
                                out = aug(image=img_np, bboxes=[(x1, y1, x2, y2)], class_labels=[cls_id])
                                tb = out.get('bboxes') or []
                                if not tb:
                                    continue
                                tx1, ty1, tx2, ty2 = tb[0]
                                out_img = out['image']
                                oh, ow = out_img.shape[:2]
                                bw = max(1.0, float(tx2 - tx1))
                                bh = max(1.0, float(ty2 - ty1))
                                cx = float(tx1 + bw / 2.0)
                                cy = float(ty1 + bh / 2.0)
                                x_c = max(0.0, min(1.0, cx / float(ow)))
                                y_c = max(0.0, min(1.0, cy / float(oh)))
                                bw_n = max(0.0, min(1.0, bw / float(ow)))
                                bh_n = max(0.0, min(1.0, bh / float(oh)))
                                synth_name = f"{Path(img_path).stem}_aug{k+1}.jpg"
                                synth_img_path = images_dir / synth_name
                                cv2.imwrite(str(synth_img_path), out_img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                                synth_label_path = labels_dir / f"{Path(synth_name).stem}.txt"
                                with open(synth_label_path, "w") as lf:
                                    lf.write(f"{cls_id} {x_c} {y_c} {bw_n} {bh_n}\n")
                        except Exception:
                            continue
                else:
                    try:
                        asyncio.run(broadcast("model_trainer_log", {"message": "[MODEL] Synthetic dataset generation skipped (disabled or no items)"}))
                    except RuntimeError:
                        pass
            except Exception:
                try:
                    asyncio.run(broadcast("model_trainer_log", {"message": "[MODEL] Synthetic dataset generation encountered an error; continuing"}))
                except RuntimeError:
                    pass
            # Write data.yaml with absolute paths
            names = [None] * len(class_to_id)
            for name, cid in class_to_id.items():
                names[cid] = name
            with open(data_yaml, "w") as dy:
                dy.write(f"train: {images_dir.as_posix()}\n")
                dy.write(f"val: {images_val_dir.as_posix()}\n")
                dy.write(f"nc: {len(names)}\n")
                dy.write("names:\n")
                for n in names:
                    dy.write(f"  - {n}\n")
            try:
                asyncio.run(broadcast("model_trainer_log", {"message": f"[MODEL] data.yaml written: train -> {images_dir}, val -> {images_val_dir}"}))
            except RuntimeError:
                pass
        except Exception as ex:
            shutil.rmtree(work_dir, ignore_errors=True)
            raise

        # Train with Ultralytics Python API and broadcast detailed logs
        self._stop_event.clear()
        self._set_status(state="running", progress=0, message="Preparing Ultralytics YOLO trainer", epoch=0, epochs=epochs, started_at=time.time(), ended_at=None)
        try:
            asyncio.run(broadcast("model_trainer_status", self.status()))
            asyncio.run(broadcast("model_trainer_log", {"message": f"[MODEL] Dataset prepared at: {work_dir}"}))
            # Resolve model path: if a local file is provided but not found, also check under library_path
            try:
                cfg_vals2 = get_configuration()
                lib_path = Path(cfg_vals2.get("library_path", "/app/media")).expanduser()
            except Exception:
                lib_path = Path("/app/media")
            resolved_model_path = yolo_model
            try:
                p = Path(yolo_model)
                if not (p.exists() and p.is_file()):
                    alt = (lib_path / yolo_model)
                    if alt.exists() and alt.is_file():
                        resolved_model_path = str(alt)
            except Exception:
                pass
            asyncio.run(broadcast("model_trainer_log", {"message": f"[MODEL] Using model: {resolved_model_path}, epochs: {epochs}"}))
        except RuntimeError:
            pass

        def _runner() -> None:
            try:
                # Attach Ultralytics log forwarder to stream logs to UI
                outer_self = self
                class _SocketLogHandler(logging.Handler):
                    def emit(self, record):
                        try:
                            msg = self.format(record)
                            clean = ANSI_ESCAPE_RE.sub('', msg).expandtabs(8).strip()
                            if clean and clean != _recent_last.get("last"):
                                asyncio.run(broadcast("model_trainer_log", {"message": clean}))
                                _recent_last["last"] = clean
                                # Try to parse best metrics from 'all' row and persist in status
                                try:
                                    line = clean
                                    if re.match(r"^\s*all\s+\d+\s+\d+", line):
                                        parts = [p for p in re.split(r"\s{2,}", line.strip()) if p]
                                        if len(parts) >= 7:
                                            # parts: [Class, Images, Instances, Box(P), R, mAP50, mAP50-95]
                                            def _to_f(v: str) -> float:
                                                try:
                                                    return float(v)
                                                except Exception:
                                                    return float('nan')
                                            precision = _to_f(parts[3])
                                            recall = _to_f(parts[4])
                                            map50 = _to_f(parts[5])
                                            map5095 = _to_f(parts[6])
                                            cur_epoch = int(outer_self.status().get('epoch') or 0)
                                            best = outer_self.status().get('best') or None
                                            prev = (best or {}).get('map5095') if isinstance(best, dict) else None
                                            if prev is None or (isinstance(prev, (int, float)) and (map5095 == map5095) and (prev < map5095)):
                                                outer_self._set_status(best={
                                                    'epoch': cur_epoch,
                                                    'precision': precision,
                                                    'recall': recall,
                                                    'map50': map50,
                                                    'map5095': map5095,
                                                })
                                except Exception:
                                    pass
                        except RuntimeError:
                            pass

                ultra_logger = logging.getLogger("ultralytics")
                # Enable INFO logs and attach our handler (we still tee stdout for progress bars)
                ultra_logger.setLevel(logging.INFO)
                _recent_last = {"last": ""}
                _ul_handler = _SocketLogHandler()
                _ul_handler.setLevel(logging.INFO)
                _ul_handler.setFormatter(logging.Formatter("%(message)s"))
                try:
                    ultra_logger.addHandler(_ul_handler)
                except Exception:
                    pass
                # Also tee stdout/stderr to socket to capture progress bars/tables not emitted via logger
                _orig_stdout = sys.stdout
                _orig_stderr = sys.stderr
                class _Tee:
                    def __init__(self, orig, prefix=""):
                        self.orig = orig
                        self.buf = ""
                        self.prefix = prefix
                        self._last = None
                    def write(self, s: str):
                        try:
                            self.orig.write(s)
                        except Exception:
                            pass
                        # Normalize carriage returns, strip ANSI sequences, split on newlines
                        s = ANSI_ESCAPE_RE.sub('', s).replace("\r", "\n")
                        self.buf += s
                        while "\n" in self.buf:
                            line, self.buf = self.buf.split("\n", 1)
                            # Preserve indentation, expand tabs; skip if truly empty after stripping whitespace
                            expanded = line.expandtabs(8)
                            if expanded.strip() == "":
                                continue
                            if expanded != self._last and expanded != _recent_last.get("last"):
                                try:
                                    asyncio.run(broadcast("model_trainer_log", {"message": f"{self.prefix} {expanded.rstrip()}"}))
                                except RuntimeError:
                                    pass
                                self._last = expanded
                                _recent_last["last"] = expanded
                    def flush(self):
                        try:
                            self.orig.flush()
                        except Exception:
                            pass
                sys.stdout = _Tee(_orig_stdout)  # type: ignore[assignment]
                sys.stderr = _Tee(_orig_stderr, prefix="[ULTRA-ERR]")  # type: ignore[assignment]
                # Define callbacks for epoch progress and termination
                last_epoch_ts = {"t": time.time()}
                def _on_epoch_end(trainer):
                    try:
                        cur = int(getattr(trainer, 'epoch', 0))
                        total = int(getattr(trainer, 'epochs', epochs))
                        # Ultralytics epoch is 0-based; display as 1-based, clamped to total
                        disp = min(total, cur + 1) if total > 0 else (cur + 1)
                        pct = int(min(100, max(0, (disp / total) * 100))) if total > 0 else 0
                        # Respect terminate request
                        if self._stop_event.is_set():
                            try:
                                setattr(trainer, 'stop_training', True)
                            except Exception:
                                pass
                            self._set_status(state="cancelled", message="Terminate requested", epoch=disp, epochs=total, progress=pct, ended_at=time.time())
                        else:
                            self._set_status(message=f"Epoch {disp}/{total}", epoch=disp, epochs=total, progress=pct)
                        last_epoch_ts["t"] = time.time()
                        try:
                            asyncio.run(broadcast("model_trainer_status", self.status()))
                        except RuntimeError:
                            pass
                        # Add a blank line separator in logs between epochs for readability
                        try:
                            asyncio.run(broadcast("model_trainer_log", {"message": ""}))
                        except RuntimeError:
                            pass
                    except Exception:
                        pass

                model = YOLO(resolved_model_path)
                # Register callbacks (epoch end and start) to react quickly to terminate
                try:
                    model.add_callback('on_train_epoch_end', _on_epoch_end)
                except Exception:
                    pass
                try:
                    model.add_callback('on_fit_epoch_end', _on_epoch_end)
                except Exception:
                    pass
                def _on_epoch_start(trainer):
                    # Early stop at start of epoch if terminating
                    try:
                        if self._stop_event.is_set():
                            setattr(trainer, 'stop_training', True)
                    except Exception:
                        pass
                for cb_name in ('on_train_epoch_start', 'on_fit_epoch_start'):
                    try:
                        model.add_callback(cb_name, _on_epoch_start)
                    except Exception:
                        pass

                # Also stop mid-epoch as soon as a batch finishes
                def _on_batch_end(trainer):
                    try:
                        if self._stop_event.is_set():
                            try:
                                setattr(trainer, 'stop_training', True)
                            except Exception:
                                pass
                            # Update status immediately to cancelled
                            cur = int(getattr(trainer, 'epoch', 0))
                            total = int(getattr(trainer, 'epochs', epochs))
                            disp = min(total, cur + 1) if total > 0 else (cur + 1)
                            pct = int(min(100, max(0, (disp / total) * 100))) if total > 0 else 0
                            self._set_status(state="cancelled", message="Terminate requested", epoch=disp, epochs=total, progress=pct, ended_at=time.time())
                            try:
                                asyncio.run(broadcast("model_trainer_status", self.status()))
                            except RuntimeError:
                                pass
                    except Exception:
                        pass
                for cb_name in ('on_train_batch_end', 'on_fit_batch_end'):
                    try:
                        model.add_callback(cb_name, _on_batch_end)
                    except Exception:
                        pass

                # Start watchdog to emit periodic heartbeats if stuck in preparation
                def _heartbeat():
                    try:
                        while not self._stop_event.is_set() and (self.status().get("state") == "running"):
                            time.sleep(10.0)
                            now = time.time()
                            if (now - last_epoch_ts.get("t", now)) >= 45.0:
                                # No epoch progress detected in 45s
                                try:
                                    self._set_status(message="Preparing training (downloading weights or compiling ops)...")
                                    asyncio.run(broadcast("model_trainer_status", self.status()))
                                except RuntimeError:
                                    pass
                    except Exception:
                        pass
                hb_thread = threading.Thread(target=_heartbeat, name="model-trainer-heartbeat", daemon=True)
                hb_thread.start()

                # Start training
                try:
                    asyncio.run(broadcast("model_trainer_log", {"message": "[MODEL] Starting training..."}))
                except RuntimeError:
                    pass
                # If terminate was requested before start, exit early
                if self._stop_event.is_set():
                    self._set_status(state="cancelled", message="Terminate requested", ended_at=time.time())
                    try:
                        asyncio.run(broadcast("model_trainer_status", self.status()))
                    except RuntimeError:
                        pass
                    return
                results = model.train(
                    data=str(data_yaml),
                    epochs=int(epochs),
                    imgsz=640,
                    mosaic=1.0,
                    mixup=0.1,
                    copy_paste=0.2,
                    degrees=10.0,
                    translate=0.10,
                    scale=0.50,
                    shear=2.0,
                    perspective=0.0005,
                    hsv_h=0.015,
                    hsv_s=0.70,
                    hsv_v=0.40,
                    fliplr=0.5,
                    flipud=0.0,
                    box=7.5,
                    cls=0.5,
                    project=str(work_dir / 'runs'),
                    name='detect',
                    verbose=True,
                )
                # Locate best weights (prefer trainer.save_dir, fall back to common patterns)
                best_file = None
                save_dir: Path | None = None
                try:
                    trainer = getattr(model, 'trainer', None)
                    if trainer is not None and getattr(trainer, 'save_dir', None):
                        save_dir = Path(trainer.save_dir)
                except Exception:
                    save_dir = None
                # If save_dir not available, try known locations
                if save_dir is None:
                    # 1) runs/detect/weights/best.pt (when name='detect')
                    direct = work_dir / 'runs' / 'detect' / 'weights' / 'best.pt'
                    if direct.exists():
                        best_file = direct
                    else:
                        # 2) any weights/best.pt under runs/**
                        try:
                            candidates = sorted((work_dir / 'runs').rglob('weights/best.pt'), key=lambda p: p.stat().st_mtime, reverse=True)
                            if candidates:
                                best_file = candidates[0]
                        except Exception:
                            pass
                if save_dir is not None and best_file is None:
                    cand = save_dir / 'weights' / 'best.pt'
                    if cand.exists():
                        best_file = cand
                if best_file and best_file.exists():
                    # Save under library_path/model/best_<ts>.pt
                    try:
                        cfg2 = get_configuration()
                    except Exception:
                        cfg2 = {}
                    lib_root = Path(cfg2.get("library_path", str(Path.home()))).expanduser()
                    model_dir = lib_root / "model"
                    model_dir.mkdir(parents=True, exist_ok=True)
                    dest = model_dir / f"best_{int(time.time())}.pt"
                    shutil.copy2(best_file, dest)
                    self._set_status(message=f"Saved best to {dest}")
                    try:
                        asyncio.run(broadcast("model_trainer_log", {"message": f"[MODEL] Best model copied to {dest}"}))
                    except RuntimeError:
                        pass
                else:
                    try:
                        asyncio.run(broadcast("model_trainer_log", {"message": "[MODEL] Warning: best.pt not found in training output"}))
                    except RuntimeError:
                        pass
                    try:
                        asyncio.run(broadcast("model_trainer_log", {"message": f"[MODEL] Saved best to {dest}"}))
                    except RuntimeError:
                        pass
                # Final state if not cancelled
                if self.status().get("state") != "cancelled":
                    self._set_status(state="completed", progress=100, ended_at=time.time())
                    try:
                        asyncio.run(broadcast("model_trainer_status", self.status()))
                    except RuntimeError:
                        pass
            except Exception as ex:
                logger.error(f"[MODEL] Training failed: {ex}")
                try:
                    asyncio.run(broadcast("model_trainer_log", {"message": f"[MODEL] Training failed: {ex}"}))
                except RuntimeError:
                    pass
                self._set_status(state="failed", message=str(ex), ended_at=time.time())
                try:
                    asyncio.run(broadcast("model_trainer_status", self.status()))
                except RuntimeError:
                    pass
            finally:
                # Detach log handler
                try:
                    logging.getLogger("ultralytics").removeHandler(_ul_handler)  # type: ignore[name-defined]
                except Exception:
                    pass
                # Restore stdout/stderr
                try:
                    if '_orig_stdout' in locals():
                        sys.stdout = _orig_stdout  # type: ignore[assignment]
                    if '_orig_stderr' in locals():
                        sys.stderr = _orig_stderr  # type: ignore[assignment]
                except Exception:
                    pass
                # Ensure any heartbeat thread can exit
                try:
                    self._stop_event.set()
                except Exception:
                    pass
                with self._lock:
                    self._proc = None
                try:
                    shutil.rmtree(work_dir, ignore_errors=True)
                except Exception:
                    pass

        t = threading.Thread(target=_runner, name="model-trainer", daemon=True)
        t.start()
        return True


_STATE = _TrainerState()


def start_training(*, epochs: int = 100, yolo_model: str = "yolov8n.pt", synth_per_image: int | None = None) -> bool:
    return _STATE.start(epochs=epochs, yolo_model=yolo_model, synth_per_image=synth_per_image)


def trainer_status() -> Dict[str, Any]:
    return _STATE.status()


def terminate_training() -> None:
    _STATE.request_stop()


