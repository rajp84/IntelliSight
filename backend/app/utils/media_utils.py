from __future__ import annotations

from typing import List, Tuple
from PIL import Image, ImageDraw, ImageFont
import base64
from io import BytesIO


def make_mosaic_grid(images: List[Image.Image], cols: int = 3, bg_color=(0, 0, 0)) -> Image.Image:
    if not images:
        raise ValueError("No images provided for mosaic")
    cols = max(1, cols)
    rows = (len(images) + cols - 1) // cols
    # Resize all to the same size (first image's size)
    w, h = images[0].size
    resized = [im.resize((w, h)) for im in images]

    out = Image.new("RGB", (cols * w, rows * h), bg_color)
    for idx, im in enumerate(resized):
        r = idx // cols
        c = idx % cols
        out.paste(im, (c * w, r * h))
    return out


def image_to_base64_jpeg(img: Image.Image, quality: int = 75) -> str:
    buf = BytesIO()
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    img.save(buf, format="JPEG", quality=quality)
    data = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{data}"


def create_detection_mosaic(images: List[Image.Image], cols: int, tile_scale: float) -> Image.Image:
    """
    Create a mosaic for detection processing with scaled tiles.
    
    Args:
        images: List of PIL Images to create mosaic from
        cols: Number of columns in the mosaic grid
        tile_scale: Scale factor for tile size (0.1 to 1.0)
    
    Returns:
        PIL Image containing the mosaic
    """
    if not images:
        raise ValueError("No images provided for mosaic")
    
    cols = max(1, min(cols, 5))  # Limit to reasonable range
    need = cols * cols
    
    # Take only the needed number of images
    images_to_use = images[:need]
    
    # Calculate tile dimensions based on first image and scale
    base_w, base_h = images_to_use[0].width, images_to_use[0].height
    tile_w = max(1, int(base_w * tile_scale))
    tile_h = max(1, int(base_h * tile_scale))
    
    # Resize all images to tile size
    tiles = [im.resize((tile_w, tile_h), Image.BILINEAR) for im in images_to_use]
    
    # Create mosaic
    mosaic = Image.new('RGB', (tile_w * cols, tile_h * cols))
    positions = [(c * tile_w, r * tile_h) for r in range(cols) for c in range(cols)]
    
    for pos, timg in zip(positions, tiles):
        mosaic.paste(timg, pos)
    
    return mosaic


def split_mosaic_detections(
    mosaic_boxes: List[Tuple[float, float, float, float]],
    mosaic_labels: List[str],
    mosaic_scores: List[float],
    cols: int,
    tile_scale: float,
    original_size: Tuple[int, int]
) -> List[Tuple[List[Tuple[float, float, float, float]], List[str], List[float]]]:
    """
    Split detection results from a mosaic back to individual frame detections.
    
    Args:
        mosaic_boxes: List of bounding boxes from mosaic detection
        mosaic_labels: List of labels from mosaic detection
        mosaic_scores: List of scores from mosaic detection
        cols: Number of columns in the mosaic grid
        tile_scale: Scale factor used for tile size
        original_size: (width, height) of original images
    
    Returns:
        List of tuples, each containing (boxes, labels, scores) for one frame
    """
    base_w, base_h = original_size
    tile_w = max(1, int(base_w * tile_scale))
    tile_h = max(1, int(base_h * tile_scale))
    
    # Initialize result arrays for each frame
    per_boxes = [[] for _ in range(cols * cols)]
    per_labels = [[] for _ in range(cols * cols)]
    per_scores = [[] for _ in range(cols * cols)]
    
    # Calculate positions for each tile
    positions = [(c * tile_w, r * tile_h) for r in range(cols) for c in range(cols)]
    
    # Distribute detections to appropriate frames
    for b, l, s in zip(mosaic_boxes, mosaic_labels, mosaic_scores):
        x1, y1, x2, y2 = b
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        col = min(cols - 1, int(cx // tile_w))
        row = min(cols - 1, int(cy // tile_h))
        qi = row * cols + col
        
        if qi < len(per_boxes):  # Safety check
            ox, oy = positions[qi]
            
            # Scale back to original size
            sx = 1.0 / float(tile_scale)
            sy = 1.0 / float(tile_scale)
            rx1 = (x1 - ox) * sx
            ry1 = (y1 - oy) * sy
            rx2 = (x2 - ox) * sx
            ry2 = (y2 - oy) * sy
            
            per_boxes[qi].append((rx1, ry1, rx2, ry2))
            per_labels[qi].append(l)
            per_scores[qi].append(s)
    
    return list(zip(per_boxes, per_labels, per_scores))


def draw_boxes_pil(
    image: Image.Image,
    boxes: List[Tuple[float, float, float, float]],
    labels: List[str],
    scores: List[float],
    color=(0, 255, 0),
    score_thr: float = 0.15,
) -> Image.Image:
    out = image.copy().convert("RGB")
    draw = ImageDraw.Draw(out)
    w, h = out.size
    for i, box in enumerate(boxes):
        if scores and i < len(scores) and scores[i] < score_thr:
            continue
        x1, y1, x2, y2 = box
        x1 = max(0, min(w - 1, int(x1)))
        y1 = max(0, min(h - 1, int(y1)))
        x2 = max(0, min(w - 1, int(x2)))
        y2 = max(0, min(h - 1, int(y2)))
        draw.rectangle([x1, y1, x2, y2], outline=tuple(color), width=2)
        label = labels[i] if i < len(labels) else ""
        score = scores[i] if i < len(scores) else None
        if label or score is not None:
            text = f"{label}" if score is None else f"{label} {score:.2f}"
            # Background box for text
            tw, th = draw.textlength(text), 12
            draw.rectangle([x1, max(0, y1 - th - 4), x1 + tw + 6, y1], fill=tuple(color))
            draw.text((x1 + 3, y1 - th - 2), text, fill=(0, 0, 0))
    return out

