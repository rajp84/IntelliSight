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

