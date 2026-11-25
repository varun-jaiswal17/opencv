"""What the adapters return, and the JSON that the tools send back.

Adapters return the classes below, so that validate.py can compare two engines on the real values.
to_json() turns them into the reply of a tool.
"""

import uuid
from dataclasses import dataclass
from pathlib import Path

import cv2 as cv
import numpy as np


@dataclass
class Detection:
    box: tuple   # x, y, width, height in pixels of the original image
    label: str
    score: float


@dataclass
class Mask:
    mask: np.ndarray   # uint8, 255 inside the object, the size of the original image
    box: tuple         # x, y, width, height around the mask
    score: float = None
    label: str = None


@dataclass
class Document:
    markdown: str
    raw: str           # what the model generated, e.g. DocTags for Granite-Docling


def to_json(task, result, image_path, model, outputs_dir):
    """The reply of a tool. Masks are saved as PNG files and returned as paths, because a mask
    written out as text would be far too long for an AI app to read."""
    if task == "docRead":
        return {"markdown": result.markdown, "raw": result.raw}
    if task == "detect":
        return {"detections": [{"box": [int(v) for v in d.box], "class": d.label, "score": round(float(d.score), 4)}
                               for d in sorted(result, key=lambda d: -d.score)]}
    masks = []
    outputs_dir = Path(outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    call = uuid.uuid4().hex[:8]   # so that a later call doesn't overwrite the files of this one
    for i, m in enumerate(result):
        file = outputs_dir / f"{Path(image_path).stem}_{model}_{call}_{i}.png"
        cv.imwrite(str(file), m.mask)
        item = {"file": str(file), "box": [int(v) for v in m.box], "area": int(np.count_nonzero(m.mask))}
        if m.score is not None:
            item["score"] = round(float(m.score), 4)
        if m.label is not None:
            item["class"] = m.label
        masks.append(item)
    return {"masks": masks}


COLORS = [(0, 200, 0), (255, 120, 0), (0, 0, 255), (0, 200, 255), (255, 0, 200), (200, 200, 0)]


def draw(task, result, image):
    """A copy of the image with the boxes or masks drawn on it, for a tool called with annotate=true.
    Each class name gets its own color."""
    picture = image.copy()
    names = []
    for r in result:
        name = r.label or "object"
        if name not in names:
            names.append(name)
        color = COLORS[names.index(name) % len(COLORS)]
        if task == "segment":
            inside = r.mask > 0
            picture[inside] = (0.45 * picture[inside] + 0.55 * np.array(color)).astype(np.uint8)
        x, y, w, h = (int(v) for v in r.box)
        cv.rectangle(picture, (x, y), (x + w, y + h), color, 2 if task == "detect" else 1)
        text = name if r.score is None else f"{name} {r.score:.2f}"
        (tw, th), _ = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        top = max(y - 2, th + 4)
        cv.rectangle(picture, (x, top - th - 4), (x + tw + 4, top), color, -1)
        cv.putText(picture, text, (x + 2, top - 3), cv.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv.LINE_AA)
    return picture
