from __future__ import annotations

import cv2
import numpy as np

from .config import FERY_STICKER, MEGAWATI_STICKER, MULYADI_STICKER, ensure_data_dirs
from .detect import detect_faces
from .domain import FaceHit
from .image_io import read_image

_crops: dict[str, np.ndarray] = {}

STICKER_FILES = {
    "mulyadi": MULYADI_STICKER,
    "megawati": MEGAWATI_STICKER,
    "fery": FERY_STICKER,
}


def _prepare_crop(key: str) -> np.ndarray | None:
    if key in _crops:
        return _crops[key]
    path = STICKER_FILES.get(key)
    ensure_data_dirs()
    if path is None or not path.is_file():
        return None
    image = read_image(path)
    faces = detect_faces(image)
    if faces:
        x, y, width, height = faces[0].box
        pad_x = int(width * 0.18)
        pad_y = int(height * 0.22)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(image.shape[1], x + width + pad_x)
        y2 = min(image.shape[0], y + height + pad_y)
        crop = image[y1:y2, x1:x2]
    else:
        crop = image
    _crops[key] = crop
    return crop


def paste_sticker(frame: np.ndarray, face: FaceHit, key: str = "mulyadi") -> np.ndarray:
    source = _prepare_crop(key)
    if source is None or source.size == 0:
        return frame
    x, y, width, height = face.box
    pad_x = int(width * 0.22)
    pad_y = int(height * 0.28)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(frame.shape[1], x + width + pad_x)
    y2 = min(frame.shape[0], y + height + pad_y)
    roi_w, roi_h = x2 - x1, y2 - y1
    if roi_w < 20 or roi_h < 20:
        return frame
    sticker = cv2.resize(source, (roi_w, roi_h), interpolation=cv2.INTER_LINEAR)
    mask = np.zeros((roi_h, roi_w), dtype=np.float32)
    cv2.ellipse(
        mask,
        (roi_w // 2, roi_h // 2),
        (max(1, roi_w // 2 - 2), max(1, roi_h // 2 - 2)),
        0,
        0,
        360,
        1.0,
        -1,
    )
    mask = cv2.GaussianBlur(mask, (31, 31), 0)
    mask = np.clip(mask, 0, 1)
    roi = frame[y1:y2, x1:x2].astype(np.float32)
    blended = sticker.astype(np.float32) * mask[..., None] + roi * (1.0 - mask[..., None])
    frame[y1:y2, x1:x2] = blended.astype(np.uint8)
    return frame
