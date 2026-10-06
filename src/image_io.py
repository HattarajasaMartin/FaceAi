from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def read_image(path: str | Path) -> np.ndarray:
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"File tidak ditemukan: {file_path}")
    data = np.fromfile(str(file_path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Gagal membaca gambar: {file_path}")
    return image


def write_image(path: str | Path, image: np.ndarray) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise ValueError(f"Gagal menyimpan gambar: {file_path}")
    encoded.tofile(str(file_path))
