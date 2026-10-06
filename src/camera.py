from __future__ import annotations

import cv2

from .config import CAMERA_INDEX


class CameraError(RuntimeError):
    """Webcam tidak tersedia atau sedang dipakai aplikasi lain."""


def _try_open(index: int) -> cv2.VideoCapture | None:
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        return None
    return cap


def _has_picture(cap: cv2.VideoCapture) -> bool:
    frame = None
    ok = False
    for _ in range(8):
        ok, frame = cap.read()
    if not ok or frame is None or frame.size == 0:
        return False
    return float(frame.mean()) > 5.0


def open_camera(index: int = CAMERA_INDEX) -> cv2.VideoCapture:
    seen: list[int] = []
    for candidate in (index, 0, 1):
        if candidate in seen:
            continue
        seen.append(candidate)
        cap = _try_open(candidate)
        if cap is None:
            continue
        if _has_picture(cap):
            return cap
        cap.release()
    raise CameraError(
        "Webcam tidak bisa dibuka. Tutup aplikasi lain yang memakai kamera, lalu jalankan lagi."
    )
