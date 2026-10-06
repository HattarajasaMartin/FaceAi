from __future__ import annotations

import cv2
import numpy as np

from .config import BLUR_REJECT, DARK_REJECT, DARK_WARN, MAX_ABS_YAW, MIN_FACE_SIZE
from .domain import FaceHit, QualityReport


def face_yaw(face: FaceHit) -> float:
    """Perkiraan belok kiri/kanan dari landmark YuNet (mata + hidung)."""
    right_eye_x = float(face.raw[4])
    left_eye_x = float(face.raw[6])
    nose_x = float(face.raw[8])
    eye_span = abs(left_eye_x - right_eye_x) + 1e-6
    midpoint = (left_eye_x + right_eye_x) / 2.0
    return (nose_x - midpoint) / eye_span


def _crop(image: np.ndarray, face: FaceHit) -> np.ndarray:
    x, y, width, height = face.box
    return image[y : y + height, x : x + width]


def assess_quality(image: np.ndarray, face: FaceHit) -> QualityReport:
    x, y, width, height = face.box
    warnings: list[str] = []
    errors: list[str] = []
    yaw = face_yaw(face)

    if min(width, height) < MIN_FACE_SIZE:
        errors.append("Wajah terlalu kecil, mendekat ke kamera")

    if abs(yaw) > MAX_ABS_YAW:
        errors.append("Wajah terlalu miring, hadap kamera")

    crop = _crop(image, face)
    if crop.size == 0:
        errors.append("Gagal membaca area wajah")
        return QualityReport(False, yaw, 0.0, 0.0, tuple(warnings), tuple(errors))

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    if brightness < DARK_REJECT:
        errors.append("Cahaya terlalu gelap")
    elif brightness < DARK_WARN:
        warnings.append("Cahaya gelap, tambah penerangan")

    if sharpness < BLUR_REJECT:
        errors.append("Gambar buram, jangan bergerak")

    return QualityReport(
        ok=not errors,
        yaw=yaw,
        brightness=brightness,
        sharpness=sharpness,
        warnings=tuple(warnings),
        errors=tuple(errors),
    )


def liveness_from_yaw(yaws: list[float], min_range: float) -> tuple[bool, str]:
    if len(yaws) < 2:
        return False, "Gerakkan kepala sedikit kiri-kanan"
    if max(yaws) - min(yaws) < min_range:
        return False, "Gerakkan kepala sedikit kiri-kanan"
    return True, "Liveness OK"
