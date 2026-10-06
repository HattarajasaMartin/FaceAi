from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .detect import detect_result, extract_embedding, face_count_message
from .domain import FaceHit, QualityReport
from .quality import assess_quality


@dataclass(frozen=True)
class FaceSample:
    face: FaceHit
    embedding: np.ndarray
    quality: QualityReport


def analyze_image(image: np.ndarray) -> tuple[FaceSample | None, str]:
    detection = detect_result(image)
    ok, message = face_count_message(detection)
    if not ok:
        return None, message
    face = detection.faces[0]
    quality = assess_quality(image, face)
    if quality.errors:
        return None, quality.errors[0]
    embedding = extract_embedding(image, face)
    return FaceSample(face=face, embedding=embedding, quality=quality), message


def average_vectors(vectors: list[np.ndarray]) -> np.ndarray:
    stacked = np.stack([np.asarray(item, dtype=np.float32).reshape(-1) for item in vectors], axis=0)
    mean = stacked.mean(axis=0)
    norm = np.linalg.norm(mean)
    if norm == 0:
        raise ValueError("Embedding rata-rata kosong.")
    return mean / norm
