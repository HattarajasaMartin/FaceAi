from __future__ import annotations

import numpy as np

from .config import MATCH_THRESHOLD, display_name, score_percent
from .domain import NameScore, RecognitionResult


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.dot(left, right))


def score_against_gallery(
    embedding: np.ndarray,
    gallery_vectors: dict[str, np.ndarray],
) -> tuple[NameScore, ...]:
    scores = [
        NameScore(name=name, score=cosine_similarity(embedding, vector))
        for name, vector in gallery_vectors.items()
    ]
    return tuple(sorted(scores, key=lambda item: item.score, reverse=True))


def decide_recognition(
    scores: tuple[NameScore, ...],
    threshold: float = MATCH_THRESHOLD,
) -> RecognitionResult:
    if not scores:
        return RecognitionResult(
            status="tidak_dikenali",
            best_name=None,
            best_score=0.0,
            scores=scores,
            message="Belum ada karyawan yang terdaftar",
        )
    best = scores[0]
    if best.score < threshold:
        return RecognitionResult(
            status="tidak_dikenali",
            best_name=best.name,
            best_score=best.score,
            scores=scores,
            message="tidak_dikenali",
        )
    return RecognitionResult(
        status="cocok",
        best_name=best.name,
        best_score=best.score,
        scores=scores,
        message=f"Cocok: {display_name(best.name)}  {score_percent(best.score)}%",
    )
