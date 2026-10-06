from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class FaceHit:
    box: tuple[int, int, int, int]
    score: float
    raw: np.ndarray


@dataclass(frozen=True)
class NameScore:
    name: str
    score: float


@dataclass(frozen=True)
class QualityReport:
    ok: bool
    yaw: float
    brightness: float
    sharpness: float
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class RecognitionResult:
    status: str
    best_name: str | None
    best_score: float
    scores: tuple[NameScore, ...]
    message: str


@dataclass(frozen=True)
class VerifyResult:
    accepted: bool
    code: str
    claimed: str
    best_name: str | None
    claimed_score: float
    best_score: float
    scores: tuple[NameScore, ...] = field(default_factory=tuple)
    message: str = ""
