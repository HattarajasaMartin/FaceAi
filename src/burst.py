from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import MIN_YAW_RANGE, MULTI_FRAME_COUNT
from .pipeline import analyze_image, average_vectors
from .quality import liveness_from_yaw


@dataclass
class FrameBurst:
    embeddings: list[np.ndarray] = field(default_factory=list)
    yaws: list[float] = field(default_factory=list)
    active: bool = False

    def reset(self) -> None:
        self.embeddings.clear()
        self.yaws.clear()
        self.active = True

    def add_frame(self, image: np.ndarray) -> str | None:
        sample, message = analyze_image(image)
        if sample is None:
            return message
        self.embeddings.append(sample.embedding)
        self.yaws.append(sample.quality.yaw)
        return None

    @property
    def progress(self) -> str:
        return f"{len(self.embeddings)}/{MULTI_FRAME_COUNT}"

    def complete(self) -> tuple[np.ndarray | None, str]:
        if len(self.embeddings) < MULTI_FRAME_COUNT:
            return None, f"Mengambil frame {self.progress}"
        live_ok, live_msg = liveness_from_yaw(self.yaws, MIN_YAW_RANGE)
        vectors = list(self.embeddings)
        self.embeddings.clear()
        self.yaws.clear()
        self.active = False
        if not live_ok:
            return None, live_msg
        return average_vectors(vectors), "OK"
