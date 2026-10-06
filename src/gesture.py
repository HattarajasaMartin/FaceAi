from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import MODELS_DIR, ensure_data_dirs

HAND_MODEL = MODELS_DIR / "hand_landmarker.task"
HAND_MODEL_URLS = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
)
_TIP_PIP = ((8, 6), (12, 10), (16, 14), (20, 18))


@dataclass(frozen=True)
class FingerState:
    index_up: bool
    middle_up: bool
    ring_up: bool
    pinky_up: bool

    @property
    def one_finger(self) -> bool:
        return self.index_up and not self.middle_up and not self.ring_up and not self.pinky_up

    @property
    def peace(self) -> bool:
        return self.index_up and self.middle_up and not self.ring_up and not self.pinky_up

    @property
    def three_fingers(self) -> bool:
        return self.index_up and self.middle_up and self.ring_up and not self.pinky_up

    @property
    def gesture(self) -> str | None:
        if self.three_fingers:
            return "three"
        if self.peace:
            return "peace"
        if self.one_finger:
            return "one"
        return None


def finger_state_from_landmarks(landmarks) -> FingerState:
    def is_up(tip: int, pip: int) -> bool:
        return landmarks[tip].y < landmarks[pip].y - 0.02

    flags = [is_up(tip, pip) for tip, pip in _TIP_PIP]
    return FingerState(*flags)


def _ensure_hand_model() -> Path:
    ensure_data_dirs()
    if HAND_MODEL.is_file() and HAND_MODEL.stat().st_size > 1_000_000:
        return HAND_MODEL
    import urllib.request

    HAND_MODEL.parent.mkdir(parents=True, exist_ok=True)
    last_error = None
    for url in HAND_MODEL_URLS:
        tmp = HAND_MODEL.with_suffix(".part")
        try:
            print("Mengunduh model tangan ...")
            urllib.request.urlretrieve(url, tmp)
            if tmp.stat().st_size < 1_000_000:
                tmp.unlink(missing_ok=True)
                continue
            tmp.replace(HAND_MODEL)
            return HAND_MODEL
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            tmp.unlink(missing_ok=True)
    raise RuntimeError(f"Gagal mengunduh model tangan: {last_error}")


class HandTracker:
    def __init__(self) -> None:
        self._landmarker = None
        self._hold: dict[str, int] = {"one": 0, "peace": 0, "three": 0}
        self._stamp = 0

    def _ensure(self):
        if self._landmarker is not None:
            return self._landmarker
        try:
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
        except ImportError:
            return None
        model = _ensure_hand_model()
        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(model)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        return self._landmarker

    def detect(self, frame) -> str | None:
        landmarker = self._ensure()
        if landmarker is None:
            return None
        import cv2
        import mediapipe as mp

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self._stamp += 33
        result = landmarker.detect_for_video(image, self._stamp)
        seen: str | None = None
        for hand in result.hand_landmarks or []:
            found = finger_state_from_landmarks(hand).gesture
            if found:
                seen = found
                break
        for name in self._hold:
            if seen == name:
                self._hold[name] = 14
            elif self._hold[name] > 0:
                self._hold[name] -= 1
        if self._hold["three"] > 0:
            return "three"
        if self._hold["peace"] > 0:
            return "peace"
        if self._hold["one"] > 0:
            return "one"
        return None
