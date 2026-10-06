from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import cv2
import numpy as np

from .camera import CameraError, open_camera
from .config import DETECT_SCORE, MIN_FACE_SIZE
from .domain import FaceHit
from .image_io import read_image
from .models import get_detector, get_recognizer
from .overlay import draw_face_box, draw_lines
from .quality import assess_quality

__all__ = [
    "CameraError",
    "FaceHit",
    "annotate",
    "crop_face",
    "detect_faces",
    "detect_result",
    "extract_embedding",
    "open_camera",
    "require_single_face",
    "validate_face_count",
]


@dataclass
class Detection:
    faces: list[FaceHit]
    discarded_small: int


def validate_face_count(count: int) -> tuple[bool, str]:
    if count == 0:
        return False, "Wajah tidak terdeteksi"
    if count > 1:
        return False, f"Harus tepat satu wajah (terdeteksi {count})"
    return True, "1 wajah - OK"


def detect_result(image: np.ndarray, score_threshold: float = DETECT_SCORE) -> Detection:
    if image is None or image.size == 0:
        return Detection([], 0)
    height, width = image.shape[:2]
    detector = get_detector((width, height))
    detector.setInputSize((width, height))
    _, faces = detector.detect(image)
    if faces is None:
        return Detection([], 0)

    hits: list[FaceHit] = []
    discarded_small = 0
    for face in faces:
        x, y, box_w, box_h = [int(round(value)) for value in face[:4]]
        x = max(0, x)
        y = max(0, y)
        box_w = max(1, min(box_w, width - x))
        box_h = max(1, min(box_h, height - y))
        score = float(face[-1])
        if score < score_threshold:
            continue
        if min(box_w, box_h) < MIN_FACE_SIZE:
            discarded_small += 1
            continue
        hits.append(FaceHit(box=(x, y, box_w, box_h), score=score, raw=face))
    return Detection(hits, discarded_small)


def detect_faces(image: np.ndarray, score_threshold: float = DETECT_SCORE) -> list[FaceHit]:
    return detect_result(image, score_threshold).faces


def face_count_message(detection: Detection) -> tuple[bool, str]:
    if not detection.faces and detection.discarded_small:
        return False, "Wajah terlalu kecil, mendekat ke kamera"
    return validate_face_count(len(detection.faces))


def require_single_face(image: np.ndarray) -> tuple[FaceHit | None, str]:
    detection = detect_result(image)
    ok, message = face_count_message(detection)
    if not ok:
        return None, message
    return detection.faces[0], message


def crop_face(image: np.ndarray, face: FaceHit, padding: float = 0.2) -> np.ndarray:
    height, width = image.shape[:2]
    x, y, box_w, box_h = face.box
    pad_x = int(box_w * padding)
    pad_y = int(box_h * padding)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(width, x + box_w + pad_x)
    y2 = min(height, y + box_h + pad_y)
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        raise ValueError("Gagal memotong wajah.")
    return crop


def extract_embedding(image: np.ndarray, face: FaceHit) -> np.ndarray:
    recognizer = get_recognizer()
    aligned = recognizer.alignCrop(image, face.raw)
    feature = recognizer.feature(aligned)
    vector = np.asarray(feature, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise ValueError("Embedding wajah kosong.")
    return vector / norm


def annotate(
    image: np.ndarray,
    faces: list[FaceHit] | None = None,
    detection: Detection | None = None,
    show_hud: bool = True,
) -> tuple[np.ndarray, bool, str]:
    detection = detection or detect_result(image)
    if faces is None:
        faces = detection.faces
    frame = image.copy()
    ok, message = face_count_message(detection) if faces is detection.faces else validate_face_count(len(faces))
    for face in faces:
        draw_face_box(frame, face, ok)
        if ok and show_hud:
            quality = assess_quality(image, face)
            if quality.warnings:
                draw_lines(frame, list(quality.warnings), (12, 88), (40, 180, 220), 0.55)
    if show_hud:
        draw_lines(
            frame,
            [f"Jumlah wajah: {len(faces)}", message],
            (12, 28),
            (40, 180, 40) if ok else (40, 40, 220),
        )
    return frame, ok, message


def run_camera() -> int:
    try:
        cap = open_camera()
    except CameraError as exc:
        print(exc)
        return 1
    print("Deteksi kamera. Q = keluar.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Frame kamera gagal dibaca.")
                return 1
            vis, _, _ = annotate(frame)
            draw_lines(vis, ["Q = keluar"], (12, vis.shape[0] - 16), scale=0.6)
            cv2.imshow("Deteksi Wajah - Kamera", vis)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


def run_image(path: str, show_window: bool = True) -> int:
    image = read_image(path)
    vis, ok, message = annotate(image)
    print(message)
    print(f"Jumlah wajah: {len(detect_faces(image))}")
    if not show_window:
        return 0 if ok else 2
    cv2.imshow("Deteksi Wajah - Foto", vis)
    print("Tekan tombol apa saja di jendela untuk menutup.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return 0 if ok else 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Deteksi wajah dari kamera atau foto.")
    parser.add_argument("image", nargs="?", help="Path foto. Jika kosong, pakai webcam.")
    parser.add_argument("--no-window", action="store_true", help="Tanpa jendela, hanya teks.")
    args = parser.parse_args(argv)
    if args.image:
        return run_image(args.image, show_window=not args.no_window)
    return run_camera()


if __name__ == "__main__":
    sys.exit(main())
