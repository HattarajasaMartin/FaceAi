from __future__ import annotations

import argparse
import sys

import cv2
import numpy as np

from .burst import FrameBurst
from .camera import CameraError, open_camera
from .config import MULTI_FRAME_COUNT, display_name
from .detect import annotate
from .domain import RecognitionResult
from .gallery import enrolled_vectors
from .image_io import read_image
from .overlay import draw_lines
from .pipeline import analyze_image
from .recognize_core import decide_recognition, score_against_gallery


def recognize_embedding(embedding: np.ndarray) -> RecognitionResult:
    return decide_recognition(score_against_gallery(embedding, enrolled_vectors()))


def recognize_image(image: np.ndarray) -> RecognitionResult:
    sample, message = analyze_image(image)
    if sample is None:
        return RecognitionResult(
            status="wajah_tidak_valid",
            best_name=None,
            best_score=0.0,
            scores=(),
            message=message,
        )
    return recognize_embedding(sample.embedding)


def print_recognition(result: RecognitionResult) -> None:
    print(f"status: {result.status}")
    print(f"nama terdekat: {display_name(result.best_name) if result.best_name else '-'}")
    print(f"skor: {result.best_score:.3f}")
    print(f"pesan: {result.message}")
    for item in result.scores:
        print(f"  {display_name(item.name)}: {item.score:.3f}")


def run_image(path: str, show_window: bool = True) -> int:
    image = read_image(path)
    vis, _, _ = annotate(image)
    result = recognize_image(image)
    print_recognition(result)
    if show_window:
        draw_lines(vis, [result.message], (12, 90))
        cv2.imshow("Kenali Wajah", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return 0 if result.status == "cocok" else 2


def run_camera() -> int:
    try:
        cap = open_camera()
    except CameraError as exc:
        print(exc)
        return 1
    burst = FrameBurst()
    last = "SPACE = ambil 5 frame | Q = keluar"
    print("Kenali wajah. SPACE = ambil, gerakkan kepala sedikit, Q = keluar.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Frame kamera gagal dibaca.")
                return 1
            vis, _, message = annotate(frame)
            if burst.active:
                err = burst.add_frame(frame)
                if len(burst.embeddings) >= MULTI_FRAME_COUNT:
                    embedding, status = burst.complete()
                    if embedding is None:
                        last = status
                    else:
                        result = recognize_embedding(embedding)
                        print_recognition(result)
                        last = result.message
                elif err:
                    last = err
            draw_lines(
                vis,
                [message, last, f"sampel {burst.progress}" if burst.active else "SPACE = ambil | Q = keluar"],
                (12, vis.shape[0] - 72),
            )
            cv2.imshow("Kenali Wajah", vis)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key == 32 and not burst.active:
                burst.reset()
                last = "Ambil frame..."
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Kenali siapa wajah ini.")
    parser.add_argument("image", nargs="?", help="Path foto. Jika kosong, pakai webcam.")
    parser.add_argument("--no-window", action="store_true")
    args = parser.parse_args(argv)
    if args.image:
        return run_image(args.image, show_window=not args.no_window)
    return run_camera()


if __name__ == "__main__":
    sys.exit(main())
