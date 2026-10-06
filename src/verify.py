from __future__ import annotations

import argparse
import sys

import cv2
import numpy as np

from .audit import append_verify_log
from .burst import FrameBurst
from .camera import CameraError, open_camera
from .config import MULTI_FRAME_COUNT, display_name, resolve_name
from .detect import annotate
from .domain import VerifyResult
from .gallery import enrolled_vectors
from .image_io import read_image
from .lockout import LockoutStore
from .overlay import draw_lines
from .pipeline import analyze_image
from .recognize_core import score_against_gallery
from .verify_core import decide_verify


def invalid_face_result(claimed: str, message: str) -> VerifyResult:
    code = "TOLAK_NO_FACE"
    if "tepat satu" in message.lower():
        code = "TOLAK_MULTI_FACE"
    elif "buram" in message.lower() or "gelap" in message.lower() or "miring" in message.lower() or "kecil" in message.lower():
        code = "TOLAK_QUALITY"
    elif "kiri-kanan" in message.lower():
        code = "TOLAK_LIVENESS"
    return VerifyResult(
        accepted=False,
        code=code,
        claimed=claimed,
        best_name=None,
        claimed_score=0.0,
        best_score=0.0,
        scores=(),
        message=message,
    )


def verify_embedding(claimed: str, embedding: np.ndarray, lockout: LockoutStore | None = None) -> VerifyResult:
    store = lockout or LockoutStore()
    remaining = store.remaining_seconds(claimed)
    if remaining > 0:
        return VerifyResult(
            accepted=False,
            code="TOLAK_LOCKOUT",
            claimed=claimed,
            best_name=None,
            claimed_score=0.0,
            best_score=0.0,
            scores=(),
            message=f"Terlalu banyak gagal. Coba lagi dalam {remaining} detik",
        )

    gallery = enrolled_vectors()
    scores = score_against_gallery(embedding, gallery)
    result = decide_verify(claimed, scores, frozenset(gallery))
    if result.accepted:
        store.register_success(claimed)
    else:
        store.register_failure(claimed)
    append_verify_log(result)
    return result


def verify_image(claimed: str, image: np.ndarray, lockout: LockoutStore | None = None) -> VerifyResult:
    sample, message = analyze_image(image)
    if sample is None:
        result = invalid_face_result(claimed, message)
        append_verify_log(result)
        return result
    return verify_embedding(claimed, sample.embedding, lockout=lockout)


def print_verify(result: VerifyResult) -> None:
    state = "TERIMA" if result.accepted else "TOLAK"
    print(f"{state} [{result.code}] {result.message}")
    print(f"klaim: {display_name(result.claimed)}")
    if result.best_name:
        print(f"terdekat: {display_name(result.best_name)} ({result.best_score:.3f})")
    for item in result.scores:
        print(f"  {display_name(item.name)}: {item.score:.3f}")


def run_image(claimed: str, path: str, show_window: bool = True) -> int:
    image = read_image(path)
    vis, _, _ = annotate(image)
    result = verify_image(claimed, image)
    print_verify(result)
    if show_window:
        color = (40, 180, 40) if result.accepted else (40, 40, 220)
        draw_lines(vis, [result.message], (12, 90), color)
        cv2.imshow("Verifikasi Klaim", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return 0 if result.accepted else 2


def run_camera(claimed: str) -> int:
    try:
        cap = open_camera()
    except CameraError as exc:
        print(exc)
        return 1
    burst = FrameBurst()
    last = "SPACE = scan | Q = keluar"
    print(f"Verifikasi klaim {display_name(claimed)}. SPACE = scan, gerakkan kepala.")
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
                        result = invalid_face_result(claimed, status)
                        append_verify_log(result)
                        last = result.message
                        print_verify(result)
                    else:
                        result = verify_embedding(claimed, embedding)
                        print_verify(result)
                        last = result.message
                elif err:
                    last = err
            color = (40, 180, 40) if last.startswith("TERIMA") else (255, 255, 255)
            draw_lines(
                vis,
                [
                    f"Klaim: {display_name(claimed)}",
                    message,
                    last,
                    f"sampel {burst.progress}" if burst.active else "SPACE = scan | Q = keluar",
                ],
                (12, vis.shape[0] - 96),
                color,
            )
            cv2.imshow("Verifikasi Klaim", vis)
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
    parser = argparse.ArgumentParser(description="Verifikasi klaim nama vs wajah.")
    parser.add_argument("name", help="Nama yang diklaim, contoh: anies")
    parser.add_argument("image", nargs="?", help="Path foto. Jika kosong, pakai webcam.")
    parser.add_argument("--no-window", action="store_true")
    args = parser.parse_args(argv)
    try:
        claimed = resolve_name(args.name)
    except ValueError as exc:
        print(exc)
        return 1
    if args.image:
        return run_image(claimed, args.image, show_window=not args.no_window)
    return run_camera(claimed)


if __name__ == "__main__":
    sys.exit(main())
