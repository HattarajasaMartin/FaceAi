"""Fase 0: cek webcam Windows, tanpa deteksi wajah."""

from __future__ import annotations

import sys

import cv2

from .camera import CameraError, open_camera
from .config import CAMERA_INDEX


def main() -> int:
    try:
        cap = open_camera(CAMERA_INDEX)
    except CameraError as exc:
        print(exc)
        return 1
    print("Kamera terbuka. Tekan Q untuk keluar.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Gagal membaca frame dari webcam.")
                return 1
            cv2.putText(
                frame,
                "Cek kamera - tekan Q untuk keluar",
                (12, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("Cek Kamera", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
