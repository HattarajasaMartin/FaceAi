from __future__ import annotations

import urllib.request
from pathlib import Path

import cv2

from .config import (
    MODELS_DIR,
    SFACE_FILE,
    SFACE_MIN_BYTES,
    SFACE_URLS,
    YUNET_FILE,
    YUNET_MIN_BYTES,
    YUNET_URLS,
    ensure_data_dirs,
)

_detector = None
_recognizer = None


def _download(urls: tuple[str, ...], dest: Path, min_bytes: int) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_error = None
    for url in urls:
        tmp = dest.with_suffix(dest.suffix + ".part")
        try:
            print(f"Mengunduh {dest.name} ...")
            urllib.request.urlretrieve(url, tmp)
            size = tmp.stat().st_size
            if size < min_bytes:
                tmp.unlink(missing_ok=True)
                last_error = RuntimeError(
                    f"File {dest.name} terlalu kecil ({size} byte) dari {url}"
                )
                continue
            tmp.replace(dest)
            print(f"Selesai: {dest} ({size} byte)")
            return
        except Exception as exc:  # noqa: BLE001 - coba URL berikutnya
            last_error = exc
            tmp.unlink(missing_ok=True)
    raise RuntimeError(
        f"Gagal mengunduh {dest.name}. Periksa koneksi internet. Detail: {last_error}"
    )


def ensure_models() -> tuple[Path, Path]:
    ensure_data_dirs()
    yunet = MODELS_DIR / YUNET_FILE
    sface = MODELS_DIR / SFACE_FILE
    if not yunet.is_file() or yunet.stat().st_size < YUNET_MIN_BYTES:
        _download(YUNET_URLS, yunet, YUNET_MIN_BYTES)
    if not sface.is_file() or sface.stat().st_size < SFACE_MIN_BYTES:
        _download(SFACE_URLS, sface, SFACE_MIN_BYTES)
    return yunet, sface


def get_detector(input_size: tuple[int, int] = (320, 320)):
    global _detector
    yunet, _ = ensure_models()
    if _detector is None:
        _detector = cv2.FaceDetectorYN.create(
            str(yunet),
            "",
            input_size,
            0.7,
            0.3,
            5000,
        )
    return _detector


def get_recognizer():
    global _recognizer
    _, sface = ensure_models()
    if _recognizer is None:
        _recognizer = cv2.FaceRecognizerSF.create(str(sface), "")
    return _recognizer
