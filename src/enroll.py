from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

from .camera import CameraError, open_camera
from .config import (
    ENROLLED_DIR,
    MAX_SAMPLES,
    MIN_SAMPLES,
    RAW_DIR,
    display_name,
    parse_person_name,
)
from .detect import annotate, crop_face, extract_embedding
from .gallery import add_sample, all_status, clear_person, ensure_person, load_gallery
from .image_io import read_image, write_image
from .overlay import draw_lines
from .pipeline import analyze_image


def print_status() -> None:
    print("Status pendaftaran:")
    for row in all_status():
        print(f"  - {row['display']:<10} {row['status']:<14} foto={row['count']}")


def _next_index(name: str) -> int:
    folder = ENROLLED_DIR / name
    existing = [int(p.stem) for p in folder.glob("*.jpg") if p.stem.isdigit()]
    return (max(existing) + 1) if existing else 1


def save_sample(name: str, image, face) -> dict:
    index = _next_index(name)
    crop = crop_face(image, face)
    embedding = extract_embedding(image, face)
    crop_path = ENROLLED_DIR / name / f"{index:03d}.jpg"
    raw_path = RAW_DIR / name / f"{index:03d}.jpg"
    write_image(crop_path, crop)
    write_image(raw_path, image)
    return add_sample(name, embedding, crop_path)


def enroll_image(name: str, path: str) -> dict:
    image = read_image(path)
    sample, message = analyze_image(image)
    if sample is None:
        raise ValueError(f"{Path(path).name}: {message}")
    return save_sample(name, image, sample.face)


def enroll_images(name: str, paths: list[str], replace: bool) -> dict:
    if replace:
        clear_person(name)
    last = None
    for path in paths:
        last = enroll_image(name, path)
        print(f"OK {Path(path).name} -> {last['display']} ({last['count']} foto, {last['status']})")
    return last or next(row for row in all_status() if row["name"] == name)


def enroll_camera(name: str, replace: bool) -> int:
    if replace:
        clear_person(name)

    gallery = load_gallery()
    start_count = len(gallery["people"].get(name, {}).get("samples", []))
    if start_count >= MAX_SAMPLES:
        print(f"{display_name(name)} sudah {start_count} foto. Pakai --replace untuk daftar ulang.")
        return 1

    try:
        cap = open_camera()
    except CameraError as exc:
        print(exc)
        return 1
    captured = 0
    label = display_name(name)
    print(f"Daftar {label}. SPACE = ambil, Q = selesai, ESC = batal.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Frame kamera gagal dibaca.")
                return 1
            vis, valid, message = annotate(frame)
            total = start_count + captured
            draw_lines(
                vis,
                [
                    f"Daftar: {label}",
                    f"Foto: {total}/{MAX_SAMPLES} (minimal {MIN_SAMPLES} = terdaftar)",
                    message,
                    "SPACE = ambil | Q = selesai | ESC = batal",
                ],
                (12, vis.shape[0] - 88),
            )
            cv2.imshow("Pendaftaran Wajah", vis)
            key = cv2.waitKey(1) & 0xFF
            if key in (27,):
                print("Dibatalkan. Data yang sudah tersimpan tetap ada.")
                break
            if key in (ord("q"), ord("Q")):
                break
            if key == 32:
                if not valid:
                    print(f"Tidak diambil: {message}")
                    continue
                if start_count + captured >= MAX_SAMPLES:
                    print(f"Sudah mencapai {MAX_SAMPLES} foto.")
                    break
                sample, err = analyze_image(frame)
                if sample is None:
                    print(err)
                    continue
                status = save_sample(name, frame, sample.face)
                captured += 1
                print(f"Tersimpan foto {status['count']} ({status['status']})")
                if status["count"] >= MAX_SAMPLES:
                    print("Kuota foto penuh.")
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    status = next(row for row in all_status() if row["name"] == name)
    print(
        f"Selesai {status['display']}: {status['count']} foto, status {status['status']}."
    )
    if status["status"] != "terdaftar":
        print(f"Belum lengkap. Tambah foto sampai minimal {MIN_SAMPLES}.")
        return 2
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Daftarkan wajah karyawan (ketik nama sendiri)."
    )
    parser.add_argument(
        "name",
        nargs="?",
        help='Nama lengkap, contoh: "Budi Santoso"',
    )
    parser.add_argument(
        "images",
        nargs="*",
        help="Path foto. Jika kosong, pakai webcam.",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Tampilkan status pendaftaran saja.",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Hapus data lama nama ini, lalu daftar ulang.",
    )
    args = parser.parse_args(argv)

    load_gallery()
    if args.status:
        print_status()
        return 0
    if not args.name:
        print_status()
        print(
            "\nContoh:\n"
            '  python -m src.enroll "Budi Santoso"\n'
            '  python -m src.enroll "Budi Santoso" foto1.jpg foto2.jpg foto3.jpg foto4.jpg foto5.jpg\n'
            '  python -m src.enroll "Budi Santoso" --replace'
        )
        return 0

    try:
        name, display = parse_person_name(args.name)
        ensure_person(name, display)
    except ValueError as exc:
        print(exc)
        return 1

    if args.images:
        try:
            enroll_images(name, args.images, replace=args.replace)
        except (OSError, ValueError) as exc:
            print(exc)
            return 1
        print_status()
        status = next(row for row in all_status() if row["name"] == name)
        return 0 if status["status"] == "terdaftar" else 2

    return enroll_camera(name, replace=args.replace)


if __name__ == "__main__":
    sys.exit(main())
