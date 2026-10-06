from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .config import (
    EMBEDDINGS_PATH,
    ENROLLED_DIR,
    RAW_DIR,
    display_name,
    ensure_data_dirs,
    status_from_count,
)


def average_embeddings(vectors: list[np.ndarray]) -> np.ndarray:
    if not vectors:
        raise ValueError("Tidak ada embedding untuk dirata-rata.")
    stacked = np.stack([np.asarray(v, dtype=np.float32).reshape(-1) for v in vectors], axis=0)
    mean = stacked.mean(axis=0)
    norm = np.linalg.norm(mean)
    if norm == 0:
        raise ValueError("Embedding rata-rata kosong.")
    return mean / norm


def _empty_record(display: str) -> dict:
    return {
        "display": display,
        "samples": [],
        "embedding": None,
        "status": "belum daftar",
        "updated_at": None,
    }


def load_gallery() -> dict:
    ensure_data_dirs()
    if not EMBEDDINGS_PATH.is_file():
        gallery = {"version": 1, "people": {}}
        save_gallery(gallery)
        return gallery
    with EMBEDDINGS_PATH.open("r", encoding="utf-8") as handle:
        gallery = json.load(handle)
    gallery.setdefault("people", {})
    return gallery


def save_gallery(gallery: dict) -> None:
    ensure_data_dirs()
    with EMBEDDINGS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(gallery, handle, indent=2)
        handle.write("\n")


def ensure_person(slug: str, display: str | None = None) -> dict:
    gallery = load_gallery()
    people = gallery["people"]
    label = display or display_name(slug)
    if slug not in people:
        people[slug] = _empty_record(label)
        save_gallery(gallery)
    elif display and people[slug].get("display") != display:
        people[slug]["display"] = display
        save_gallery(gallery)
    (ENROLLED_DIR / slug).mkdir(parents=True, exist_ok=True)
    (RAW_DIR / slug).mkdir(parents=True, exist_ok=True)
    return people[slug]


def person_display(name: str, gallery: dict | None = None) -> str:
    gallery = gallery or load_gallery()
    record = gallery.get("people", {}).get(name) or {}
    return record.get("display") or display_name(name)


def person_status(gallery: dict, name: str) -> dict:
    record = gallery["people"][name]
    count = len(record.get("samples", []))
    return {
        "name": name,
        "display": record.get("display") or display_name(name),
        "count": count,
        "status": status_from_count(count),
        "updated_at": record.get("updated_at"),
    }


def all_status(gallery: dict | None = None) -> list[dict]:
    gallery = gallery or load_gallery()
    names = sorted(gallery["people"], key=lambda item: person_display(item, gallery).lower())
    return [person_status(gallery, name) for name in names]


def _wipe_person_files(name: str) -> None:
    for folder in (ENROLLED_DIR / name, RAW_DIR / name):
        if folder.exists():
            for item in folder.glob("*.jpg"):
                item.unlink()
            try:
                folder.rmdir()
            except OSError:
                pass


def clear_person(name: str) -> None:
    gallery = load_gallery()
    display = person_display(name, gallery)
    gallery["people"][name] = _empty_record(display)
    save_gallery(gallery)
    _wipe_person_files(name)


def remove_person(name: str) -> bool:
    gallery = load_gallery()
    if name not in gallery["people"]:
        _wipe_person_files(name)
        return False
    del gallery["people"][name]
    save_gallery(gallery)
    _wipe_person_files(name)
    return True


def enrolled_vectors(gallery: dict | None = None) -> dict[str, np.ndarray]:
    gallery = gallery or load_gallery()
    vectors: dict[str, np.ndarray] = {}
    for name, record in gallery["people"].items():
        if status_from_count(len(record.get("samples", []))) != "terdaftar":
            continue
        embedding = record.get("embedding")
        if not embedding:
            continue
        vectors[name] = np.asarray(embedding, dtype=np.float32).reshape(-1)
    return vectors


def add_sample(name: str, embedding: np.ndarray, crop_path: Path) -> dict:
    ensure_person(name)
    gallery = load_gallery()
    record = gallery["people"][name]
    samples = record.setdefault("samples", [])
    samples.append(
        {
            "file": crop_path.name,
            "embedding": np.asarray(embedding, dtype=np.float32).reshape(-1).tolist(),
        }
    )
    vectors = [np.asarray(sample["embedding"], dtype=np.float32) for sample in samples]
    record["embedding"] = average_embeddings(vectors).tolist()
    record["status"] = status_from_count(len(samples))
    record["updated_at"] = datetime.now(timezone.utc).isoformat()
    save_gallery(gallery)
    return person_status(gallery, name)
