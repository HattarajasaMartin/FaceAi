from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = DATA_DIR / "models"
ENROLLED_DIR = DATA_DIR / "enrolled"
RAW_DIR = DATA_DIR / "raw"
EMBEDDINGS_PATH = DATA_DIR / "embeddings.json"
VERIFY_LOG_PATH = DATA_DIR / "verify_log.jsonl"
LOCKOUT_PATH = DATA_DIR / "lockout.json"
OVERLAYS_DIR = DATA_DIR / "overlays"
MULYADI_STICKER = OVERLAYS_DIR / "mulyadi.jpg"
MEGAWATI_STICKER = OVERLAYS_DIR / "megawati.jpg"
FERY_STICKER = OVERLAYS_DIR / "fery_irwandi.jpg"

NAME_ALIASES = {
    "anis": "anies",
}

COMPANY_NAME = "PT. Testing"
MIN_SAMPLES = 5
MAX_SAMPLES = 10
MIN_NAME_LENGTH = 2
MAX_NAME_LENGTH = 40
MIN_FACE_SIZE = 80
DETECT_SCORE = 0.7
CAMERA_INDEX = 0

MATCH_THRESHOLD = 0.36
IMPOSTOR_MARGIN = 0.05
MULTI_FRAME_COUNT = 5
MAX_ABS_YAW = 0.55
MIN_YAW_RANGE = 0.08
DARK_WARN = 55.0
DARK_REJECT = 22.0
BLUR_REJECT = 28.0
LOCKOUT_LIMIT = 5
LOCKOUT_SECONDS = 30


def score_percent(score: float) -> int:
    return int(round(max(0.0, min(1.0, float(score))) * 100))

YUNET_FILE = "face_detection_yunet_2023mar.onnx"
SFACE_FILE = "face_recognition_sface_2021dec.onnx"
YUNET_MIN_BYTES = 200_000
SFACE_MIN_BYTES = 30_000_000

YUNET_URLS = (
    "https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx",
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
)
SFACE_URLS = (
    "https://huggingface.co/opencv/face_recognition_sface/resolve/main/face_recognition_sface_2021dec.onnx",
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
)


def slugify(raw: str) -> str:
    text = (raw or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")[:MAX_NAME_LENGTH]


def parse_person_name(raw: str) -> tuple[str, str]:
    display = " ".join((raw or "").split())
    if len(display) < MIN_NAME_LENGTH:
        raise ValueError("Nama harus diisi, minimal 2 huruf.")
    if len(display) > MAX_NAME_LENGTH:
        raise ValueError(f"Nama terlalu panjang (maksimal {MAX_NAME_LENGTH} huruf).")
    slug = NAME_ALIASES.get(slugify(display), slugify(display))
    if not slug:
        raise ValueError("Nama harus memakai huruf atau angka.")
    return slug, display


def resolve_name(raw: str) -> str:
    slug, _ = parse_person_name(raw)
    return slug


def display_name(name: str) -> str:
    return name.replace("_", " ").title()


def status_from_count(count: int) -> str:
    if count >= MIN_SAMPLES:
        return "terdaftar"
    if count > 0:
        return "belum lengkap"
    return "belum daftar"


def ensure_data_dirs() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ENROLLED_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    OVERLAYS_DIR.mkdir(parents=True, exist_ok=True)
