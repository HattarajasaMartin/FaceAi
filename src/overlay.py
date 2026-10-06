from __future__ import annotations

import cv2
import numpy as np

from .config import COMPANY_NAME, score_percent
from .domain import FaceHit
from .gallery import person_display

FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE = (255, 255, 255)
GREEN = (92, 196, 88)
RED = (82, 82, 230)
AMBER = (72, 186, 232)
TEAL = (120, 150, 32)
NAVY = (42, 32, 22)
GOLD = (90, 190, 230)


def pretty_name(raw: str) -> str:
    text = (raw or "").strip()
    if not text:
        return text
    if text.islower() or text.isupper():
        return text.title()
    return text


def put_text(
    frame: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int] = WHITE,
    scale: float = 0.7,
    thickness: int = 2,
) -> None:
    cv2.putText(frame, text, origin, FONT, scale, color, thickness, cv2.LINE_AA)


def employee_caption(name: str | None) -> tuple[str, str]:
    if name:
        return pretty_name(person_display(name)), f"Karyawan {COMPANY_NAME}"
    return "Tidak dikenali", COMPANY_NAME


def _filled_round(img: np.ndarray, pt1: tuple[int, int], pt2: tuple[int, int], color, radius: int) -> None:
    x1, y1 = pt1
    x2, y2 = pt2
    radius = max(1, min(radius, (x2 - x1) // 2, (y2 - y1) // 2))
    cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1)
    cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1)
    cv2.circle(img, (x1 + radius, y1 + radius), radius, color, -1)
    cv2.circle(img, (x2 - radius, y1 + radius), radius, color, -1)
    cv2.circle(img, (x1 + radius, y2 - radius), radius, color, -1)
    cv2.circle(img, (x2 - radius, y2 - radius), radius, color, -1)


def glass_panel(
    frame: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
    color: tuple[int, int, int] = NAVY,
    alpha: float = 0.62,
    radius: int = 16,
) -> None:
    x = max(0, x)
    y = max(0, y)
    width = min(width, frame.shape[1] - x)
    height = min(height, frame.shape[0] - y)
    if width < 8 or height < 8:
        return
    overlay = frame.copy()
    _filled_round(overlay, (x, y), (x + width, y + height), color, radius)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.rectangle(frame, (x + 1, y + 1), (x + width - 2, y + 4), TEAL, -1)


def _label_size(lines: list[str], scale: float, thickness: int) -> tuple[int, int]:
    widths = []
    height = 18
    for line in lines:
        size, _ = cv2.getTextSize(line, FONT, scale, thickness)
        widths.append(size[0])
        height += size[1] + 10
    return max(widths, default=0) + 28, height


def draw_label_block(
    frame: np.ndarray,
    origin: tuple[int, int],
    lines: list[str],
    fg: tuple[int, int, int],
    bg: tuple[int, int, int],
    scale: float = 0.65,
) -> None:
    width, height = _label_size(lines, scale, 2)
    x, y = origin
    x = max(10, min(x, frame.shape[1] - width - 10))
    y = max(10, min(y, frame.shape[0] - height - 10))
    glass_panel(frame, x, y, width, height, bg, 0.72, 14)
    text_y = y + 28
    for line in lines:
        put_text(frame, line, (x + 14, text_y), fg, scale, 2)
        text_y += 26


def draw_face_box(frame: np.ndarray, face: FaceHit, ok: bool) -> None:
    color = GREEN if ok else RED
    x, y, width, height = face.box
    length = min(28, width // 4, height // 4)
    thick = 3
    cv2.line(frame, (x, y), (x + length, y), color, thick)
    cv2.line(frame, (x, y), (x, y + length), color, thick)
    cv2.line(frame, (x + width, y), (x + width - length, y), color, thick)
    cv2.line(frame, (x + width, y), (x + width, y + length), color, thick)
    cv2.line(frame, (x, y + height), (x + length, y + height), color, thick)
    cv2.line(frame, (x, y + height), (x, y + height - length), color, thick)
    cv2.line(frame, (x + width, y + height), (x + width - length, y + height), color, thick)
    cv2.line(frame, (x + width, y + height), (x + width, y + height - length), color, thick)


def draw_custom_employee_label(frame: np.ndarray, face: FaceHit, title: str) -> None:
    _identity_card(frame, face, pretty_name(title), f"Karyawan {COMPANY_NAME}", True)


def draw_employee_label(
    frame: np.ndarray,
    face: FaceHit,
    name: str | None,
    score: float | None = None,
) -> None:
    title, subtitle = employee_caption(name)
    percent = score_percent(score) if name and score is not None else None
    _identity_card(frame, face, title, subtitle, bool(name), percent)


def _identity_card(
    frame: np.ndarray,
    face: FaceHit,
    title: str,
    subtitle: str,
    known: bool,
    percent: int | None = None,
) -> None:
    fg = GREEN if known else RED
    bg = (28, 48, 22) if known else (28, 22, 48)
    x, y, width, _ = face.box
    lines = [title, subtitle]
    name_scale = 0.52
    pct_scale = 0.95
    box_w, box_h = _label_size(lines, name_scale, 2)
    if percent is not None:
        pct_text = f"{percent}%"
        size, _ = cv2.getTextSize(pct_text, FONT, pct_scale, 4)
        box_w = max(box_w, size[0] + 28)
        box_h += size[1] + 18
    origin_x = x + max(0, (width - box_w) // 2)
    origin_y = y - box_h - 10
    origin_x = max(8, min(origin_x, frame.shape[1] - box_w - 8))
    origin_y = max(8, min(origin_y, frame.shape[0] - box_h - 8))
    glass_panel(frame, origin_x, origin_y, box_w, box_h, bg, 0.72, 10)
    text_y = origin_y + 20
    put_text(frame, title, (origin_x + 10, text_y), fg, name_scale, 1)
    text_y += 18
    put_text(frame, subtitle, (origin_x + 10, text_y), fg, 0.42, 1)
    if percent is not None:
        text_y += 26
        put_text(frame, f"{percent}%", (origin_x + 10, text_y), fg, pct_scale, 2)


def draw_header(frame: np.ndarray, mode_title: str) -> None:
    height = 46
    glass_panel(frame, 0, 0, frame.shape[1], height, (36, 28, 16), 0.68, 0)
    put_text(frame, COMPANY_NAME, (14, 20), WHITE, 0.5, 1)
    put_text(frame, "Absensi Karyawan", (14, 38), GOLD, 0.4, 1)
    chip = mode_title
    size, _ = cv2.getTextSize(chip, FONT, 0.42, 1)
    chip_w = size[0] + 20
    chip_x = frame.shape[1] - chip_w - 12
    glass_panel(frame, chip_x, 10, chip_w, 26, TEAL, 0.85, 10)
    put_text(frame, chip, (chip_x + 10, 28), WHITE, 0.42, 1)


def draw_footer(frame: np.ndarray, lines: list[str], accent: tuple[int, int, int] = WHITE) -> None:
    if not lines:
        return
    box_w, box_h = _label_size(lines, 0.45, 1)
    box_w = min(max(box_w, 280), frame.shape[1] - 16)
    x = 8
    y = frame.shape[0] - box_h - 8
    glass_panel(frame, x, y, box_w, box_h, NAVY, 0.7, 12)
    text_y = y + 20
    for index, line in enumerate(lines):
        color = accent if index == 0 else (220, 220, 220)
        put_text(frame, line, (x + 12, text_y), color, 0.45 if index else 0.48, 1)
        text_y += 18


def draw_progress(frame: np.ndarray, x: int, y: int, width: int, filled: float) -> None:
    filled = max(0.0, min(1.0, filled))
    height = 12
    glass_panel(frame, x, y, width, height, (20, 20, 20), 0.75, 6)
    inner = max(0, int((width - 4) * filled))
    if inner > 2:
        _filled_round(frame, (x + 2, y + 2), (x + 2 + inner, y + height - 2), GREEN, 4)


def draw_toast(frame: np.ndarray, text: str, kind: str = "info", y: int | None = None) -> None:
    if not text:
        return
    palette = {
        "success": (GREEN, (24, 52, 28)),
        "error": (RED, (28, 22, 48)),
        "info": (GOLD, NAVY),
    }
    fg, bg = palette.get(kind, palette["info"])
    lines = [part.strip() for part in text.split("\n") if part.strip()]
    widths = []
    height = 16
    scales = []
    for line in lines:
        scale = 1.05 if line.endswith("%") and line[:-1].isdigit() else 0.55
        thick = 2 if scale > 1 else 1
        size, _ = cv2.getTextSize(line, FONT, scale, thick)
        widths.append(size[0])
        height += size[1] + (14 if scale > 1 else 8)
        scales.append(scale)
    width = max(widths, default=0) + 24
    x = (frame.shape[1] - width) // 2
    y = 78 if y is None else y
    x = max(10, min(x, frame.shape[1] - width - 10))
    y = max(10, min(y, frame.shape[0] - height - 10))
    glass_panel(frame, x, y, width, height, bg, 0.78, 16)
    text_y = y + 24
    for line, scale in zip(lines, scales):
        thick = 2 if scale > 1 else 1
        put_text(frame, line, (x + 12, text_y), fg, scale, thick)
        text_y += 32 if scale > 1 else 20


def draw_lines(
    frame: np.ndarray,
    lines: list[str],
    origin: tuple[int, int] = (12, 28),
    color: tuple[int, int, int] = WHITE,
    scale: float = 0.65,
    gap: int = 24,
) -> None:
    x, y = origin
    for line in lines:
        put_text(frame, line, (x, y), color, scale, 2)
        y += gap


def _center_text(frame: np.ndarray, text: str, y: int, color, scale: float, thick: int) -> None:
    size, _ = cv2.getTextSize(text, FONT, scale, thick)
    x = max(8, (frame.shape[1] - size[0]) // 2)
    put_text(frame, text, (x, y), color, scale, thick)


def _pill(frame: np.ndarray, x: int, y: int, text: str, fg, bg) -> int:
    size, _ = cv2.getTextSize(text, FONT, 0.45, 1)
    width, height = size[0] + 28, 32
    glass_panel(frame, x, y, width, height, bg, 0.88, 12)
    put_text(frame, text, (x + 14, y + 22), fg, 0.45, 1)
    return width


def draw_login_page(
    frame: np.ndarray,
    face: FaceHit | None = None,
    valid: bool = False,
    message: str = "",
) -> None:
    height, width = frame.shape[:2]
    shade = frame.copy()
    cv2.rectangle(shade, (0, 0), (width, 88), (16, 12, 8), -1)
    cv2.rectangle(shade, (0, height - 118), (width, height), (16, 12, 8), -1)
    cv2.addWeighted(shade, 0.55, frame, 0.45, 0, frame)

    glass_panel(frame, 0, 0, width, 48, (32, 24, 14), 0.72, 0)
    put_text(frame, COMPANY_NAME, (14, 20), GOLD, 0.48, 1)
    put_text(frame, "Absensi Karyawan", (14, 38), (210, 210, 210), 0.38, 1)
    chip = "Login Face ID"
    chip_size, _ = cv2.getTextSize(chip, FONT, 0.4, 1)
    glass_panel(frame, width - chip_size[0] - 32, 12, chip_size[0] + 20, 26, TEAL, 0.88, 10)
    put_text(frame, chip, (width - chip_size[0] - 22, 30), WHITE, 0.4, 1)

    _center_text(frame, "Selamat datang", 72, WHITE, 0.62, 2)
    _center_text(frame, "Arahkan wajah ke lingkaran", 94, (210, 210, 210), 0.42, 1)

    cx, cy = width // 2, height // 2 + 8
    rx, ry = min(88, width // 7), min(108, height // 4)
    ring = GREEN if valid else GOLD
    cv2.ellipse(frame, (cx, cy), (rx + 10, ry + 10), 0, 0, 360, ring, 2, cv2.LINE_AA)
    cv2.ellipse(frame, (cx, cy), (rx, ry), 0, 0, 360, ring, 3, cv2.LINE_AA)
    if face is not None:
        draw_face_box(frame, face, valid)

    status = "Wajah terdeteksi" if valid else "Menunggu wajah"
    if message:
        status = message
    if message.startswith("TOLAK"):
        status_color = RED
    elif valid:
        status_color = GREEN
    else:
        status_color = GOLD
    _center_text(frame, status, min(height - 78, cy + ry + 28), status_color, 0.45, 1)

    glass_panel(frame, 0, height - 64, width, 64, (28, 22, 14), 0.78, 0)
    daftar = "2  Daftar"
    keluar = "Q  Keluar"
    daftar_size, _ = cv2.getTextSize(daftar, FONT, 0.45, 1)
    keluar_size, _ = cv2.getTextSize(keluar, FONT, 0.45, 1)
    daftar_w, keluar_w, gap = daftar_size[0] + 28, keluar_size[0] + 28, 12
    start_x = (width - (daftar_w + keluar_w + gap)) // 2
    _pill(frame, start_x, height - 40, daftar, WHITE, TEAL)
    _pill(frame, start_x + daftar_w + gap, height - 40, keluar, WHITE, (48, 42, 36))
    _center_text(frame, "SPASI = login", height - 54, (180, 180, 180), 0.38, 1)
