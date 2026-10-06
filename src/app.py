from __future__ import annotations

import sys
from enum import Enum

import cv2

from .audit import append_verify_log, recent_logs
from .burst import FrameBurst
from .camera import CameraError, open_camera
from .config import (
    COMPANY_NAME,
    MAX_SAMPLES,
    MIN_SAMPLES,
    MULTI_FRAME_COUNT,
    score_percent,
)
from .detect import annotate, detect_result, extract_embedding
from .enroll import save_sample
from .faceid import decide_faceid
from .gallery import all_status, clear_person, person_display, remove_person
from .gesture import HandTracker
from .lockout import LockoutStore
from .overlay import (
    draw_custom_employee_label,
    draw_employee_label,
    draw_footer,
    draw_header,
    draw_login_page,
    draw_progress,
    draw_toast,
    pretty_name,
)
from .pipeline import analyze_image
from .prompt import ask_employee_name
from .recognize import print_recognition, recognize_embedding
from .sticker import paste_sticker
from .verify import print_verify, verify_embedding


class Mode(str, Enum):
    LOGIN = "login"
    MENU = "menu"
    DETECT = "detect"
    ADMIN = "admin"
    ENROLL = "enroll"
    RECOGNIZE = "recognize"
    VERIFY = "verify"
    FACEID = "faceid"


def _is_admin_area(mode: Mode) -> bool:
    return mode in (Mode.ADMIN, Mode.ENROLL, Mode.RECOGNIZE, Mode.VERIFY)


def _name_help() -> str:
    rows = all_status()
    if not rows:
        return "Belum ada karyawan terdaftar"
    parts = [f"{index + 1}={row['display']}" for index, row in enumerate(rows[:9])]
    return "  ".join(parts)


def _status_lines() -> list[str]:
    lines = ["Status:"]
    for row in all_status():
        lines.append(f"  {row['display']}: {row['status']} ({row['count']} foto)")
    logs = recent_logs(3)
    if logs:
        lines.append("Log terakhir:")
        for item in logs:
            mark = "TERIMA" if item.get("accepted") else "TOLAK"
            lines.append(f"  {mark} klaim={item.get('claimed')} {item.get('message')}")
    return lines


class FaceApp:
    def __init__(self) -> None:
        self.mode = Mode.LOGIN
        self.selected = ""
        self.burst = FrameBurst()
        self.message = "Login dengan Face ID"
        self.lockout = LockoutStore()
        self.identity_name: str | None = None
        self.identity_score: float | None = None
        self._identity_tick = 0
        self._hands = HandTracker()
        self.toast_text = ""
        self.toast_kind = "info"
        self.toast_ttl = 0
        self._faceid_cooldown = 20
        self._faceid_ready = 0
        self.session: str | None = None

    def flash(self, text: str, kind: str = "info", frames: int = 70) -> None:
        self.toast_text = text
        self.toast_kind = kind
        self.toast_ttl = frames

    def _select_name(self, key: int) -> None:
        rows = all_status()
        index = key - ord("1")
        if 0 <= index < len(rows):
            self.selected = rows[index]["name"]
            self.message = f"Nama: {rows[index]['display']}"

    def _refresh_identity(self, frame, valid: bool, detection) -> None:
        if self.mode == Mode.LOGIN:
            return
        if self.mode == Mode.ENROLL:
            self.identity_name = self.selected
            self.identity_score = None
            return
        if self.mode == Mode.VERIFY:
            if not self.message.startswith("TERIMA"):
                self.identity_name = self.selected
                self.identity_score = None
            return
        if not valid:
            self.identity_name = None
            self.identity_score = None
            return
        self._identity_tick += 1
        if self._identity_tick % 6 != 0 and self.identity_name is not None:
            return
        embedding = extract_embedding(frame, detection.faces[0])
        result = recognize_embedding(embedding)
        if result.status == "cocok":
            self.identity_name = result.best_name
            self.identity_score = result.best_score
        else:
            self.identity_name = None
            self.identity_score = None

    def _draw(self, frame, extra: list[str]) -> None:
        detection = detect_result(frame)
        vis, valid, detect_msg = annotate(
            frame, faces=detection.faces, detection=detection, show_hud=False
        )
        self._refresh_identity(frame, valid, detection)
        if self.mode == Mode.LOGIN:
            face = detection.faces[0] if detection.faces else None
            draw_login_page(vis, face, valid, self.message)
            if self.toast_ttl > 0:
                draw_toast(vis, self.toast_text, self.toast_kind, y=max(86, vis.shape[0] // 2 - 160))
                self.toast_ttl -= 1
            if self._faceid_cooldown > 0:
                self._faceid_cooldown -= 1
            return vis, valid
        gesture = self._hands.detect(frame) if valid and detection.faces else None
        if gesture == "three":
            vis = paste_sticker(vis, detection.faces[0], "fery")
            draw_custom_employee_label(vis, detection.faces[0], "Fery Irwandi")
        elif gesture == "peace":
            vis = paste_sticker(vis, detection.faces[0], "mulyadi")
            draw_custom_employee_label(vis, detection.faces[0], "Dedi Mulyadi")
        elif gesture == "one":
            vis = paste_sticker(vis, detection.faces[0], "megawati")
            draw_custom_employee_label(vis, detection.faces[0], "Megawati")
        elif valid and detection.faces:
            show_percent = self.mode in (Mode.RECOGNIZE, Mode.FACEID) or (
                self.mode == Mode.VERIFY and self.message.startswith("TERIMA")
            )
            draw_employee_label(
                vis,
                detection.faces[0],
                self.identity_name,
                self.identity_score if show_percent else None,
            )
        titles = {
            Mode.LOGIN: "Login",
            Mode.MENU: "Beranda",
            Mode.DETECT: "Deteksi",
            Mode.ADMIN: "Admin",
            Mode.ENROLL: "Pendaftaran",
            Mode.RECOGNIZE: "Kenali",
            Mode.VERIFY: "Verifikasi",
            Mode.FACEID: "Face ID",
        }
        draw_header(vis, titles.get(self.mode, "Deteksi"))
        accent = (255, 255, 255)
        if self.message.startswith("TERIMA") or self.message.startswith("BERHASIL"):
            accent = (92, 196, 88)
        elif self.message.startswith("TOLAK") or "gagal" in self.message.lower():
            accent = (82, 82, 230)
        draw_footer(vis, extra + [detect_msg, self.message], accent)
        if self.mode == Mode.ENROLL:
            row = next((item for item in all_status() if item["name"] == self.selected), None)
            count = row["count"] if row else 0
            bar_w = min(280, vis.shape[1] // 3)
            draw_progress(vis, vis.shape[1] - bar_w - 24, 72, bar_w, count / MAX_SAMPLES)
        if self.mode in (Mode.FACEID, Mode.LOGIN) and self._faceid_cooldown > 0:
            self._faceid_cooldown -= 1
        if self.toast_ttl > 0:
            draw_toast(vis, self.toast_text, self.toast_kind)
            self.toast_ttl -= 1
        return vis, valid

    def _handle_burst(self, frame, on_ready) -> None:
        if not self.burst.active:
            return
        err = self.burst.add_frame(frame)
        if len(self.burst.embeddings) >= MULTI_FRAME_COUNT:
            embedding, status = self.burst.complete()
            if embedding is None:
                self.message = status
                return
            on_ready(embedding)
        elif err:
            self.message = err

    def run(self) -> int:
        try:
            cap = open_camera()
        except CameraError as exc:
            print(exc)
            return 1
        print("Login Face ID. 2 daftar | Q keluar", flush=True)
        cv2.namedWindow("Deteksi Wajah", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Deteksi Wajah", 960, 540)
        try:
            cv2.setWindowProperty("Deteksi Wajah", cv2.WND_PROP_TOPMOST, 1)
        except cv2.error:
            pass
        ok, preview = cap.read()
        if ok:
            cv2.imshow("Deteksi Wajah", preview)
            cv2.waitKey(1)
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("Frame kamera gagal dibaca.")
                    return 1
                extra = self._overlay_for_mode()
                vis, valid = self._draw(frame, extra)
                if self.mode in (Mode.FACEID, Mode.LOGIN):
                    self._maybe_auto_faceid(frame, valid)
                if self.mode == Mode.RECOGNIZE:
                    self._handle_burst(frame, self._on_recognize)
                elif self.mode == Mode.VERIFY:
                    self._handle_burst(frame, self._on_verify)
                cv2.imshow("Deteksi Wajah", vis)
                key = cv2.waitKey(1) & 0xFF
                if key == 255:
                    continue
                if self._handle_key(key, frame, valid):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
        return 0

    def _overlay_for_mode(self) -> list[str]:
        if self.mode == Mode.LOGIN:
            return ["Lihat kamera untuk login     2  Daftar     Q  keluar"]
        if self.mode == Mode.MENU:
            return [
                "1  Deteksi     2  Daftar     5  Face ID",
                "3  Kenali      4  Verifikasi     L  Logout     Q  keluar",
            ]
        if self.mode == Mode.DETECT:
            return ["Deteksi wajah  |  B kembali  |  Q keluar"]
        if self.mode == Mode.ADMIN:
            return [
                "Halaman Admin",
                "5  Hapus user     S  Status",
                "B  kembali ke beranda     Q  keluar",
            ]
        if self.mode == Mode.ENROLL:
            row = next((item for item in all_status() if item["name"] == self.selected), None)
            label = person_display(self.selected) if self.selected else "-"
            count = row["count"] if row else 0
            status = row["status"] if row else "belum daftar"
            return [
                f"Mendaftarkan {pretty_name(label)}   {count}/{MAX_SAMPLES}   {status}",
                "SPASI ambil foto    N ganti nama    X ulang    B kembali",
            ]
        if self.mode == Mode.RECOGNIZE:
            progress = f" ambil {self.burst.progress}" if self.burst.active else ""
            return [f"Kenali siapa{progress}", "SPACE scan  B kembali"]
        if self.mode == Mode.FACEID:
            return [
                "Lihat kamera, atau tekan SPASI",
                "Cocok = TERIMA     Tidak kenal = TOLAK     B kembali",
            ]
        progress = f" ambil {self.burst.progress}" if self.burst.active else ""
        return [
            f"Verifikasi klaim {person_display(self.selected) if self.selected else '-' }{progress}",
            _name_help(),
            "SPACE scan (gerak kepala)  B kembali",
        ]

    def _on_recognize(self, embedding) -> None:
        result = recognize_embedding(embedding)
        print_recognition(result)
        self.message = result.message
        if result.status == "cocok":
            self.identity_name = result.best_name
            self.identity_score = result.best_score
            self.flash(
                f"Cocok: {pretty_name(person_display(result.best_name))}\n{score_percent(result.best_score)}%",
                "success",
            )
        else:
            self.identity_name = None
            self.identity_score = None
            self.flash(result.message, "error")

    def _on_verify(self, embedding) -> None:
        if not self.selected:
            self.message = "Isi nama klaim dulu"
            self.flash("Isi nama klaim dulu", "error")
            return
        result = verify_embedding(self.selected, embedding, lockout=self.lockout)
        print_verify(result)
        self.message = result.message
        if result.accepted:
            self.identity_name = self.selected
            self.identity_score = result.claimed_score
            self.flash(
                f"{result.message.split('  ')[0]}\n{score_percent(result.claimed_score)}%",
                "success",
            )
        else:
            self.identity_score = None
            claimed = pretty_name(person_display(self.selected))
            self.flash(f"TOLAK\nBukan {claimed}", "error")

    def _handle_key(self, key: int, frame, valid: bool) -> bool:
        if key in (ord("q"), ord("Q")):
            return True
        if self.mode == Mode.MENU and key == ord("9"):
            self.mode = Mode.ADMIN
            self.message = "Halaman admin"
            return False
        if self.mode == Mode.DETECT and key == ord("9"):
            self.mode = Mode.ADMIN
            self.message = "Halaman admin"
            return False
        if key in (ord("b"), ord("B"), 27):
            if self.mode == Mode.ENROLL:
                self.mode = Mode.LOGIN if self.session is None else Mode.MENU
                self.burst.active = False
                self._faceid_ready = 0
                self.message = "Login dengan Face ID" if self.mode == Mode.LOGIN else "Pilih menu"
                return False
            if self.mode in (Mode.DETECT, Mode.RECOGNIZE, Mode.VERIFY, Mode.ADMIN, Mode.FACEID):
                self.mode = Mode.MENU
                self.burst.active = False
                self._faceid_ready = 0
                self.message = "Pilih menu"
                return False
        if self.mode == Mode.LOGIN:
            return self._login_key(key, frame, valid)
        if self.mode == Mode.MENU:
            return self._home_key(key)
        if self.mode == Mode.ADMIN:
            return self._admin_key(key)
        if self.mode == Mode.ENROLL:
            self._enroll_key(key, frame, valid)
            return False
        elif self.mode == Mode.FACEID and key == 32:
            self._run_faceid(frame, valid)
            return False
        elif self.mode == Mode.RECOGNIZE and key == 32 and not self.burst.active:
            self.burst.reset()
            self.message = "Gerakkan kepala sedikit..."
        elif self.mode == Mode.VERIFY:
            self._select_name(key)
            if key == 32 and not self.burst.active:
                self.burst.reset()
                self.message = "Gerakkan kepala sedikit..."
        return False

    def _maybe_auto_faceid(self, frame, valid: bool) -> None:
        if self._faceid_cooldown > 0:
            self._faceid_ready = 0
            return
        if not valid:
            self._faceid_ready = 0
            return
        self._faceid_ready += 1
        if self._faceid_ready >= 12:
            self._faceid_ready = 0
            self._run_faceid(frame, valid)

    def _run_faceid(self, frame, valid: bool) -> None:
        remaining = self.lockout.remaining_seconds("faceid")
        if remaining > 0:
            self.message = f"TOLAK: coba lagi dalam {remaining} detik"
            self.flash(f"TOLAK\nTunggu {remaining} detik", "error")
            self._faceid_cooldown = 40
            return
        if not valid:
            self.message = "TOLAK: harus tepat satu wajah"
            self.flash("TOLAK\nHarus satu wajah", "error")
            self._faceid_cooldown = 25
            return
        sample, err = analyze_image(frame)
        if sample is None:
            self.message = f"TOLAK: {err}"
            self.flash(f"TOLAK\n{err}", "error")
            self._faceid_cooldown = 25
            return
        result = decide_faceid(recognize_embedding(sample.embedding))
        append_verify_log(result)
        self.message = result.message
        self._faceid_cooldown = 80
        if result.accepted:
            self.lockout.register_success("faceid")
            self.identity_name = result.best_name
            self.identity_score = result.best_score
            label = pretty_name(person_display(result.best_name or ""))
            self.session = result.best_name
            self.flash(f"Login berhasil\n{label}\n{score_percent(result.best_score)}%", "success")
            if self.mode == Mode.LOGIN:
                self.mode = Mode.MENU
                self.message = f"Halo, {label}"
            else:
                self.message = result.message
        else:
            self.lockout.register_failure("faceid")
            self.identity_name = None
            self.identity_score = None
            self.flash(result.message, "error")

    def _start_enroll(self) -> bool:
        slug = ask_employee_name(f"Daftar Karyawan {COMPANY_NAME}")
        if not slug:
            self.flash("Pendaftaran dibatalkan", "info")
            self.message = "Daftar dibatalkan"
            return False
        self.selected = slug
        self.mode = Mode.ENROLL
        self.message = f"Daftar {pretty_name(person_display(slug))}"
        self.flash(f"Siap mendaftar\n{pretty_name(person_display(slug))}", "info")
        return False

    def _start_verify(self) -> bool:
        slug = ask_employee_name(f"Verifikasi {COMPANY_NAME}", create=False)
        if not slug:
            self.message = "Verifikasi dibatalkan"
            return False
        self.selected = slug
        self.mode = Mode.VERIFY
        self.message = f"Klaim {person_display(slug)}"
        return False

    def _home_key(self, key: int) -> bool:
        if key == ord("1"):
            self.mode = Mode.DETECT
            self.message = "Arahkan wajah ke kamera"
            return False
        if key == ord("2"):
            return self._start_enroll()
        if key == ord("3"):
            self.mode = Mode.RECOGNIZE
            self.message = "Tekan SPASI untuk kenali"
            return False
        if key == ord("4"):
            return self._start_verify()
        if key == ord("5"):
            self.mode = Mode.FACEID
            self._faceid_ready = 0
            self._faceid_cooldown = 20
            self.message = "Face ID: lihat kamera"
            self.flash("Face ID\nLihat kamera", "info")
            return False
        if key in (ord("l"), ord("L"), ord("0")):
            self.session = None
            self.identity_name = None
            self.identity_score = None
            self.mode = Mode.LOGIN
            self._faceid_ready = 0
            self._faceid_cooldown = 25
            self.message = "Login dengan Face ID"
            self.flash("Logout", "info")
            return False
        return False

    def _login_key(self, key: int, frame, valid: bool) -> bool:
        if key == ord("2"):
            return self._start_enroll()
        if key == 32:
            self._run_faceid(frame, valid)
        return False

    def _admin_key(self, key: int) -> bool:
        if key == ord("5"):
            slug = ask_employee_name("Hapus Karyawan", create=False, button="Hapus")
            if not slug:
                self.message = "Hapus dibatalkan"
                return False
            label = pretty_name(person_display(slug))
            if remove_person(slug):
                if self.selected == slug:
                    self.selected = ""
                if self.identity_name == slug:
                    self.identity_name = None
                    self.identity_score = None
                self.message = f"{label} dihapus"
                self.flash(f"User dihapus\n{label}", "success")
            else:
                self.message = f"{label} tidak ada di data"
                self.flash(f"Tidak ditemukan\n{label}", "error")
            return False
        if key in (ord("s"), ord("S")):
            for line in _status_lines():
                print(line)
            self.message = " | ".join(
                f"{row['display']}={row['status']}" for row in all_status()
            ) or "Belum ada karyawan terdaftar"
        return False

    def _enroll_key(self, key: int, frame, valid: bool) -> None:
        if key in (ord("n"), ord("N")):
            slug = ask_employee_name(f"Daftar Karyawan {COMPANY_NAME}")
            if slug:
                self.selected = slug
                self.message = f"Daftar {person_display(slug)}"
            return
        if key in (ord("x"), ord("X")):
            if self.selected:
                clear_person(self.selected)
                label = pretty_name(person_display(self.selected))
                self.message = f"{label} dihapus"
                self.flash(f"Data dihapus\n{label}", "info")
            return
        if key != 32:
            return
        if not self.selected:
            self.message = "Isi nama dulu (tombol N)"
            return
        if not valid:
            self.message = "Harus tepat satu wajah"
            self.flash("Gagal menyimpan\nHarus tepat satu wajah", "error")
            return
        sample, err = analyze_image(frame)
        if sample is None:
            self.message = err
            self.flash(f"Gagal menyimpan\n{err}", "error")
            return
        gallery = next((item for item in all_status() if item["name"] == self.selected), None)
        if gallery is None:
            self.message = "Nama belum siap, tekan N"
            self.flash("Gagal menyimpan\nNama belum siap", "error")
            return
        if gallery["count"] >= MAX_SAMPLES:
            self.message = f"Kuota {MAX_SAMPLES} foto penuh. Tekan X untuk ulang."
            self.flash("Kuota foto penuh", "error")
            return
        status = save_sample(self.selected, frame, sample.face)
        need = max(0, MIN_SAMPLES - status["count"])
        label = pretty_name(status["display"])
        if status["status"] == "terdaftar":
            self.message = f"BERHASIL {label} terdaftar ({status['count']} foto)"
            self.flash(f"BERHASIL TERDAFTAR\n{label}  ·  {status['count']} foto", "success")
            if self.session is None:
                self.flash(f"Akun siap\n{label}\nLogin Face ID", "success")
        else:
            extra = f" kurang {need}" if need else ""
            self.message = f"BERHASIL tersimpan {status['count']} foto ({status['status']}){extra}"
            self.flash(
                f"BERHASIL DISIMPAN\n{label}  ·  {status['count']}/{MIN_SAMPLES} foto",
                "success",
            )


def main(argv: list[str] | None = None) -> int:
    return FaceApp().run()


if __name__ == "__main__":
    sys.exit(main())
