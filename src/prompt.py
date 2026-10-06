from __future__ import annotations

from .config import COMPANY_NAME, parse_person_name
from .gallery import ensure_person


def ask_employee_name(
    title: str = "Nama Karyawan",
    *,
    create: bool = True,
    button: str = "Simpan",
) -> str | None:
    """Tanya nama lewat jendela input. None jika dibatalkan."""
    import tkinter as tk
    from tkinter import font as tkfont

    result: list[str | None] = [None]

    root = tk.Tk()
    root.title(title)
    root.configure(bg="#12181f")
    root.resizable(False, False)
    root.attributes("-topmost", True)
    width, height = 420, 280
    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()
    root.geometry(f"{width}x{height}+{(screen_w - width) // 2}+{(screen_h - height) // 3}")

    title_font = tkfont.Font(family="Segoe UI", size=16, weight="bold")
    body_font = tkfont.Font(family="Segoe UI", size=11)
    small_font = tkfont.Font(family="Segoe UI", size=9)

    tk.Frame(root, bg="#2a8f4a", height=6).pack(fill="x")
    pad = tk.Frame(root, bg="#12181f")
    pad.pack(fill="both", expand=True, padx=28, pady=18)

    tk.Label(pad, text=COMPANY_NAME, fg="#e8c36a", bg="#12181f", font=small_font).pack(anchor="w")
    tk.Label(pad, text=title, fg="#f4f6f8", bg="#12181f", font=title_font).pack(anchor="w", pady=(4, 4))
    tk.Label(
        pad,
        text="Ketik nama lengkap karyawan." if create else "Ketik nama yang akan dihapus.",
        fg="#9aa7b5",
        bg="#12181f",
        font=body_font,
    ).pack(anchor="w", pady=(0, 12))

    entry = tk.Entry(
        pad,
        font=tkfont.Font(family="Segoe UI", size=13),
        bg="#1c2530",
        fg="#f4f6f8",
        insertbackground="#f4f6f8",
        relief="flat",
        highlightthickness=1,
        highlightbackground="#2a8f4a",
        highlightcolor="#2a8f4a",
    )
    entry.pack(fill="x", ipady=10)
    entry.focus_set()

    error = tk.Label(pad, text="", fg="#e06c75", bg="#12181f", font=small_font)
    error.pack(anchor="w", pady=(8, 0))

    def submit(_event=None) -> None:
        try:
            slug, display = parse_person_name(entry.get())
        except ValueError as exc:
            error.config(text=str(exc))
            return
        if create:
            ensure_person(slug, display)
        result[0] = slug
        root.destroy()

    def cancel(_event=None) -> None:
        result[0] = None
        root.destroy()

    buttons = tk.Frame(pad, bg="#12181f")
    buttons.pack(fill="x", pady=(18, 0))
    cancel_btn = tk.Button(
        buttons,
        text="Batal",
        command=cancel,
        bg="#1c2530",
        fg="#f4f6f8",
        relief="flat",
        font=body_font,
        padx=16,
        pady=8,
        cursor="hand2",
    )
    cancel_btn.pack(side="right")
    save_btn = tk.Button(
        buttons,
        text=button,
        command=submit,
        bg="#2a8f4a",
        fg="#ffffff",
        relief="flat",
        font=body_font,
        padx=18,
        pady=8,
        cursor="hand2",
    )
    save_btn.pack(side="right", padx=(0, 10))

    root.bind("<Return>", submit)
    root.bind("<Escape>", cancel)
    root.protocol("WM_DELETE_WINDOW", cancel)
    root.mainloop()
    return result[0]
