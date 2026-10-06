# Deteksi Wajah

Kamera live (Python + OpenCV + YuNet). Bukan website.

## Cara pakai

```powershell
cd d:\face
.\.venv\Scripts\activate
python main.py
```

Aplikasi mulai di **halaman login Face ID**.

| Saat | Tombol | Fungsi |
| --- | --- | --- |
| Login | Lihat kamera / SPASI | Masuk jika wajah terdaftar |
| Login | 2 | Daftar akun baru |
| Login | Q | Keluar |
| Setelah login | 1 | Deteksi |
| Setelah login | L | Logout |
| Setelah login | Q | Keluar |

Belum punya akun: tekan **2**, isi nama, SPASI foto (minimal 5), **B** kembali, lalu login Face ID.

Wajah tidak kenal → **TOLAK**, tetap di login.
Wajah cocok → masuk beranda.
