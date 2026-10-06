# Roadmap: Sistem Verifikasi Wajah 5 Tokoh

Dokumen ini merinci langkah pembangunan aplikasi yang **mengenali wajah**, **mendaftarkan 5 orang**, dan **menolak klaim nama yang tidak cocok** (contoh: wajah Prabowo mengaku Anies).

Tokoh yang didaftarkan:

1. Prabowo
2. Anies
3. Jokowi
4. Mulyadi
5. Ahok

Status saat ini: **Fase 0–7 selesai.** Daftar wajah 5 tokoh masih langkah pengguna (`python -m src.enroll` atau UI `python main.py`).

---

## Tujuan produk

- Webcam atau foto dipakai untuk mendeteksi wajah.
- Tiap tokoh punya data wajah tersimpan (enrollment).
- Saat login/verifikasi, pengguna **mengklaim nama**.
- Sistem membandingkan wajah dengan data nama yang diklaim.
- Jika wajah lebih mirip orang lain, atau tidak kenal, permintaan **ditolak** beserta alasan.

Bukan hanya “ada wajah atau tidak”. Ini **verifikasi identitas**.

---

## Prinsip pencegahan error

| Situasi | Hasil yang diharapkan |
| --- | --- |
| Klaim Anies + wajah Anies | Diterima |
| Klaim Anies + wajah Prabowo | Ditolak: wajah tidak cocok dengan Anies |
| Klaim Anies + orang di luar 5 tokoh | Ditolak: tidak dikenali |
| Tidak ada wajah di kamera | Ditolak: wajah tidak terdeteksi |
| Beberapa wajah sekaligus | Ditolak atau minta satu wajah saja |
| Foto HP di depan kamera (opsional belakangan) | Ditolak jika liveness aktif |

Ambang kemiripan (threshold) wajib ada. Jangan terima “orang terdekat” tanpa batas skor.

---

## Tumpukan teknologi yang disarankan

Untuk demo Windows di `d:\face`:

- **Python 3.10+**
- **OpenCV** — kamera, YuNet (deteksi), SFace (embedding)
- **NumPy** — kemiripan vektor
- Data lokal: folder foto + `embeddings.json` + `verify_log.jsonl`

Alasan: ringan, bisa jalan di laptop, cukup akurat untuk 5 orang.

---

## Fase 0 — Siapkan proyek `[x]`

**Hasil:** lingkungan bisa dijalankan tanpa error install.

- [x] Buat struktur folder (`src`, `data/enrolled`, `data/raw`, `tests`)
- [x] `requirements.txt`
- [x] README cara install dan jalankan
- [x] Cek webcam Windows (`VideoCapture(0)`)
- [x] Script uji: jendela kamera terbuka, tekan Q untuk keluar

**Kriteria selesai:** `python -m src.camera_check` menampilkan video live. **Sudah.**

---

## Fase 1 — Deteksi wajah saja `[x]`

**Hasil:** sistem tahu ada berapa wajah, dan di mana posisinya.

- [x] Deteksi wajah di frame webcam
- [x] Deteksi wajah dari foto
- [x] Gambar kotak + jumlah wajah
- [x] Tolak frame jika 0 wajah atau lebih dari 1 wajah
- [x] Crop wajah untuk langkah berikutnya

**Kriteria selesai:** satu wajah = lanjut; selain itu pesan error jelas. **Sudah.**

---

## Fase 2 — Daftar 5 tokoh (enrollment) `[x]`

**Hasil:** tiap nama punya beberapa foto dan satu embedding rata-rata.

Alur daftar:

- [x] Pilih nama (salah satu dari 5)
- [x] Ambil 5–10 foto dari kamera atau file foto
- [x] Deteksi 1 wajah per foto
- [x] Simpan crop + embedding
- [x] Tandai status: `terdaftar` / `belum lengkap`

Aturan:

- [x] Nama di luar daftar 5 tokoh tidak boleh didaftarkan
- [x] Foto tanpa wajah ditolak
- [x] Embedding disimpan terpisah dari foto mentah (`data/embeddings.json`)
- [x] Boleh daftar ulang (replace) jika foto lama buruk
- [x] Data bisa diload ulang setelah program ditutup

**Kriteria selesai:** fitur enrollment untuk 5 nama siap. **Sudah.** Pengisian foto masing-masing tokoh masih langkah pengguna (`python -m src.enroll anies`, dst.).

---

## Fase 3 — Kenali siapa (recognition) `[x]`

**Hasil:** dari satu wajah, sistem mengeluarkan nama + skor kemiripan.

- [x] Hitung embedding wajah baru
- [x] Bandingkan ke 5 embedding tersimpan
- [x] Ambil kandidat skor tertinggi
- [x] Jika skor di bawah threshold → `tidak_dikenali`
- [x] Tampilkan 3 info: nama terdekat, skor, status (cocok / tidak kenal)

**Kriteria selesai:** uji foto masing-masing tokoh mayoritas teridentifikasi benar. **Logika diuji di `tests/test_verify_matrix.py`.** Akurasi nyata setelah 5 tokoh di-enroll.

---

## Fase 4 — Cegah salah klaim (inti permintaan Anda) `[x]`

**Hasil:** nama yang diklaim dicek ketat, bukan hanya “siapa yang paling mirip”.

Alur verifikasi:

- [x] Pengguna pilih klaim, misalnya `Anies`
- [x] Kamera atau foto ambil 1 wajah
- [x] Sistem hitung kemiripan ke Anies, dan ke 4 orang lain
- [x] Keputusan TERIMA / TOLAK sesuai ambang dan margin

Pesan error:

- [x] `Wajah tidak terdeteksi`
- [x] `Harus tepat satu wajah`
- [x] `Wajah tidak cocok dengan [nama klaim]`
- [x] `Wajah lebih mirip [nama lain] daripada [nama klaim]`
- [x] `[nama] belum terdaftar`
- [x] Log percobaan di `data/verify_log.jsonl`

**Kriteria selesai:** uji silang 5×5 (tiap orang mengklaim tiap nama). Hanya diagonal yang diterima. **Sudah di unit test.**

---

## Fase 5 — Antarmuka sederhana `[x]`

**Hasil:** bisa dipakai tanpa mengetik perintah panjang.

- [x] UI jendela OpenCV + keyboard (`python main.py`)
- [x] Daftar / Enrollment
- [x] Verifikasi (pilih nama → scan → hasil)
- [x] Status 5 tokoh: sudah daftar atau belum
- [x] Log percobaan terakhir

**Kriteria selesai:** alur daftar dan verifikasi jalan end-to-end dari UI. **Sudah.**

---

## Fase 6 — Perkuat anti-error `[x]`

**Hasil:** lebih sulit ditipu kondisi nyata.

- [x] Ambil beberapa frame, bukan 1 potret (rata-rata skor)
- [x] Tolak wajah terlalu kecil / terlalu miring
- [x] Peringatkan cahaya gelap (tolak jika terlalu gelap)
- [x] Liveness sederhana: gerakkan kepala kiri-kanan
- [x] Kunci percobaan gagal berulang (5x tolak → jeda 30 detik)

Ini fase pengaman, bukan syarat demo pertama. **Sudah.**

---

## Fase 7 — Uji dan rilis demo `[x]`

**Hasil:** demo stabil untuk 5 tokoh.

Matriks uji wajib: **lolos di `tests/test_verify_matrix.py`** (hanya diagonal TERIMA).

Uji tambahan:

- [x] Tanpa kamera / kamera dipakai aplikasi lain
- [x] Foto buram
- [x] Dua orang dalam frame
- [x] Nama belum di-enroll

**Kriteria selesai:** matriks uji logika lolos. Tes wajah nyata dilakukan setelah enrollment.

---

## Urutan pengerjaan yang disarankan

```text
[x] Fase 0  Siapkan proyek + webcam
   ↓
[x] Fase 1  Deteksi 1 wajah
   ↓
[x] Fase 2  Enrollment 5 tokoh
   ↓
[x] Fase 3  Recognition (siapa ini?)
   ↓
[x] Fase 4  Verifikasi klaim (anti ngaku-ngaku)
   ↓
[x] Fase 5  UI
   ↓
[x] Fase 6  Perkuat anti-error
   ↓
[x] Fase 7  Uji matriks 5x5 + rilis demo
```

Jangan loncat ke UI sebelum Fase 4 jalan di script, supaya logika tolak/terima sudah benar.

---

## Perkiraan waktu (jika dikerjakan berurutan)

| Fase | Perkiraan | Keterangan | Status |
| --- | --- | --- | --- |
| 0 | 0,5 hari | Setup | [x] selesai |
| 1 | 0,5 hari | Deteksi | [x] selesai |
| 2 | 1 hari | Enrollment + penyimpanan | [x] selesai |
| 3 | 1 hari | Embedding + threshold | [x] selesai |
| 4 | 1 hari | Logika anti salah klaim | [x] selesai |
| 5 | 1–2 hari | UI | [x] selesai |
| 6 | 1 hari | Pengaman tambahan | [x] selesai |
| 7 | 0,5–1 hari | Uji | [x] selesai |

Total kasar: **sekitar 1–2 minggu** untuk demo yang rapi, atau **2–3 hari** jika hanya script + jendela OpenCV.

---

## Struktur folder target

```text
d:\face\
  ROADMAP.md
  README.md
  requirements.txt
  src\
    camera_check.py
    detect.py
    enroll.py
    recognize.py
    verify.py
    app.py
  data\
    enrolled\
      prabowo\
      anies\
      jokowi\
      mulyadi\
      ahok\
    embeddings.json
    verify_log.jsonl
  tests\
```

---

## Yang sengaja ditunda

Agar demo tidak membengkak:

- Database server
- Akun login berbasis password
- Pengenalan ribuan orang
- GPU / training model sendiri
- Klaim akurasi setara sistem resmi

Fokus dulu: **5 orang, daftar, klaim nama, tolak jika wajahnya orang lain.**

---

## Keputusan yang perlu dikunci sebelum coding

- [x] Input: webcam **dan** upload foto
- [x] UI: jendela OpenCV + keyboard (`python main.py`)
- [x] Liveness (anti foto HP): gerakkan kepala (Fase 6)
- [x] Foto enrollment: kamera live **atau** file foto

Semua fase 0–7 sudah diimplementasi.
