# Labeling Guidelines for FER & Engagement (Positif / Netral / Negatif)

Tujuan: memberikan pedoman anotasi yang konsisten untuk dataset Facial Emotion Recognition (FER) yang akan dipakai sebagai input ke sistem fuzzy engagement. Skema ini sengaja dibuat minimal dan stabil untuk konteks mahasiswa menonton video pembelajaran.

## Prinsip umum
- Label diberikan per *temporal window* (rekomendasi: 0.5–1.0 detik), bukan per frame, untuk menangkap konteks dan mengurangi noise.
- Gunakan soft labels atau probabilitas bila annotator tidak yakin; simpan confidence metadata.
- Jika ekspresi berubah cepat dalam window → beri label kelas dominan atau tandai sebagai `Ambiguous`.
- Untuk proyek ini, gunakan **3 label utama saja**: `Positif`, `Netral`, `Negatif`.

## Definisi kelas
- **Positif**: ekspresi menyenangkan/tertarik/puas.
  - Indikator visual: senyum nyata (corner-of-mouth up), eye‑crinkle (AU6), alis rileks, wajah terbuka.
  - Contoh: tersenyum saat melihat slide lucu atau memahami konsep.

- **Netral**: tidak ada muatan emosi jelas.
  - Indikator: mulut rileks/tertutup, alis netral, tidak ada gerakan otot wajah ekstrem.
  - Contoh: menonton bagian video tanpa reaksi emosional.

- **Negatif**: ekspresi tidak menyenangkan (frustrasi, marah, sedih).
  - Indikator: mouth corner down, brows lowered/together, lip press, mata sempit.
  - Contoh: frustrasi karena penjelasan tidak jelas atau gagal memahami materi.

## Perlakuan untuk keadaan bingung
- `Bingung` **bukan kelas utama** dalam skema ini.
- Jika bingung masih ringan dan wajah tampak netral → labelkan `Netral`.
- Jika bingung disertai frustrasi atau tanda emosi negatif yang kuat → labelkan `Negatif`.
- Jika ragu, gunakan `Ambiguous` sebagai catatan, bukan kelas training.

## Anotasi temporal (prosedur)
1. Potong video menjadi window 1.0s (sliding window 0.5s overlap) atau gunakan 0.5s fixed jika responsivitas dibutuhkan.
2. Annotator menonton tiap window utuh sebelum memberi label.
3. Jika tidak yakin, beri `Ambiguous` dan catat confidence (0–1). Hindari multi-label untuk training utama.
4. Untuk dataset training, gunakan majority vote dari minimal 3 annotator; simpan distribusi label sebagai soft target.

## Pedoman pelabelan cepat (cheat‑sheet)
- Jika smile jelas >0.5s → `Positif`.
- Jika wajah datar tanpa tanda aksi otot → `Netral`.
- Jika cues frustasi (brows down + mouth corner down) → `Negatif`.
- Jika ada tanda bingung ringan tanpa emosi kuat → cenderung `Netral`.
- Jika bingung disertai frustrasi jelas → `Negatif`.

## Contoh kasus & keputusan
- Frame tunggal: jangan label per frame. Ambiguity tinggi → tandai `Ambiguous`.
- Senyum singkat (<0.3s) di tengah → abaikan kecuali berulang/berdurasi.
- Ekspresi campuran: pilih label dominan >0.5s; jika tetap ambigu, tandai `Ambiguous`.

## Metadata yang harus disimpan per sample/window
- `timestamp_start`, `timestamp_end`
- `label_distribution` (contoh: {"Positif":0.7, "Netral":0.3})
- `majority_label` (jika ada)
- `avg_confidence` (dari annotator)
- optional: `notes` untuk kasus khusus

## Mapping ke fuzzy engine (praktis)
- Simpan label distribution (soft) dari annotator dan gunakan langsung sebagai probabilitas input ke FIS (lebih baik daripada argmax).
- Jika hanya ada majority_label, gunakan mapping crisp berikut sebagai nilai emosi (0–10):
  - Negatif = 0.0
  - Netral  = 5.0
  - Positif = 10.0
- Rekomendasi: prefer soft mapping bila memungkinkan.
- Jika Anda tetap ingin menyimpan informasi bingung, simpan di `notes` atau `metadata`, bukan sebagai label utama.

## Tips anotator & quality control
- Beri annotator 20–30 contoh per kelas saat training anotasi.
- Hitung Cohen's kappa / Fleiss' kappa; jika <0.6, perjelas pedoman dan ulangi training.
- Simpan contoh kontroversial untuk adjudication oleh expert.

---

Jika Anda ingin saya tambahkan template CSV/JSON untuk anotasi atau contoh gambar yang diberi label, saya bisa buatkan file `annotation_template.csv` dan beberapa contoh anotasi. 