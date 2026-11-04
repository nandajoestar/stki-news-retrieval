# Sistem Temu Kembali Informasi (STKI) - Mesin Pencari Berita

Proyek Tugas Akhir mata kuliah **Sistem Temu Kembali Informasi (STKI)**. Aplikasi web ini adalah mesin pencari artikel berita berbahasa Indonesia yang menerapkan pemrosesan teks, pembobotan **TF-IDF** (implementasi manual), dan pemeringkatan dengan **Cosine Similarity** melalui *inverted index*.

## Struktur Direktori

```text
stki-final-project/
├── data/
│   ├── processed/indonesia_news_clean.csv   # Dataset berita (500 artikel: Kompas, Tempo, Detik)
│   └── index/                               # Indeks & cache stemming (dibuat otomatis, tidak di-commit)
├── src/                                     # Inti algoritma STKI
│   ├── config.py                            # Path & parameter
│   ├── preprocessing.py                     # Cleaning, case folding, tokenizing, stopword, stemming
│   ├── tfidf.py                             # TF, IDF, vektor ternormalisasi, inverted index, cosine
│   ├── search_engine.py                     # Build/load indeks, ranking, snippet
│   └── build_index.py                       # Skrip membangun indeks
├── app/                                     # Aplikasi web Flask
│   ├── app.py                               # Routing (/, /search, /document/<id>, /api/search)
│   ├── templates/                           # base, index, results, document
│   └── static/style.css
├── tests/test_search.py                     # Uji otomatis (pytest)
├── notebooks/
│   └── 01_eda.ipynb                         # Exploratory Data Analysis lengkap (17 bagian)
├── docs/arsitektur.md                       # Penjelasan alur & rumus
├── requirements.txt
└── README.md
```

## Cara Menjalankan

```bash
# 1. Buat virtual environment (opsional tapi disarankan)
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

# 2. Pasang dependensi
pip install -r requirements.txt

# 3. (Opsional) Eksplorasi data
jupyter notebook notebooks/01_eda.ipynb

# 4. Bangun indeks. Proses pertama lambat karena stemming Sastrawi,
#    hasil stemming di-cache sehingga build berikutnya cepat.
python -m src.build_index

# 5. Jalankan web (indeks dibangun otomatis bila belum ada)
python -m app.app              # buka http://127.0.0.1:5000

# 6. Jalankan pengujian
python -m pytest
```

Semua perintah dijalankan dari **root proyek**.

## Alur Sistem

1. **Preprocessing**: bersihkan noise portal berita (boilerplate Tempo, dateline Kompas, "Simak video" Detik, URL) → case folding → tokenizing → stopword removal → stemming (Sastrawi).
2. **Pembobotan TF-IDF**: `TF = 1 + log10(f)`, `IDF = log10(N/df) + 1`, vektor dokumen dinormalisasi L2. Judul dihitung dua kali agar lebih berpengaruh.
3. **Pencarian**: query diproses dengan pipeline yang sama, lalu skor cosine dihitung lewat inverted index dan diurutkan.
4. **Tampilan**: 10 hasil teratas dengan skor, cuplikan, dan tag; tersedia juga endpoint JSON `/api/search?q=...`.

## Catatan Dataset

Sumber: `final_merge_dataset.csv` (80.472 artikel), diambil 500 baris pertama. Format kolom `Waktu` berbeda antar sumber dan belum distandarkan.
