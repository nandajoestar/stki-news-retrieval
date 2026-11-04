"""Konfigurasi path dan parameter proyek."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "data" / "processed" / "indonesia_news_clean.csv"
INDEX_PATH = BASE_DIR / "data" / "index" / "search_index.pkl"

TOP_K = 10            # jumlah hasil default per pencarian
SNIPPET_LENGTH = 250  # panjang cuplikan teks pada halaman hasil
