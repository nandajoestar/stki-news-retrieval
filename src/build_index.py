"""Bangun indeks pencarian dan simpan ke data/index/.

Jalankan dari root proyek:  python -m src.build_index
"""
import time

from .config import INDEX_PATH
from .search_engine import SearchEngine


def main():
    start = time.time()
    print("Membangun indeks (preprocessing + TF-IDF)...")
    engine = SearchEngine.build()
    engine.save()
    print(f"Selesai: {len(engine.documents)} dokumen, "
          f"{len(engine.model.idf)} term unik, {time.time() - start:.1f} detik.")
    print(f"Indeks disimpan di {INDEX_PATH}")


if __name__ == "__main__":
    main()
