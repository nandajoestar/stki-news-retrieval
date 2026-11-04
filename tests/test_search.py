"""Uji dasar pipeline: python -m pytest"""
import pandas as pd
import pytest

from src.preprocessing import clean_text, preprocess
from src.search_engine import SearchEngine
from src.tfidf import TfidfModel


def test_clean_text_removes_portal_noise():
    raw = "JAKARTA, KOMPAS.com – Harga beras naik. https://kompas.com/x"
    cleaned = clean_text(raw)
    assert "KOMPAS" not in cleaned and "http" not in cleaned
    assert clean_text("Baca berita dengan sedikit iklan, klik di sini Gempa terjadi") == "Gempa terjadi"


def test_clean_text_removes_tempo_and_detik_promos():
    tempo = ("Pemilu digelar Sabtu. Pilihan Editor: Berita Lain Yang Ditautkan "
             "Scroll ke bawah untuk melanjutkan membaca Baca berita dengan sedikit iklan, klik di sini Hasil resmi diumumkan.")
    out = clean_text(tempo).lower()
    assert "pilihan editor" not in out and "scroll" not in out and "iklan" not in out
    assert "pemilu digelar sabtu" in out and "hasil resmi diumumkan" in out
    detik = "Banjir surut. Simak selengkapnya di halaman selanjutnya: Warga kembali. Baca selengkapnya di sini."
    out = clean_text(detik).lower()
    assert "halaman selanjutnya" not in out and "selengkapnya" not in out
    assert "banjir surut" in out and "warga kembali" in out


def test_preprocess_lowercase_no_stopwords_no_digits():
    tokens = preprocess("Harga BERAS yang naik 2025 di pasar")
    assert "yang" not in tokens and "di" not in tokens and "2025" not in tokens
    assert all(t == t.lower() for t in tokens)


def test_tfidf_vectors_are_l2_normalized():
    model = TfidfModel().fit([["beras", "naik"], ["gempa", "bumi"], ["beras", "gempa"]])
    for vec in model.doc_vectors:
        assert sum(w * w for w in vec.values()) == pytest.approx(1.0)


def test_ranking_prefers_relevant_document(tmp_path):
    csv = tmp_path / "mini.csv"
    pd.DataFrame({
        "Judul": ["Harga beras naik", "Gempa guncang Maluku", "Timnas menang"],
        "Waktu": ["1 Jan 2025"] * 3, "Link": ["a", "b", "c"],
        "Content": ["Harga beras di pasar naik tajam pekan ini.",
                    "Gempa bumi magnitudo 5 guncang Maluku Utara.",
                    "Tim nasional sepak bola menang atas lawannya."],
        "tag1": ["ekonomi", "gempa", "bola"], "tag2": [None] * 3, "tag3": [None] * 3,
        "tag4": [None] * 3, "tag5": [None] * 3, "source": ["kompas"] * 3,
    }).to_csv(csv, index=False)
    engine = SearchEngine.build(csv, verbose=False)
    results = engine.search("gempa bumi", top_k=3)
    assert results and results[0]["id"] == 1
    assert engine.search("zzzxyz") == []
