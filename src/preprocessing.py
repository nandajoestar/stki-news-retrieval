"""Tahap preprocessing teks berita: cleaning, case folding, tokenizing,
stopword removal, dan stemming (Sastrawi)."""
import json
import re
from pathlib import Path
from multiprocessing import Pool, cpu_count

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# Pola noise khas portal berita (Kompas, Tempo, Detik).
# Urutan penting: pola "Pilihan Editor" memakai penanda Tempo sebagai batas,
# jadi harus dijalankan sebelum penanda itu sendiri dihapus.
_NOISE_PATTERNS = [
    # --- Tempo ---
    re.compile(r"pilihan editor\s*:.{0,200}?(?=scroll ke bawah|baca berita dengan sedikit iklan|$)", re.I | re.S),
    re.compile(r"pilihan editor\s*:", re.I),                                   # sisa label
    re.compile(r"(?:tempo\.co)?editor\s*:.{0,60}?(?=scroll ke bawah|baca berita dengan sedikit iklan)", re.I | re.S),
    re.compile(r"scroll ke bawah untuk melanjutkan membaca", re.I),
    re.compile(r"baca berita dengan sedikit iklan,?\s*klik di sini", re.I),
    # --- Kompas ---
    re.compile(r"^\s*[A-Z][A-Za-z ,.\-]{2,40},\s*KOMPAS\.com\s*[–-]?", re.U),   # dateline
    # --- Detik ---
    re.compile(r"simak\s*(?:juga)?\s*:?\s*video\b.*?(?=\[gambas|$)", re.I),
    re.compile(r"\[gambas:[^\]]*\]", re.I),
    re.compile(r"simak [^.:]{0,60}halaman selanjutnya[^.:]{0,60}:?", re.I),
    re.compile(r"baca selengkapnya di sini\.?", re.I),
    # --- Umum ---
    re.compile(r"baca juga\s*:[^.]*\.?", re.I),
    re.compile(r"https?://\S+|www\.\S+"),
]

# Stopword bawaan Sastrawi + tambahan yang sering muncul di berita
EXTRA_STOPWORDS = {
    "kata", "ujar", "tutur", "kompas", "tempo", "detik", "com", "co", "id",
    "jakarta", "wib", "https", "http", "www", "hari", "tahun", "bulan",
    # kata umum hasil stemming yang muncul di >30% dokumen (lihat EDA bagian 12)
    "sebut", "jadi", "laku", "satu", "lebih", "hingga", "lalu", "sama",
    "banyak", "kemudian", "atas",
}

_stemmer = StemmerFactory().create_stemmer()
STOPWORDS = set(StopWordRemoverFactory().get_stop_words()) | EXTRA_STOPWORDS

# Stemming Sastrawi lambat (~40 ms/kata), jadi hasilnya disimpan di cache
# (memori + file JSON) sehingga build ulang indeks hanya perlu hitungan detik.
STEM_CACHE_PATH = Path(__file__).resolve().parent.parent / "data" / "index" / "stem_cache.json"
_stem_cache = {}
if STEM_CACHE_PATH.exists():
    _stem_cache = json.loads(STEM_CACHE_PATH.read_text(encoding="utf-8"))


def _stem(word: str) -> str:
    if word not in _stem_cache:
        _stem_cache[word] = _stemmer.stem(word)
    return _stem_cache[word]


def warm_stem_cache(words, workers=None, verbose=True) -> None:
    """Stem semua kata unik sekaligus (paralel) lalu simpan cache ke disk."""
    todo = sorted(set(words) - set(_stem_cache))
    if not todo:
        return
    workers = workers or max(1, cpu_count() - 1)
    if verbose:
        print(f"  stemming {len(todo)} kata unik dengan {workers} proses...")
    if workers == 1:
        stems = [_stemmer.stem(w) for w in todo]
    else:
        with Pool(workers) as pool:
            stems = pool.map(_stemmer.stem, todo, chunksize=200)
    _stem_cache.update(zip(todo, stems))
    STEM_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STEM_CACHE_PATH.write_text(json.dumps(_stem_cache, ensure_ascii=False), encoding="utf-8")


def base_tokens(text) -> list:
    """Cleaning -> case folding -> tokenizing -> stopword removal (tanpa stemming)."""
    return remove_stopwords(tokenize(case_folding(clean_text(text))))


def clean_text(text) -> str:
    """Hapus noise portal berita, URL, dan karakter non-teks."""
    if not isinstance(text, str):
        return ""
    text = text.replace("\xa0", " ")
    for pattern in _NOISE_PATTERNS:
        text = pattern.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def case_folding(text: str) -> str:
    return text.lower()


def tokenize(text: str) -> list:
    """Ambil token alfabet saja (angka & tanda baca dibuang)."""
    return re.findall(r"[a-z]{2,}", text)


def remove_stopwords(tokens: list) -> list:
    return [t for t in tokens if t not in STOPWORDS]


def stem_tokens(tokens: list) -> list:
    return [_stem(t) for t in tokens]


def preprocess(text, stemming: bool = True) -> list:
    """Pipeline lengkap: teks mentah -> daftar token bersih."""
    tokens = base_tokens(text)
    if stemming:
        tokens = remove_stopwords(stem_tokens(tokens))
    return tokens
