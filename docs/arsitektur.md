# Arsitektur & Rumus

```
CSV -> clean_text -> case folding -> tokenize -> stopword -> stemming
    -> TF-IDF (vektor L2) -> inverted index -> disimpan (pickle)

Query -> pipeline sama -> vektor query -> cosine via inverted index -> ranking top-k
```

| Komponen | Rumus / Metode |
|---|---|
| TF | `1 + log10(f)` |
| IDF | `log10(N / df) + 1` |
| Bobot | `TF x IDF`, lalu dibagi norma L2 |
| Cosine | dot product (vektor sudah ternormalisasi) |

Keputusan desain:
- TF-IDF ditulis manual agar tiap langkah bisa dijelaskan saat presentasi; scikit-learn tetap ada di requirements untuk perbandingan/evaluasi di notebook.
- Cache stemming (`data/index/stem_cache.json`) menghindari stemming ulang karena Sastrawi sangat lambat.
- Judul diulang 2x sebelum diproses sebagai pembobotan sederhana.
