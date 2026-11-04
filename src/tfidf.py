"""Pembobotan TF-IDF yang diimplementasikan manual (tanpa scikit-learn)
beserta inverted index untuk pencarian cepat.

  TF     = 1 + log10(f)            (log-scaled term frequency)
  IDF    = log10(N / df) + 1       (inverse document frequency, +1 agar tidak 0)
  bobot  = TF x IDF, lalu dinormalisasi L2 per dokumen
"""
import math
from collections import Counter, defaultdict


class TfidfModel:
    def __init__(self):
        self.idf = {}                 # term -> nilai idf
        self.doc_vectors = []         # per dokumen: {term: bobot ternormalisasi}
        self.inverted_index = {}      # term -> [(doc_id, bobot), ...]
        self.n_docs = 0

    # ---------- pelatihan ----------
    def fit(self, tokenized_docs: list) -> "TfidfModel":
        self.n_docs = len(tokenized_docs)
        df = Counter()
        for tokens in tokenized_docs:
            df.update(set(tokens))
        self.idf = {t: math.log10(self.n_docs / n) + 1 for t, n in df.items()}

        index = defaultdict(list)
        self.doc_vectors = []
        for doc_id, tokens in enumerate(tokenized_docs):
            vec = self._vectorize(Counter(tokens))
            self.doc_vectors.append(vec)
            for term, w in vec.items():
                index[term].append((doc_id, w))
        self.inverted_index = dict(index)
        return self

    # ---------- transformasi ----------
    def _vectorize(self, counts: Counter) -> dict:
        vec = {t: (1 + math.log10(f)) * self.idf[t]
               for t, f in counts.items() if t in self.idf}
        norm = math.sqrt(sum(w * w for w in vec.values()))
        return {t: w / norm for t, w in vec.items()} if norm else {}

    def transform_query(self, tokens: list) -> dict:
        return self._vectorize(Counter(tokens))

    # ---------- cosine similarity ----------
    def cosine_scores(self, query_vec: dict) -> dict:
        """Cosine similarity query vs dokumen. Vektor sudah L2-normal,
        sehingga cosine = dot product (dihitung lewat inverted index)."""
        scores = defaultdict(float)
        for term, qw in query_vec.items():
            for doc_id, dw in self.inverted_index.get(term, ()):
                scores[doc_id] += qw * dw
        return scores
