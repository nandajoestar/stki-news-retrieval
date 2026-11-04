"""Mesin pencari: membangun indeks dari dataset, menyimpan/memuatnya,
lalu melakukan ranking dokumen berdasarkan cosine similarity."""
import pickle
import re

import pandas as pd

from .config import DATASET_PATH, INDEX_PATH, SNIPPET_LENGTH, TOP_K
from .preprocessing import base_tokens, clean_text, preprocess, remove_stopwords, stem_tokens, warm_stem_cache
from .tfidf import TfidfModel


class SearchEngine:
    def __init__(self, documents: list, model: TfidfModel):
        self.documents = documents   # list of dict (judul, waktu, link, isi, ...)
        self.model = model

    # ---------- membangun & menyimpan ----------
    @classmethod
    def build(cls, dataset_path=DATASET_PATH, verbose: bool = True) -> "SearchEngine":
        df = pd.read_csv(dataset_path)
        df["Content"] = df["Content"].fillna("")
        documents, raw_tokens = [], []
        for i, row in df.iterrows():
            tags = [str(row[c]) for c in ("tag1", "tag2", "tag3", "tag4", "tag5")
                    if pd.notna(row[c])]
            documents.append({
                "id": int(i),
                "judul": row["Judul"],
                "waktu": row["Waktu"],
                "link": row["Link"],
                "sumber": row["source"],
                "tags": tags,
                "isi": clean_text(row["Content"]),
            })
            # judul diberi bobot ekstra dengan diulang 2x
            raw_tokens.append(base_tokens(f"{row['Judul']} {row['Judul']} {row['Content']}"))
        # stemming kata unik sekali saja (paralel + cache), lalu petakan ke dokumen
        warm_stem_cache({w for toks in raw_tokens for w in toks}, verbose=verbose)
        tokenized = [remove_stopwords(stem_tokens(toks)) for toks in raw_tokens]
        return cls(documents, TfidfModel().fit(tokenized))

    def save(self, path=INDEX_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"documents": self.documents, "model": self.model}, f)

    @classmethod
    def load(cls, path=INDEX_PATH) -> "SearchEngine":
        with open(path, "rb") as f:
            data = pickle.load(f)
        return cls(data["documents"], data["model"])

    @classmethod
    def load_or_build(cls, path=INDEX_PATH) -> "SearchEngine":
        if path.exists():
            return cls.load(path)
        engine = cls.build()
        engine.save(path)
        return engine

    # ---------- pencarian ----------
    def search(self, query: str, top_k: int = TOP_K) -> list:
        tokens = preprocess(query)
        qvec = self.model.transform_query(tokens)
        if not qvec:
            return []
        scores = self.model.cosine_scores(qvec)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        results = []
        for doc_id, score in ranked:
            doc = dict(self.documents[doc_id])
            doc["score"] = round(score, 4)
            doc["snippet"] = self._snippet(doc["isi"], query)
            results.append(doc)
        return results

    @staticmethod
    def _snippet(text: str, query: str, length: int = SNIPPET_LENGTH) -> str:
        """Cuplikan di sekitar kata query pertama yang ditemukan."""
        pos = 0
        for word in re.findall(r"\w{3,}", query.lower()):
            found = text.lower().find(word)
            if found != -1:
                pos = max(0, found - length // 4)
                break
        snippet = text[pos:pos + length].strip()
        return ("… " if pos else "") + snippet + (" …" if pos + length < len(text) else "")

    def get_document(self, doc_id: int):
        if 0 <= doc_id < len(self.documents):
            return self.documents[doc_id]
        return None
