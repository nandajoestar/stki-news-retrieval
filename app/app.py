"""Aplikasi web Flask untuk mesin pencari berita.

Jalankan dari root proyek:  python -m app.app
"""
from flask import Flask, abort, jsonify, render_template, request

from src.config import TOP_K
from src.search_engine import SearchEngine

app = Flask(__name__)
_engine = None


def get_engine() -> SearchEngine:
    """Muat indeks sekali saja (dibangun otomatis jika belum ada)."""
    global _engine
    if _engine is None:
        _engine = SearchEngine.load_or_build()
    return _engine


@app.route("/")
def index():
    return render_template("index.html", total=len(get_engine().documents))


@app.route("/search")
def search():
    query = request.args.get("q", "").strip()
    results = get_engine().search(query, top_k=TOP_K) if query else []
    return render_template("results.html", query=query, results=results)


@app.route("/document/<int:doc_id>")
def document(doc_id):
    doc = get_engine().get_document(doc_id)
    if doc is None:
        abort(404)
    return render_template("document.html", doc=doc)


@app.route("/api/search")
def api_search():
    query = request.args.get("q", "").strip()
    results = get_engine().search(query, top_k=TOP_K) if query else []
    return jsonify({"query": query, "count": len(results), "results": [
        {k: r[k] for k in ("id", "judul", "sumber", "waktu", "link", "score")}
        for r in results]})


if __name__ == "__main__":
    app.run(debug=True)
