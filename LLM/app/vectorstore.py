from __future__ import annotations
import json
import os
from typing import List, Dict, Any

import chromadb
from chromadb.config import Settings

from .embedding import Embedder

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "book_summaries.json")
PERSIST_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_db")
COLLECTION_NAME = "book_summaries"


def _load_books() -> List[Dict[str, Any]]:
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _client() -> chromadb.Client:
    os.makedirs(PERSIST_DIR, exist_ok=True)
    return chromadb.PersistentClient(path=PERSIST_DIR, settings=Settings(anonymized_telemetry=False))


def get_or_create_collection():
    client = _client()
    try:
        return client.get_collection(COLLECTION_NAME)
    except Exception:
        return client.create_collection(COLLECTION_NAME)


def _slug(title: str) -> str:
    return "".join(c.lower() if c.isalnum() else "-" for c in title).strip("-")


def _sanitize_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Chroma acceptă DOAR primitive (str/int/float/bool/None) în metadata.
    Aici convertim *orice* listă/tuple/dict/alt tip -> string JSON sau listă ca string.
    """
    out: Dict[str, Any] = {}
    for k, v in meta.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = v
        elif isinstance(v, (list, tuple, set)):
            try:
                out[k] = ", ".join(str(x) for x in v)
            except Exception:
                out[k] = json.dumps(list(v), ensure_ascii=False)
        elif isinstance(v, dict):
            out[k] = json.dumps(v, ensure_ascii=False)
        else:
            out[k] = str(v)
    return out


def seed_if_empty() -> bool:
    """
    Seed ChromaDB with books if the collection is empty.
    Returns True if seeding occurred, False otherwise.
    """
    col = get_or_create_collection()
    count = col.count()
    if count > 0:
        return False

    books = _load_books()
    embedder = Embedder()
    ids, docs, metas = [], [], []

    for book in books:
        title = book["title"]
        themes_list = book.get("themes", [])
        themes = ", ".join(themes_list) if isinstance(themes_list, list) else str(themes_list)

        content = (
            f"Title: {title}\n"
            f"Themes: {themes}\n"
            f"Short: {book['short_summary']}\n"
            f"Long: {book['long_summary']}"
        )

        ids.append(_slug(title))
        docs.append(content)
        metas.append(_sanitize_meta({"title": title, "themes": themes}))

    vectors = embedder.embed_texts(docs)
    col.add(ids=ids, embeddings=vectors, documents=docs, metadatas=metas)
    return True


def semantic_search(query: str, k: int = 5) -> List[Dict[str, Any]]:
    """
    Return top-k results from Chroma by embedding the query.
    """
    col = get_or_create_collection()
    qvec = Embedder().embed_text(query)
    res = col.query(query_embeddings=[qvec], n_results=k)
    results = []
    for i in range(len(res["ids"][0])):
        results.append({
            "id": res["ids"][0][i],
            "document": res["documents"][0][i],
            "metadata": res["metadatas"][0][i],
            "distance": res.get("distances", [[None]])[0][i] if "distances" in res else None
        })
    return results


if __name__ == "__main__":
    did_seed = seed_if_empty()
    print("Seeded ChromaDB." if did_seed else "ChromaDB already populated.")
