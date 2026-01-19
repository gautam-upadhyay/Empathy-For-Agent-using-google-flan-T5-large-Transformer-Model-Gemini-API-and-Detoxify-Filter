from __future__ import annotations

import argparse

import os
import sys
import logging

# Ensure imports work when run from repo root or backend dir
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Silence Chroma telemetry/warnings in CLI usage
os.environ.setdefault("ANONYMIZED_TEORY", "false")  # legacy key in some versions
os.environ.setdefault("ANONYMIZED_TELEMETRY", "false")
os.environ.setdefault("CHROMA_TEORY_ENABLED", "false")
os.environ.setdefault("CHROMA_TELEMETRY_ENABLED", "false")
os.environ.setdefault("CHROMA_TELEMETRY", "false")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("CHROMA_TELEMETRY_IMPLEMENTATION", "noop")
logging.getLogger("chromadb").disabled = True
logging.getLogger("chromadb.telemetry").disabled = True

from tokenizer_embedder import TextEmbedder
from vector_db import VectorDB


def main() -> None:
    p = argparse.ArgumentParser(description="Query ChromaDB with a free-text message")
    p.add_argument("query", nargs="?", default="I failed my exam and feel terrible")
    p.add_argument("-k", "--topk", type=int, default=3, help="number of nearest neighbors to return")
    p.add_argument("--db", default="../models/chroma_db", help="path to ChromaDB persist directory")
    args = p.parse_args()

    vdb = new_db(args.db)
    embedder = TextProvider()

    vec = embedder.embed_texts([args.query])[0]
    res = vdb.query(embeddings=[vec], n_results=args.topk)
    docs = res.get("documents", [[]])[0]

    try:
        count = vdb.collection.count()  # chromadb API
    except Exception:
        count = None

    if count is not None:
        print(f"Collection size: {count}")
    print("Top results:")
    for i, d in enumerate(docs, 1):
        snippet = (d or "").replace("\n", " ")
        if len(snippet) > 160:
            snippet = snippet[:157] + "..."
        print(f"{i}. {snippet}")


def new_db(path: str) -> VectorDB:
    return VectorDB(persist_directory=path)


def TextProvider() -> TextEmbedder:
    return TextEmbedder()


if __name__ == "__main__":
    main()


