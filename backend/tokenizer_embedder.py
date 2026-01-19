from __future__ import annotations

import argparse
import os
from typing import List, Optional

import pandas as pd
from sentence_transformers import SentenceTransformer

from config import settings
from vector_db import VectorDB


class TextEmbedder:
	def __init__(self, model_name: str = settings.EMBEDDING_MODEL_NAME, device: Optional[str] = None) -> None:
		self.model_name = model_name
		self.model = SentenceTransformer(model_name, device=device)
		# Encourage consistent embedding length for MPNet/E5
		try:
			self.model.max_seq_length = 384
		except Exception:
			pass

	def embed_queries(self, texts: List[str]) -> List[List[float]]:
		# E5 expects a "query: " prefix for queries
		prefixed = [f"query: {t}" for t in texts]
		embs = self.model.encode(prefixed, normalize_embeddings=True, convert_to_numpy=True)
		return [e.tolist() for e in embs]

	def embed_passages(self, texts: List[str]) -> List[List[float]]:
		# E5 expects a "passage: " prefix for documents
		prefixed = [f"passage: {t}" for t in texts]
		embs = self.model.encode(prefixed, normalize_embeddings=True, convert_to_numpy=True)
		return [e.tolist() for e in embs]

	# Backward compat: alias to passage embedding
	def embed_texts(self, texts: List[str]) -> List[List[float]]:
		return self.embed_passages(texts)


def _infer_texts(df: pd.DataFrame) -> List[str]:
	candidate_cols = [
		("text",),
		("utterance",),
		("context", "response"),
		("dialogue",),
	]
	for cols in candidate_cols:
		if all(c in df.columns for c in cols):
			if len(cols) == 1:
				return df[cols[0]].astype(str).fillna("").tolist()
			return (df[cols[0]].astype(str).fillna("") + " \n " + df[cols[1]].astype(str).fillna("")).tolist()
	raise ValueError("Could not infer text columns. Provide a CSV with 'text' or 'context'+'response'.")


def cli_embed_to_chroma(csv_path: str, persist_dir: str) -> None:
	df = pd.read_csv(csv_path)
	texts = _infer_texts(df)
	embedder = TextEmbedder()
	vectordb = VectorDB(persist_directory=persist_dir)

	BATCH = 2000  # safe chunk size under Chroma's default limits
	total = len(texts)
	for start in range(0, total, BATCH):
		end = min(start + BATCH, total)
		batch_texts = texts[start:end]
		batch_ids = [f"doc-{i}" for i in range(start, end)]
		batch_embs = embedder.embed_passages(batch_texts)
		vectordb.add(texts=batch_texts, embeddings=batch_embs, ids=batch_ids)
		print(f"Indexed {end}/{total}")

	print(f"Indexed {total} rows into {persist_dir}")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Tokenize/embed and populate ChromaDB")
	parser.add_argument("--embed", action="store_true", help="Create embeddings and store in ChromaDB")
	parser.add_argument("--input", type=str, default=os.path.normpath(os.path.join(os.path.dirname(__file__), "../datasets/processed/empathy_final.csv")))
	parser.add_argument("--persist", type=str, default=settings.CHROMA_DB_PATH)
	args = parser.parse_args()

	if args.embed:
		cli_embed_to_chroma(csv_path=args.input, persist_dir=args.persist)
	else:
		print("Nothing to do. Use --embed to build ChromaDB.")
