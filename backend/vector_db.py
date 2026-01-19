from __future__ import annotations

from typing import Any, Dict, List, Optional
import os
import logging
import numpy as np

# numpy 2.0+ dropped np.float_ alias used by chromadb; add compatibility shim.
if not hasattr(np, "float_"):
	np.float_ = np.float64

import chromadb
from chromadb.config import Settings as ChromaSettings


class VectorDB:
	def __init__(self, persist_directory: str, collection_name: str = "empathetic_contexts") -> None:
		# Proactively disable Chroma telemetry/warnings to avoid noisy output
		os.environ.setdefault("ANONYMIZED_TELEMETRY", "false")
		os.environ.setdefault("CHROMA_TELEMETRY_IMPLEMENTATION", "noop")
		os.environ.setdefault("CHROMA_TELEMETRY_ENABLED", "false")
		logging.getLogger("chromadb").disabled = True
		logging.getLogger("chromadb.telemetry").disabled = True
		# Initialize client with telemetry disabled in settings
		self.client = chromadb.PersistentClient(
			path=persist_directory,
			settings=ChromaSettings(anonymized_telemetry=False),
		)
		self.collection = self.client.get_or_create_collection(name=collection_name)

	def add(self, texts: List[str], embeddings: Optional[List[List[float]]] = None, metadatas: Optional[List[Dict[str, Any]]] = None, ids: Optional[List[str]] = None) -> None:
		payload: Dict[str, Any] = {"documents": texts}
		if embeddings is not None:
			payload["embeddings"] = embeddings
		if metadatas is not None:
			payload["metadatas"] = metadatas
		if ids is not None:
			payload["ids"] = ids
		self.collection.add(**payload)

	def query(self, embeddings: List[List[float]], n_results: int = 3) -> Dict[str, Any]:
		try:
			return self.collection.query(query_embeddings=embeddings, n_results=n_results)
		except TypeError as e:
			# Handle ChromaDB corruption: 'int' has no len()
			print(f"[VectorDB] Database corruption error: {e}")
			return {"documents": [[]], "metadatas": [[]], "distances": [[]]}
		except Exception as e:
			print(f"[VectorDB] Query failed: {e}")
			return {"documents": [[]], "metadatas": [[]], "distances": [[]]}
