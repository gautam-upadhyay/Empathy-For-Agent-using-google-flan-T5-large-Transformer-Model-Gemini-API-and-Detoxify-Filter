from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List

from config import settings
from model_training import EmotionDetector
from response_generation import EmpatheticResponder
from tokenizer_embedder import TextEmbedder
from responsible_filter import ToxicityFilter
from vector_db import VectorDB


def run_once(message: str, use_filter: bool, model_dir: str, persist_dir: str, temperature: float, max_context: int) -> Dict[str, Any]:
	embedder = TextEmbedder(model_name=settings.EMBEDDING_MODEL_NAME)
	vectordb = VectorDB(persist_directory=persist_dir)
	detector = EmotionDetector(model_dir=model_dir, base_model=settings.T5_MODEL_NAME)
	responder = EmpatheticResponder(api_key=settings.GEMINI_API_KEY, model_name=settings.GEMINI_MODEL_NAME)
	tox_filter = ToxicityFilter() if use_filter else None

	# 1) Retrieve context
	query_emb = embedder.embed_queries([message])[0]
	results = vectordb.query(embeddings=[query_emb], n_results=max_context)
	contexts: List[str] = results.get("documents", [[]])[0] if results else []

	# 2) Predict emotion
	emotion = detector.predict_emotion(message)

	# 3) Generate response
	response_text = responder.generate_response(
		user_message=message,
		detected_emotion=emotion,
		contexts=contexts,
		temperature=temperature,
		safety_mode=use_filter,
	)

	# 4) Optional responsible filtering
	toxic = False
	scores: Dict[str, float] = {}
	if tox_filter is not None:
		toxic, scores = tox_filter.is_toxic(response_text)
		if toxic:
			response_text = tox_filter.sanitize_response(response_text, scores)

	return {
		"emotion": emotion,
		"response": response_text,
		"toxic": bool(toxic),
		"toxicity_scores": scores,
		"context": contexts,
	}


def main() -> None:
	parser = argparse.ArgumentParser(description="CLI chat: empathy detector + Gemini, optional responsible filter")
	parser.add_argument("--message", type=str, required=True, help="User message text")
	parser.add_argument("--model_dir", type=str, default=settings.EMP_DETECTOR_PATH, help="Path to trained empathy detector")
	parser.add_argument("--persist", type=str, default=settings.CHROMA_DB_PATH, help="Path to ChromaDB persist directory")
	parser.add_argument("--temperature", type=float, default=settings.GENERATION_TEMPERATURE, help="Generation temperature")
	parser.add_argument("--max_context", type=int, default=settings.MAX_CONTEXT, help="Number of retrieved contexts")
	parser.add_argument("--use_filter", action="store_true", help="Enable responsible filtering (Detoxify)")
	args = parser.parse_args()

	out = run_once(
		message=args.message,
		use_filter=bool(args.use_filter),
		model_dir=args.model_dir,
		persist_dir=args.persist,
		temperature=float(args.temperature),
		max_context=int(args.max_context),
	)
	print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
	main()


