from __future__ import annotations

import os
from typing import Any, Dict, List

from flask import Flask, jsonify, request
from flask_cors import CORS

from config import settings
from responsible_filter import ToxicityFilter
from response_generation import EmpatheticResponder
from tokenizer_embedder import TextEmbedder
from vector_db import VectorDB
from model_training import EmotionDetector
from user_memory import UserMemory


def create_app() -> Flask:
	app = Flask(__name__)
	CORS(app)

	# Initialize pipeline components
	embedder = TextEmbedder(model_name=settings.EMBEDDING_MODEL_NAME)
	vectordb = VectorDB(persist_directory=settings.CHROMA_DB_PATH)
	detector = EmotionDetector(model_dir=settings.EMP_DETECTOR_PATH, base_model=settings.T5_MODEL_NAME)
	responder = EmpatheticResponder(api_key=settings.GEMINI_API_KEY, model_name=settings.GEMINI_MODEL_NAME)
	tox_filter = ToxicityFilter()
	user_memory = UserMemory(embedder=embedder)

	@app.get("/api/health")
	def health() -> Any:
		return jsonify({"status": "ok"})

	@app.post("/api/chat")
	def chat() -> Any:
		data: Dict[str, Any] = request.get_json(force=True, silent=True) or {}
		message: str = (data.get("message") or "").strip()
		use_filter: bool = bool(data.get("use_filter", True))
		user_id: str = data.get("user_id", "default")
		conversation_id: str = data.get("conversation_id", "")
		if not message:
			return jsonify({"error": "message is required"}), 400

		# 1) Retrieve relevant user memories (conversations + facts)
		memories = user_memory.retrieve_relevant_memories(message, user_id=user_id, n_results=10)
		memory_context = user_memory.format_memories_for_prompt(memories)

		# 2) Embed and retrieve semantic context
		query_emb = embedder.embed_queries([message])[0]
		results = vectordb.query(embeddings=[query_emb], n_results=settings.MAX_CONTEXT)
		contexts: List[str] = results.get("documents", [[]])[0] if results else []

		# 3) Add memory context to contexts if available
		if memory_context:
			contexts = [memory_context] + contexts

		# 4) Predict emotion
		emotion = detector.predict_emotion(message)

		# 5) Generate empathetic response
		response_text = responder.generate_response(
			user_message=message,
			detected_emotion=emotion,
			contexts=contexts,
			temperature=settings.GENERATION_TEMPERATURE,
			safety_mode=use_filter,
		)

		# 6) Optional Responsible AI filter
		toxic, scores = False, {}
		if use_filter:
			toxic, scores = tox_filter.is_toxic(response_text)
			if toxic:
				response_text = tox_filter.sanitize_response(response_text, scores)

		# 7) Save ENTIRE conversation turn to memory (user message + bot response)
		saved_turn = user_memory.save_conversation_turn(
			user_message=message,
			bot_response=response_text,
			emotion=emotion,
			user_id=user_id,
			conversation_id=conversation_id,
		)

		# Get memory stats
		memory_stats = user_memory.get_memory_stats(user_id=user_id)

		payload = {
			"emotion": emotion,
			"response": response_text,
			"toxic": bool(toxic),
			"toxicity_scores": scores,
			"context": contexts,
			"use_filter": use_filter,
			"using_fallback": responder.used_fallback,
			"llm_enabled": responder.enabled,
			"llm_error": responder.last_error,
			"llm_model": responder.used_model or settings.GEMINI_MODEL_NAME,
			"llm_candidates": getattr(responder, "_cached_models", None),
			"filter_applied": bool(use_filter and (toxic or scores)),
			"memories_used": len(memories),
			"conversation_saved": saved_turn,
			"memory_stats": memory_stats,
		}
		return jsonify(payload)

	@app.get("/api/memory")
	def get_memories() -> Any:
		"""Get all stored memories for a user."""
		user_id = request.args.get("user_id", "default")
		memories = user_memory.get_all_memories(user_id=user_id)
		stats = user_memory.get_memory_stats(user_id=user_id)
		return jsonify({
			"user_id": user_id,
			"memories": memories,
			"count": len(memories),
			"stats": stats,
		})

	@app.delete("/api/memory")
	def clear_memories() -> Any:
		"""Clear all memories for a user."""
		user_id = request.args.get("user_id", "default")
		success = user_memory.clear_memories(user_id=user_id)
		return jsonify({"user_id": user_id, "cleared": success})

	@app.post("/api/memory")
	def save_memory() -> Any:
		"""Manually save a fact/memory."""
		data: Dict[str, Any] = request.get_json(force=True, silent=True) or {}
		fact = data.get("fact", "").strip()
		user_id = data.get("user_id", "default")
		if not fact:
			return jsonify({"error": "fact is required"}), 400
		success = user_memory.save_fact(fact, user_id=user_id)
		return jsonify({"saved": success, "fact": fact})

	return app


if __name__ == "__main__":
	port = int(os.getenv("PORT", 5000))
	app = create_app()
	app.run(host="0.0.0.0", port=port, debug=True)
