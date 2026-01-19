from __future__ import annotations

import re
from typing import Dict, Tuple, List

try:
	from detoxify import Detoxify
except Exception:  # pragma: no cover - optional dependency handling
	Detoxify = None


class ToxicityFilter:
	def __init__(self, threshold: float = 0.5) -> None:
		self.threshold = threshold
		self.model = None
		# Lightweight fallback list for profanity when Detoxify is unavailable
		self._bad_words: List[str] = [
			"fuck", "shit", "bitch", "bastard", "asshole", "dick", "piss",
			"cunt", "slut", "faggot", "motherfucker", "whore",
		]
		self._safe_refusal = (
			"I can’t help with that. I can, however, help with a safer or legal alternative, "
			"or discuss the topic in a general, non-harmful way."
		)
		if Detoxify is not None:
			try:
				self.model = Detoxify("original")
			except Exception:
				self.model = None

	def is_toxic(self, text: str) -> Tuple[bool, Dict[str, float]]:
		if not text:
			return False, {}
		if self.model is None:
			# Simple heuristic: check for profanity when model isn't available
			lower = text.lower()
			for w in self._bad_words:
				if re.search(rf"\b{re.escape(w)}\b", lower):
					# mimic a minimal score payload
					return True, {w: 1.0}
			return False, {}
		scores = self.model.predict(text)
		max_score = max(scores.values()) if scores else 0.0
		return max_score >= self.threshold, {k: float(v) for k, v in scores.items()}

	def sanitize_response(self, text: str, scores: Dict[str, float]) -> str:
		# Strong mitigation: refuse and provide a safe alternative direction
		return self._safe_refusal
