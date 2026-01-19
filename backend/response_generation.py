from __future__ import annotations

from typing import List, Optional

from config import settings

try:
	import google.generativeai as genai
except Exception:  # pragma: no cover
	genai = None


_SYSTEM_PROMPT = (
	"You are an empathetic, accurate, and concise assistant.\n"
	"Goals:\n"
	"- Prefer the provided context; if it is missing or irrelevant, answer from general knowledge without mentioning retrieval.\n"
	"- Lead with a direct, factual answer when the user asks a question.\n"
	"- Then add a single, specific validating line that fits their emotion.\n"
	"- Keep it short: 2-3 sentences total.\n"
	"Safety: avoid medical, legal, or harmful advice; stay respectful and non-judgmental.\n"
)

_SAFETY_ON_GUIDELINES = (
	"Safety mode: ON. Be careful, neutral, and non-judgmental.\n"
	"- Refuse requests for illegal, harmful, or abusive content and redirect to safe alternatives.\n"
	"- Avoid profanity, slurs, or harassing language even if the user uses it.\n"
)

_SAFETY_OFF_GUIDELINES = (
	"Safety mode: OFF. You may mirror the user's tone and formality, but stay respectful.\n"
	"- Never use slurs, hate speech, or threats.\n"
	"- Do not provide instructions for wrongdoing or harm.\n"
)


class EmpatheticResponder:
	def __init__(self, api_key: str = settings.GEMINI_API_KEY, model_name: str = settings.GEMINI_MODEL_NAME) -> None:
		self.api_key = api_key
		self.model_name = model_name
		self.enabled = bool(api_key and genai is not None)
		self.used_fallback = False
		self.last_error: Optional[str] = None
		self.used_model: Optional[str] = None
		self._cached_models: Optional[List[str]] = None
		if self.enabled:
			try:
				genai.configure(api_key=api_key)
			except Exception as exc:  # pragma: no cover - network/env specific
				self.last_error = f"gemini_config_error: {exc}"
				self.enabled = False
		else:
			if not api_key:
				self.last_error = "gemini_disabled: missing GEMINI_API_KEY"
			elif genai is None:
				self.last_error = "gemini_disabled: google-generativeai not installed"

	def generate_response(
		self,
		user_message: str,
		detected_emotion: Optional[str],
		contexts: List[str],
		temperature: float = 0.7,
		safety_mode: bool = True,
	) -> str:
		self.used_fallback = False
		self.last_error = None
		self.used_model = None
		prompt = self._build_prompt(
			user_message=user_message,
			emotion=detected_emotion,
			contexts=contexts,
			safety_mode=safety_mode,
		)
		if not self.enabled:
			self.used_fallback = True
			if self.last_error is None:
				self.last_error = "gemini_disabled: not enabled"
			return self._fallback(prompt, user_message, detected_emotion, reason=self.last_error)
		# Try configured model, then fall back to a small list of common models to avoid 404s
		candidates = self._candidate_models()

		last_exc: Optional[Exception] = None
		for model_name in candidates:
			try:
				model = genai.GenerativeModel(model_name)
				resp = model.generate_content(prompt, generation_config={"temperature": float(temperature)})
				text = (resp.text or "").strip()
				if text:
					self.used_model = model_name
					return text
				last_exc = None
				self.used_fallback = True
				self.last_error = f"gemini_empty_response (model={model_name})"
				return self._fallback(prompt, user_message, detected_emotion, reason=self.last_error)
			except Exception as exc:  # pragma: no cover - network/env specific
				last_exc = exc
				continue

		self.used_fallback = True
		self.last_error = f"gemini_error: {last_exc}" if last_exc else "gemini_error: unknown"
		return self._fallback(prompt, user_message, detected_emotion, reason=self.last_error)

	def _candidate_models(self) -> List[str]:
		candidates: List[str] = []
		if self.model_name:
			candidates.append(self.model_name)

		# Query available models that support generateContent; cache the list
		if genai is not None:
			try:
				if self._cached_models is None:
					models = genai.list_models()  # type: ignore[arg-type]
					supported = []
					for m in models:
						# some SDK versions expose 'name' and 'supported_generation_methods'
						name = getattr(m, "name", None)
						methods = getattr(m, "supported_generation_methods", []) or []
						if name and "generateContent" in methods:
							# strip possible 'models/' prefix
							supported.append(str(name).replace("models/", ""))
					self._cached_models = supported
				for name in self._cached_models or []:
					if name not in candidates:
						candidates.append(name)
			except Exception as exc:  # pragma: no cover - network/env specific
				self.last_error = f"list_models_error: {exc}"

		# Add a small fallback list at the end
		for alt in ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"]:
			if alt not in candidates:
				candidates.append(alt)
		return candidates

	def _build_prompt(self, user_message: str, emotion: Optional[str], contexts: List[str], safety_mode: bool) -> str:
		ctx = "\n".join([f"- {c}" for c in contexts[:3]]) if contexts else "(no retrieved context)"
		emo = emotion or "unknown"
		mode_guidance = _SAFETY_ON_GUIDELINES if safety_mode else _SAFETY_OFF_GUIDELINES
		return (
			f"{_SYSTEM_PROMPT}\n"
			f"{mode_guidance}"
			f"Detected emotion: {emo}.\n"
			f"Context (may be partial):\n{ctx}\n\n"
			f"User: {user_message}\n"
			f"Instructions:\n"
			f"- If the user asks a factual question, answer it directly first; use context if available.\n"
			f"- Then add 1 short empathetic line tailored to the detected emotion.\n"
			f"- Do not repeat the question. Do not pad with generic phrases. Keep it concise and grounded.\n"
		)

	def _fallback(self, prompt: str, user_message: str, emotion: Optional[str], reason: Optional[str] = None) -> str:
		label = emotion or "what you're feeling"
		# keep fallback short but indicate why we couldn't call Gemini for easier debugging
		reason_note = f" (fallback: {reason})" if reason else ""
		return (
			f"I hear how hard this is, and I appreciate you sharing it{reason_note}. It sounds like you may be experiencing {label}. "
			f"If you're comfortable, can you tell me a bit more about what's making this feel especially challenging right now?"
		)
