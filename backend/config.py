import os
from dataclasses import dataclass
try:
	# Load environment variables from backend/.env if present (non-fatal if missing)
	from dotenv import load_dotenv  # type: ignore
	load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))
except Exception:
	pass


@dataclass
class Settings:
	# API keys
	# Do NOT ship a default key; require environment variable so failures fall back cleanly.
	GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
	# Gemini model name (override via env if backend complains about availability)
	# Use the public v1beta-supported alias; override via env if needed.
	GEMINI_MODEL_NAME: str = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash-latest")

	# Paths
	BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
	CHROMA_DB_PATH: str = os.getenv(
		"CHROMA_DB_PATH",
		os.path.normpath(os.path.join(BASE_DIR, "../models/chroma_db")),
	)
	EMP_DETECTOR_PATH: str = os.getenv(
		"EMP_DETECTOR_PATH",
		os.path.normpath(os.path.join(BASE_DIR, "../models/empathy_detector")),
	)

	# Models
	# E5 embeddings give stronger retrieval; documents/queries need prefixes.
	EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "intfloat/e5-base-v2")
	# Use a smaller FLAN-T5 by default to fit 6GB GPUs; override via T5_MODEL_NAME env.
	T5_MODEL_NAME: str = os.getenv("T5_MODEL_NAME", "google/flan-t5-base")

	# Other config
	# Retrieve a bit more context to help grounding
	MAX_CONTEXT: int = int(os.getenv("MAX_CONTEXT", 5))
	# Lower temperature for factual grounding; override via env if needed
	GENERATION_TEMPERATURE: float = float(os.getenv("GEN_TEMPERATURE", 0.45))


settings = Settings()
