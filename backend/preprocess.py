from __future__ import annotations

import argparse
import os
from typing import List, Tuple
import warnings

# Suppress numpy MINGW-W64 warnings on Windows
warnings.filterwarnings('ignore', category=RuntimeWarning, module='numpy')
warnings.filterwarnings('ignore', message='.*MINGW.*')

import pandas as pd

import csv
import re

# Allowed emotions derived from Empathetic Dialogues taxonomy (plus neutral)
_ALLOWED_EMOTIONS = {
    "afraid","angry","annoyed","anticipating","anxious","apprehensive","ashamed","caring",
    "confident","content","devastated","disappointed","disgusted","embarrassed","excited",
    "faithful","furious","grateful","guilty","hopeful","impressed","jealous","joyful",
    "lonely","nostalgic","prepared","proud","relaxed","sad","sarcastic","sentimental",
    "surprised","terrified","trusting","neutral"
}

# Light synonym/normalization map for common variants
_SYNONYMS = {
    "joy": "joyful",
    "happiness": "joyful",
    "happy": "joyful",
    "fear": "afraid",
    "anger": "angry",
    "thankful": "grateful",
    "sadness": "sad",
    "anxiety": "anxious",
    "nervous": "anxious",
    "faith": "faithful",
    "contented": "content",
    "loneliness": "lonely",
    "surprise": "surprised",
}

def _read_csv_robust(path: str) -> pd.DataFrame:
    # 1) normal fast path
    try:
        return pd.read_csv(path, dtype=str, encoding_errors="ignore", low_memory=False)
    except Exception:
        pass
    # 2) auto-detect delimiter, skip bad lines (no low_memory with python engine)
    try:
        return pd.read_csv(
            path,
            engine="python",
            sep=None,              # auto-detect delimiter
            on_bad_lines="skip",   # skip malformed rows
            quotechar='"',
            escapechar='\\',
            dtype=str,
            encoding_errors="ignore",
        )
    except Exception:
        # 3) fallback to tab-delimited
        return pd.read_csv(
            path,
            sep="\t",
            engine="python",
            on_bad_lines="skip",
            dtype=str,
            encoding_errors="ignore",
        )


def clean_text(t: str) -> str:
	t = (t or "").strip()
	# Replace dataset artifacts and normalize whitespace
	t = (t.replace("\x00", "")
	     .replace("_comma_", ",")
	     .replace("_period_", ".")
	     .replace("_exclamation_", "!")
	     .replace("\r", " ")
	     .replace("\n", " "))
	# Fix a few common typos/variants seen in dataset
	typo_map = {
		"sweatings": "sweating",
		"embarassed": "embarrassed",
		"defenitely": "definitely",
		"loosing": "losing",
	}
	for _src, _dst in typo_map.items():
		# replace whole-word occurrences, case-insensitive
		t = re.sub(rf"\b{re.escape(_src)}\b", _dst, t, flags=re.IGNORECASE)
	return " ".join(t.split())


def normalize_emotion(label: str) -> str:
	"""Normalize an emotion label into the allowed taxonomy, otherwise neutral."""
	lbl = clean_text(str(label)).lower()
	if not lbl:
		return "neutral"
	# Keep first token to avoid multi-word drift
	token = lbl.split()[0]
	token = _SYNONYMS.get(token, token)
	return token if token in _ALLOWED_EMOTIONS else "neutral"


def load_emp_dialogues(raw_dir: str) -> pd.DataFrame:
	# Expect files similar to EmpatheticDialogues: train.csv/dev.csv/test.csv with context + utterance + emotion
	frames: List[pd.DataFrame] = []
	for name in ["train.csv", "dev.csv", "valid.csv", "test.csv"]:
		path = os.path.join(raw_dir, name)
		if os.path.isfile(path):
			df = _read_csv_robust(path)
			frames.append(df)
	if not frames:
		raise FileNotFoundError("No dataset files found in raw directory.")
	df = pd.concat(frames, ignore_index=True)
	# Normalize expected columns
	cols = set(df.columns)
	text_col = None
	if {"context", "utterance"}.issubset(cols):
		# Strip leading emotion token from context to avoid label leakage
		ctx = df["context"].astype(str).str.strip()
		first = ctx.str.split().str[0].str.lower()
		mask = first.isin(_ALLOWED_EMOTIONS)
		ctx_wo_label = ctx.where(~mask, ctx.str.split().str[1:].str.join(" "))
		df["text"] = (ctx_wo_label.fillna("") + " \n " + df["utterance"].astype(str).fillna("")).str.strip()
		text_col = "text"
	else:
		possible_text_cols = ["utterance", "response", "dialogue", "text", "context"]
		text_col = next((c for c in possible_text_cols if c in df.columns), None)
		if text_col is None:
			raise ValueError("No text column found (context+utterance / utterance / response / dialogue / text)")
	emotion_col = "emotion" if "emotion" in df.columns else None
	if emotion_col is None and "label" in df.columns:
		emotion_col = "label"
	# If still missing, derive from the first token of 'context' when available
	if emotion_col is None:
		if "context" in df.columns:
			cand = df["context"].astype(str).str.strip().str.split().str[0].str.lower()
			cand = cand.map(lambda x: _SYNONYMS.get(x, x))
			df["emotion"] = cand.where(cand.isin(_ALLOWED_EMOTIONS), "neutral")
			emotion_col = "emotion"
		else:
			df["emotion"] = "neutral"
			emotion_col = "emotion"
	df = df[[text_col, emotion_col]].rename(columns={text_col: "text", emotion_col: "emotion"})
	df["text"] = df["text"].astype(str).map(clean_text).str.slice(0, 1200)
	df["emotion"] = df["emotion"].map(normalize_emotion)
	df = df[df["text"] != ""]
	df = df[df["emotion"].isin(_ALLOWED_EMOTIONS)]
	df = df.drop_duplicates(subset=["text", "emotion"])
	return df


def write_toy_dataset(out_path: str) -> None:
	toy = pd.DataFrame(
		{
			"text": [
				"I failed my exam and feel terrible",
				"I got a promotion today!",
				"My friend ignored me and I'm angry",
				"I feel so alone lately",
			],
			"emotion": ["sad", "joyful", "angry", "lonely"],
		}
	)
	os.makedirs(os.path.dirname(out_path), exist_ok=True)
	toy.to_csv(out_path, index=False)
	print(f"Wrote toy dataset to {out_path}")


def main(raw_dir: str, out_csv: str) -> None:
	try:
		df = load_emp_dialogues(raw_dir)
	except Exception as e:
		print(f"Warning: {e}. Writing a small toy dataset instead.")
		write_toy_dataset(out_csv)
		return
	os.makedirs(os.path.dirname(out_csv), exist_ok=True)
	df.to_csv(out_csv, index=False)
	print(f"Wrote processed dataset with {len(df)} rows to {out_csv}")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Preprocess Empathetic Dialogues dataset")
	parser.add_argument("--input", type=str, required=True, help="Path to raw dataset directory")
	parser.add_argument("--output", type=str, required=True, help="Path to output CSV")
	args = parser.parse_args()
	main(raw_dir=args.input, out_csv=args.output)
