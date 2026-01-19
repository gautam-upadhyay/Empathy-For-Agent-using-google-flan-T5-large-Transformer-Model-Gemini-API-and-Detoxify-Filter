from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd


TYPO_MAP = {
    "sweatings": "sweating",
    "embarassed": "embarrassed",
    "defenitely": "definitely",
    "loosing": "losing",
}


def fix_text(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    # normalize whitespace and common artefacts
    s = s.replace("\x00", "").replace("_comma_", ",").replace("_period_", ".").replace("_exclamation_", "!")
    s = s.replace("\r", " ").replace("\n", " ")
    for src, dst in TYPO_MAP.items():
        s = re.sub(rf"\b{re.escape(src)}\b", dst, s, flags=re.IGNORECASE)
    # collapse whitespace
    return " ".join(s.split())


def main(path: str) -> None:
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"File not found: {p}")
    df = pd.read_csv(p)
    before = df["text"].astype(str).str.contains(r"\bsweatings\b", case=False, regex=True).sum()
    df["text"] = df["text"].apply(fix_text)
    after = df["text"].astype(str).str.contains(r"\bsweatings\b", case=False, regex=True).sum()
    df.to_csv(p, index=False)
    print(f"Fixed typos in {p}. 'sweatings' count: {before} -> {after}. Rows: {len(df)}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemError("Usage: python fix_typos.py <path-to-empathy_final.csv>")
    main(sys.argv[1])


