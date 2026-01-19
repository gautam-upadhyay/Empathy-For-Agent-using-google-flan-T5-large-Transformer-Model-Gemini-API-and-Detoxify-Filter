from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import List

import numpy as np  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
from transformers import (  # type: ignore
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
import torch  # type: ignore
from config import settings


_ALLOWED_EMOTIONS = [
    "afraid","angry","annoyed","anticipating","anxious","apprehensive","ashamed","caring",
    "confident","content","devastated","disappointed","disgusted","embarrassed","excited",
    "faithful","furious","grateful","guilty","hopeful","impressed","jealous","joyful",
    "lonely","nostalgic","prepared","proud","relaxed","sad","sarcastic","sentimental",
    "surprised","terrified","trusting","neutral"
]

@dataclass
class _Example:
    text: str
    emotion: str


# -----------------------------
# 1️⃣ LOAD DATASET
# -----------------------------
def _load_dataset(csv_path: str) -> List[_Example]:
    df = pd.read_csv(csv_path)  # type: ignore
    if not ("text" in df.columns and "emotion" in df.columns):
        raise ValueError("Expected columns: text, emotion")

    # Clean and filter only allowed emotions
    df["text"] = df["text"].astype(str).str.strip()
    df["emotion"] = df["emotion"].astype(str).str.lower().str.strip()
    df = df[df["emotion"].isin(_ALLOWED_EMOTIONS)]
    df = df[df["text"] != ""].drop_duplicates(subset=["text", "emotion"])

    rows = [
        _Example(text=str(r["text"]).strip(), emotion=str(r["emotion"]).strip())
        for _, r in df.iterrows()
    ]
    return rows


# -----------------------------
# 2️⃣ TOKENIZATION FUNCTION
# -----------------------------
def _tokenize_fn(tokenizer, examples):
    # examples is a dict: {'text': [...], 'emotion': [...]}
    inputs = [f"emotion: {x}" for x in examples["text"]]
    targets = examples["emotion"]

    model_inputs = tokenizer(
		inputs, max_length=128, truncation=True, padding="max_length"
    )
    labels = tokenizer(
        targets, max_length=8, truncation=True, padding="max_length"
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# -----------------------------
# 3️⃣ TRAIN FUNCTION
# -----------------------------
def train(
    csv_path: str,
    out_dir: str,
    base_model: str = settings.T5_MODEL_NAME,
    epochs: int = 5,
    batch_size: int = 4,
    lr: float = 3e-5,
):
    print(f"🚀 Using model: {base_model}")
    print(f"📁 Dataset: {csv_path}")
    print(f"💾 Output: {out_dir}")

    data = _load_dataset(csv_path)
    print(f"✅ Loaded {len(data)} samples for training")

    train_data, val_data = train_test_split(data, test_size=0.1, random_state=42)
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    # Convert to HF Dataset format
    train_dicts = [{"text": d.text, "emotion": d.emotion} for d in train_data]
    val_dicts = [{"text": d.text, "emotion": d.emotion} for d in val_data]

    from datasets import Dataset  # Lazy import
    train_ds = Dataset.from_list(train_dicts).map(
        lambda ex: _tokenize_fn(tokenizer, ex),
        batched=True,
        remove_columns=["text", "emotion"],
    )
    val_ds = Dataset.from_list(val_dicts).map(
        lambda ex: _tokenize_fn(tokenizer, ex),
        batched=True,
        remove_columns=["text", "emotion"],
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(base_model)
    # Move to GPU and enable memory/perf optimizations when available
    if torch.cuda.is_available():
        print("GPU available: True")
        print("Using device:", torch.cuda.get_device_name(0))
        model = model.to("cuda")
        try:
            model.gradient_checkpointing_enable()
        except Exception:
            pass
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass
    else:
        print("GPU available: False (training on CPU)")

    # Keep batch size tiny for 6GB GPUs; clamp to 1 to avoid OOM with flan-t5-large.
    effective_batch = 1
    if batch_size != effective_batch:
        print(f"⚠️  Reducing batch_size from {batch_size} to {effective_batch} to avoid GPU OOM.")

    args = Seq2SeqTrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=effective_batch,
        per_device_eval_batch_size=effective_batch,
        learning_rate=lr,
        num_train_epochs=epochs,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        predict_with_generate=False,  # cheaper eval
        load_best_model_at_end=True,
        save_total_limit=2,
        report_to="none",
        dataloader_pin_memory=torch.cuda.is_available(),
        gradient_checkpointing=True,
        gradient_accumulation_steps=1,
        warmup_steps=500,
        weight_decay=0.01,
        bf16=(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
        fp16=(torch.cuda.is_available() and not torch.cuda.is_bf16_supported()),
        optim="adamw_torch",  # non-fused uses less memory
    )

    collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    print("🧠 Starting training ...")
    trainer.train()
    print("✅ Training complete!")

    # Save model and tokenizer
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"💾 Model and tokenizer saved to {out_dir}")

# -----------------------------
# 3.5️⃣ INFERENCE WRAPPER
# -----------------------------
class EmotionDetector:
	def __init__(self, model_dir: str, base_model: str = settings.T5_MODEL_NAME, device: str | None = None) -> None:
		"""
		Lightweight inference wrapper for the fine-tuned T5 emotion detector.
		Loads from a saved model directory (trainer.save_model).
		"""
		self.tokenizer = AutoTokenizer.from_pretrained(model_dir if os.path.isdir(model_dir) else base_model)
		self.model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
		if device is None:
			device = "cuda" if torch.cuda.is_available() else "cpu"
		self.device = device
		self.model.to(self.device)
		self.model.eval()

	def _normalize_label(self, label: str) -> str:
		label = (label or "").strip().lower()
		if not label:
			return "neutral"
		# keep first token, strip punctuation
		token = label.split()[0].strip(".,!?;:\"'()[]{}")
		if token in _ALLOWED_EMOTIONS:
			return token
		synonyms = {
			"joy": "joyful",
			"sadness": "sad",
			"fear": "afraid",
			"anger": "angry",
			"thankful": "grateful",
			"anxiety": "anxious",
			"loneliness": "lonely",
			"surprise": "surprised",
			"disgust": "disgusted",
			"shame": "ashamed",
			"hope": "hopeful",
			"pride": "proud",
			"relax": "relaxed",
			"trust": "trusting",
			"contented": "content",
		}
		mapped = synonyms.get(token, token)
		return mapped if mapped in _ALLOWED_EMOTIONS else "neutral"

	def predict_emotion(self, text: str) -> str:
		prompt = f"emotion: {str(text).strip()}"
		enc = self.tokenizer([prompt], max_length=256, truncation=True, return_tensors="pt")
		input_ids = enc["input_ids"].to(self.device)
		attention_mask = enc.get("attention_mask")
		if attention_mask is not None:
			attention_mask = attention_mask.to(self.device)
		with torch.no_grad():
			gen_ids = self.model.generate(
				input_ids=input_ids,
				attention_mask=attention_mask,
				max_new_tokens=4,
				num_beams=6,
				length_penalty=0.0,
			)
		out = self.tokenizer.decode(gen_ids[0], skip_special_tokens=True)
		return self._normalize_label(out)

# -----------------------------
# 4️⃣ MAIN ENTRY
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune FLAN-T5 for emotion detection")
    parser.add_argument("--data", type=str, required=True, help="Path to CSV with columns text,emotion")
    parser.add_argument("--out", type=str, required=True, help="Output dir for the fine-tuned model")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-5)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    train(
        csv_path=args.data,
        out_dir=args.out,
        base_model=settings.T5_MODEL_NAME,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
