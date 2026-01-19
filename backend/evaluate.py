from __future__ import annotations

"""
Lightweight evaluation script for the empathetic responder.

It generates responses for a dataset of user messages and computes BLEU to
capture linguistic overlap (fluency/relevance proxy). Designed to be simple to
run for class demos or paper reporting.
"""

import argparse
from typing import Dict, List, Tuple

import pandas as pd
import sacrebleu
from tqdm import tqdm
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from config import settings
from model_training import EmotionDetector
from responsible_filter import ToxicityFilter
from response_generation import EmpatheticResponder
from tokenizer_embedder import TextEmbedder
from vector_db import VectorDB


def _load_pairs(path: str, limit: int | None) -> List[Tuple[str, str]]:
    df = pd.read_csv(path)
    if not {"text", "response"}.issubset(df.columns):
        raise ValueError("Expected columns: text, response")
    df = df.dropna(subset=["text", "response"]).copy()
    if limit is not None and limit > 0:
        df = df.head(limit)
    pairs = [(str(r["text"]).strip(), str(r["response"]).strip()) for _, r in df.iterrows()]
    return [(u, r) for u, r in pairs if u and r]


def _load_pairs_with_emotion(path: str, limit: int | None) -> List[Tuple[str, str]]:
    df = pd.read_csv(path)
    if not {"text", "reference_emotion"}.issubset(df.columns):
        raise ValueError("Expected columns: text, reference_emotion")
    df = df.dropna(subset=["text", "reference_emotion"]).copy()
    if limit is not None and limit > 0:
        df = df.head(limit)
    rows = [
        (str(r["text"]).strip(), str(r["reference_emotion"]).strip().lower())
        for _, r in df.iterrows()
    ]
    return [(t, e) for t, e in rows if t and e]


def _generate_predictions(
    data_path: str,
    limit: int | None,
    use_filter: bool,
    temperature: float,
    max_context: int,
    model_dir: str,
    persist_dir: str,
) -> Tuple[List[str], List[str], int]:
    """Shared generation helper returning predictions, references, and fallback count."""
    pairs = _load_pairs(data_path, limit)
    if not pairs:
        raise ValueError("No valid text/response pairs found for evaluation.")

    embedder = TextEmbedder(model_name=settings.EMBEDDING_MODEL_NAME)
    vectordb = VectorDB(persist_directory=persist_dir)
    detector = EmotionDetector(model_dir=model_dir, base_model=settings.T5_MODEL_NAME)
    responder = EmpatheticResponder(api_key=settings.GEMINI_API_KEY, model_name=settings.GEMINI_MODEL_NAME)
    tox_filter = ToxicityFilter() if use_filter else None

    preds: List[str] = []
    refs: List[str] = []
    fallback_count = 0

    for user_msg, ref_resp in tqdm(pairs, desc="Generating", unit="sample"):
        query_emb = embedder.embed_queries([user_msg])[0]
        results = vectordb.query(embeddings=[query_emb], n_results=max_context)
        contexts: List[str] = results.get("documents", [[]])[0] if results else []

        emotion = detector.predict_emotion(user_msg)
        gen_resp = responder.generate_response(
            user_message=user_msg,
            detected_emotion=emotion,
            contexts=contexts,
            temperature=temperature,
            safety_mode=use_filter,
        )

        if responder.used_fallback:
            fallback_count += 1

        if tox_filter is not None:
            toxic, scores = tox_filter.is_toxic(gen_resp)
            if toxic:
                gen_resp = tox_filter.sanitize_response(gen_resp, scores)

        preds.append(gen_resp)
        refs.append(ref_resp)

    return preds, refs, fallback_count


def evaluate_bleu(
    data_path: str,
    limit: int | None = None,
    use_filter: bool = False,
    temperature: float = settings.GENERATION_TEMPERATURE,
    max_context: int = settings.MAX_CONTEXT,
    model_dir: str = settings.EMP_DETECTOR_PATH,
    persist_dir: str = settings.CHROMA_DB_PATH,
) -> Dict[str, float]:
    """Generate responses for the dataset and compute corpus BLEU."""
    preds, refs, fallback_count = _generate_predictions(
        data_path=data_path,
        limit=limit,
        use_filter=use_filter,
        temperature=temperature,
        max_context=max_context,
        model_dir=model_dir,
        persist_dir=persist_dir,
    )

    bleu = sacrebleu.corpus_bleu(preds, [refs])
    return {
        "metric": "bleu",
        "bleu": float(bleu.score),
        "brevity_penalty": float(bleu.bp),
        "sys_len": float(bleu.sys_len),
        "ref_len": float(bleu.ref_len),
        "samples": len(preds),
        "fallback_responses": fallback_count,
    }


def evaluate_rouge_l(
    data_path: str,
    limit: int | None = None,
    use_filter: bool = False,
    temperature: float = settings.GENERATION_TEMPERATURE,
    max_context: int = settings.MAX_CONTEXT,
    model_dir: str = settings.EMP_DETECTOR_PATH,
    persist_dir: str = settings.CHROMA_DB_PATH,
) -> Dict[str, float]:
    """Generate responses and compute average ROUGE-L F1 (meaning overlap)."""
    preds, refs, fallback_count = _generate_predictions(
        data_path=data_path,
        limit=limit,
        use_filter=use_filter,
        temperature=temperature,
        max_context=max_context,
        model_dir=model_dir,
        persist_dir=persist_dir,
    )

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    f1_scores: List[float] = []
    precision_scores: List[float] = []
    recall_scores: List[float] = []

    for ref, pred in zip(refs, preds):
        scores = scorer.score(ref, pred)["rougeL"]
        f1_scores.append(scores.fmeasure)
        precision_scores.append(scores.precision)
        recall_scores.append(scores.recall)

    avg_f1 = float(sum(f1_scores) / len(f1_scores))
    avg_p = float(sum(precision_scores) / len(precision_scores))
    avg_r = float(sum(recall_scores) / len(recall_scores))

    return {
        "metric": "rougeL",
        "rougeL_f1": avg_f1,
        "rougeL_precision": avg_p,
        "rougeL_recall": avg_r,
        "samples": len(preds),
        "fallback_responses": fallback_count,
    }


def evaluate_bertscore(
    data_path: str,
    limit: int | None = None,
    use_filter: bool = False,
    temperature: float = settings.GENERATION_TEMPERATURE,
    max_context: int = settings.MAX_CONTEXT,
    model_dir: str = settings.EMP_DETECTOR_PATH,
    persist_dir: str = settings.CHROMA_DB_PATH,
    model_type: str = "bert-base-uncased",
) -> Dict[str, float]:
    """
    Generate responses and compute BERTScore (semantic similarity).
    Returns average precision/recall/F1 over the dataset.
    """
    preds, refs, fallback_count = _generate_predictions(
        data_path=data_path,
        limit=limit,
        use_filter=use_filter,
        temperature=temperature,
        max_context=max_context,
        model_dir=model_dir,
        persist_dir=persist_dir,
    )

    # bert_score returns tensors; convert to floats
    P, R, F1 = bert_score(
        cands=preds,
        refs=refs,
        model_type=model_type,
        verbose=False,
        device=None,  # auto-select; set to "cuda" manually if desired
    )
    avg_p = float(P.mean().item())
    avg_r = float(R.mean().item())
    avg_f1 = float(F1.mean().item())

    return {
        "metric": "bertscore",
        "bertscore_precision": avg_p,
        "bertscore_recall": avg_r,
        "bertscore_f1": avg_f1,
        "samples": len(preds),
        "fallback_responses": fallback_count,
        "model_type": model_type,
    }


def evaluate_emotion_alignment(
    data_path: str,
    limit: int | None = None,
    model_dir: str = settings.EMP_DETECTOR_PATH,
) -> Dict[str, float]:
    """
    Compute Emotion Alignment Score (accuracy + macro P/R/F1) between
    the model's predicted emotion on the user text and the reference_emotion label.
    """
    rows = _load_pairs_with_emotion(data_path, limit)
    if not rows:
        raise ValueError("No valid text/reference_emotion rows found for evaluation.")

    detector = EmotionDetector(model_dir=model_dir, base_model=settings.T5_MODEL_NAME)

    gold: List[str] = []
    pred: List[str] = []
    for text, ref_emotion in tqdm(rows, desc="Scoring emotions", unit="sample"):
        pred_emotion = detector.predict_emotion(text)
        gold.append(ref_emotion)
        pred.append(pred_emotion)

    acc = float(accuracy_score(gold, pred))
    precision, recall, f1, _ = precision_recall_fscore_support(
        gold, pred, average="macro", zero_division=0
    )
    return {
        "metric": "emotion_alignment",
        "accuracy": acc,
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "samples": len(gold),
    }


def evaluate_toxicity(
    data_path: str,
    limit: int | None = None,
    use_filter: bool = False,
    temperature: float = settings.GENERATION_TEMPERATURE,
    max_context: int = settings.MAX_CONTEXT,
    model_dir: str = settings.EMP_DETECTOR_PATH,
    persist_dir: str = settings.CHROMA_DB_PATH,
) -> Dict[str, float]:
    """
    Compute toxicity before and after the responsible filter, and report reduction.
    Uses Detoxify when available; falls back to keyword list otherwise.
    """
    pairs = _load_pairs(data_path, limit)
    if not pairs:
        raise ValueError("No valid text/response pairs found for evaluation.")

    embedder = TextEmbedder(model_name=settings.EMBEDDING_MODEL_NAME)
    vectordb = VectorDB(persist_directory=persist_dir)
    detector = EmotionDetector(model_dir=model_dir, base_model=settings.T5_MODEL_NAME)
    responder = EmpatheticResponder(api_key=settings.GEMINI_API_KEY, model_name=settings.GEMINI_MODEL_NAME)

    # scorer for toxicity; threshold 0 to always return scores
    tox_scorer = ToxicityFilter(threshold=0.0)
    filter_model = ToxicityFilter() if use_filter else None

    max_raw_scores: List[float] = []
    max_post_scores: List[float] = []
    toxic_raw_count = 0
    toxic_post_count = 0
    fallback_count = 0

    for user_msg, _ in tqdm(pairs, desc="Scoring toxicity", unit="sample"):
        query_emb = embedder.embed_queries([user_msg])[0]
        results = vectordb.query(embeddings=[query_emb], n_results=max_context)
        contexts: List[str] = results.get("documents", [[]])[0] if results else []

        emotion = detector.predict_emotion(user_msg)
        raw_resp = responder.generate_response(
            user_message=user_msg,
            detected_emotion=emotion,
            contexts=contexts,
            temperature=temperature,
        )
        if responder.used_fallback:
            fallback_count += 1

        toxic_raw, scores_raw = tox_scorer.is_toxic(raw_resp)
        max_raw = max(scores_raw.values()) if scores_raw else 0.0

        post_resp = raw_resp
        if filter_model is not None:
            toxic_flag, scores_flag = filter_model.is_toxic(raw_resp)
            if toxic_flag:
                post_resp = filter_model.sanitize_response(raw_resp, scores_flag)

        toxic_post, scores_post = tox_scorer.is_toxic(post_resp)
        max_post = max(scores_post.values()) if scores_post else 0.0

        max_raw_scores.append(float(max_raw))
        max_post_scores.append(float(max_post))
        toxic_raw_count += int(toxic_raw)
        toxic_post_count += int(toxic_post)

    avg_raw = float(sum(max_raw_scores) / len(max_raw_scores))
    avg_post = float(sum(max_post_scores) / len(max_post_scores))
    reduction = avg_raw - avg_post
    pct_reduction = (reduction / avg_raw * 100.0) if avg_raw > 0 else 0.0

    return {
        "metric": "toxicity",
        "avg_toxicity_raw": avg_raw,
        "avg_toxicity_post": avg_post,
        "absolute_reduction": reduction,
        "percent_reduction": pct_reduction,
        "samples": len(max_raw_scores),
        "toxic_raw_count": toxic_raw_count,
        "toxic_post_count": toxic_post_count,
        "fallback_responses": fallback_count,
        "filter_applied": bool(use_filter),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute BLEU or ROUGE-L for the empathetic responder.")
    parser.add_argument("--data", type=str, default="../datasets/processed/responses.csv", help="CSV with columns text,response")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on number of rows to score")
    parser.add_argument("--use_filter", action="store_true", help="Apply toxicity filter to generations")
    parser.add_argument("--temperature", type=float, default=settings.GENERATION_TEMPERATURE, help="Generation temperature")
    parser.add_argument("--max_context", type=int, default=settings.MAX_CONTEXT, help="Number of retrieved contexts")
    parser.add_argument("--model_dir", type=str, default=settings.EMP_DETECTOR_PATH, help="Emotion detector model path")
    parser.add_argument("--persist_dir", type=str, default=settings.CHROMA_DB_PATH, help="ChromaDB directory")
    parser.add_argument(
        "--metric",
        type=str,
        choices=["bleu", "rougeL", "bertscore", "emotion_alignment", "toxicity"],
        default="bleu",
        help="Which metric to compute",
    )
    parser.add_argument("--bertscore_model", type=str, default="bert-base-uncased", help="HF model for BERTScore (e.g., microsoft/deberta-large-mnli)")
    args = parser.parse_args()

    if args.metric == "bleu":
        metrics = evaluate_bleu(
            data_path=args.data,
            limit=args.limit,
            use_filter=bool(args.use_filter),
            temperature=float(args.temperature),
            max_context=int(args.max_context),
            model_dir=args.model_dir,
            persist_dir=args.persist_dir,
        )
    elif args.metric == "bertscore":
        metrics = evaluate_bertscore(
            data_path=args.data,
            limit=args.limit,
            use_filter=bool(args.use_filter),
            temperature=float(args.temperature),
            max_context=int(args.max_context),
            model_dir=args.model_dir,
            persist_dir=args.persist_dir,
            model_type=args.bertscore_model,
        )
    elif args.metric == "emotion_alignment":
        metrics = evaluate_emotion_alignment(
            data_path=args.data,
            limit=args.limit,
            model_dir=args.model_dir,
        )
    elif args.metric == "toxicity":
        metrics = evaluate_toxicity(
            data_path=args.data,
            limit=args.limit,
            use_filter=bool(args.use_filter),
            temperature=float(args.temperature),
            max_context=int(args.max_context),
            model_dir=args.model_dir,
            persist_dir=args.persist_dir,
        )
    else:
        metrics = evaluate_rouge_l(
            data_path=args.data,
            limit=args.limit,
            use_filter=bool(args.use_filter),
            temperature=float(args.temperature),
            max_context=int(args.max_context),
            model_dir=args.model_dir,
            persist_dir=args.persist_dir,
        )

    print("=== Evaluation ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()

