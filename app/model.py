"""
Model loading and prediction logic for Clinical BERT assertion classification.
The model is loaded once at startup and cached for all subsequent requests.
"""

import logging
import time
from typing import Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

logger = logging.getLogger(__name__)

MODEL_NAME = "bvanaken/clinical-assertion-negation-bert"

# Module-level singletons — loaded once at startup
_pipeline: Optional[object] = None
_load_time_ms: Optional[float] = None


def load_model() -> None:
    """
    Load the HuggingFace model and tokenizer, storing them as module-level singletons.
    Should be called exactly once during application startup (via FastAPI lifespan).
    """
    global _pipeline, _load_time_ms

    if _pipeline is not None:
        logger.info("Model already loaded — skipping reload.")
        return

    logger.info(f"Loading model: {MODEL_NAME} …")
    start = time.perf_counter()

    device = 0 if torch.cuda.is_available() else -1
    device_name = "GPU (CUDA)" if device == 0 else "CPU"
    logger.info(f"Running inference on: {device_name}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

    _pipeline = pipeline(
        task="text-classification",
        model=model,
        tokenizer=tokenizer,
        device=device,
        top_k=None,          # return scores for ALL labels
        truncation=True,
        max_length=512,
    )

    elapsed = (time.perf_counter() - start) * 1000
    _load_time_ms = elapsed
    logger.info(f"Model loaded in {elapsed:.1f} ms")


def get_pipeline():
    """Return the cached inference pipeline (raises if not loaded)."""
    if _pipeline is None:
        raise RuntimeError(
            "Model pipeline is not initialised. "
            "Ensure load_model() was called during application startup."
        )
    return _pipeline


def predict_single(sentence: str) -> dict:
    """
    Run inference on a single sentence.

    Returns:
        {"label": str, "score": float}
        where label is one of PRESENT | ABSENT | CONDITIONAL
        and score is the confidence for that label (0-1).
    """
    pipe = get_pipeline()

    start = time.perf_counter()
    raw: list[list[dict]] = pipe(sentence)
    latency_ms = (time.perf_counter() - start) * 1000

    # pipeline with top_k=None returns [[{label, score}, …]]
    all_scores: list[dict] = raw[0]
    best = max(all_scores, key=lambda x: x["score"])

    logger.debug(
        f"Prediction: label={best['label']!r} score={best['score']:.4f} "
        f"latency={latency_ms:.1f}ms"
    )

    return {"label": best["label"], "score": round(best["score"], 4)}


def predict_batch(sentences: list[str]) -> list[dict]:
    """
    Run inference on a list of sentences efficiently using the pipeline's
    built-in batching.

    Returns:
        List of {"label": str, "score": float, "sentence": str}
    """
    pipe = get_pipeline()

    start = time.perf_counter()
    raw: list[list[dict]] = pipe(sentences, batch_size=16)
    latency_ms = (time.perf_counter() - start) * 1000

    results = []
    for sentence, all_scores in zip(sentences, raw):
        best = max(all_scores, key=lambda x: x["score"])
        results.append(
            {
                "sentence": sentence,
                "label": best["label"],
                "score": round(best["score"], 4),
            }
        )

    logger.debug(
        f"Batch prediction: {len(sentences)} sentences in {latency_ms:.1f}ms"
    )

    return results


def get_model_info() -> dict:
    """Return metadata about the loaded model."""
    return {
        "model_name": MODEL_NAME,
        "loaded": _pipeline is not None,
        "load_time_ms": round(_load_time_ms, 1) if _load_time_ms else None,
        "device": (
            "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
        ),
        "labels": ["PRESENT", "ABSENT", "CONDITIONAL"],
    }