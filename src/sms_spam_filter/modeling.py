"""TF-IDF candidates, precision thresholding and diagnostics."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import FeatureUnion, Pipeline

KEYWORDS = ("free", "win", "winner", "claim", "urgent", "prize", "reward")


def candidate_models(seed: int = 42) -> dict[str, Pipeline]:
    def classifier() -> LogisticRegression:
        return LogisticRegression(class_weight="balanced", max_iter=2_000, random_state=seed)

    word = TfidfVectorizer(
        lowercase=True, strip_accents="unicode", ngram_range=(1, 2), sublinear_tf=True
    )
    char = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        analyzer="char_wb",
        ngram_range=(3, 5),
        sublinear_tf=True,
        max_features=25_000,
    )
    return {
        "word_tfidf": Pipeline([("tfidf", word), ("classifier", classifier())]),
        "char_tfidf": Pipeline([("tfidf", char), ("classifier", classifier())]),
        "word_char_tfidf": Pipeline(
            [
                (
                    "tfidf",
                    FeatureUnion(
                        [
                            (
                                "word",
                                TfidfVectorizer(
                                    lowercase=True,
                                    strip_accents="unicode",
                                    ngram_range=(1, 2),
                                    sublinear_tf=True,
                                ),
                            ),
                            (
                                "char",
                                TfidfVectorizer(
                                    lowercase=True,
                                    strip_accents="unicode",
                                    analyzer="char_wb",
                                    ngram_range=(3, 5),
                                    sublinear_tf=True,
                                    max_features=25_000,
                                ),
                            ),
                        ]
                    ),
                ),
                ("classifier", classifier()),
            ]
        ),
    }


def keyword_probability(messages: pd.Series) -> np.ndarray:
    return (
        messages.str.casefold()
        .map(lambda text: float(any(word in text for word in KEYWORDS)))
        .to_numpy()
    )


def choose_precision_threshold(
    y_true: pd.Series | np.ndarray, probability: np.ndarray, min_precision: float
) -> dict[str, float | bool]:
    if not 0 < min_precision <= 1:
        raise ValueError("min_precision must be in (0, 1]")
    candidates: list[dict[str, float | bool]] = []
    for threshold in sorted(set(float(value) for value in probability)):
        prediction = probability >= threshold
        if not prediction.any():
            continue
        candidates.append(
            {
                "threshold": threshold,
                "precision": float(precision_score(y_true, prediction, zero_division=0)),
                "recall": float(recall_score(y_true, prediction, zero_division=0)),
            }
        )
    feasible = [candidate for candidate in candidates if candidate["precision"] >= min_precision]
    pool = feasible or candidates
    selected = (
        max(
            pool,
            key=lambda item: (
                float(item["recall"]),
                float(item["precision"]),
                float(item["threshold"]),
            ),
        )
        if feasible
        else max(
            pool,
            key=lambda item: (
                float(item["precision"]),
                float(item["recall"]),
                float(item["threshold"]),
            ),
        )
    )
    selected["constraint_met"] = bool(feasible)
    return selected


def choose_model_for_precision(candidates: dict[str, dict[str, object]]) -> str:
    """Выбрать модель по достижимости precision-ограничения, затем recall и PR-AUC."""
    if not candidates:
        raise ValueError("At least one validation candidate is required")

    def operating_point(name: str) -> dict[str, float | bool]:
        point = candidates[name].get("precision_constraint")
        if not isinstance(point, dict):
            raise ValueError(f"Candidate {name} has no precision_constraint report")
        return point

    feasible = [name for name in candidates if bool(operating_point(name)["constraint_met"])]
    pool = feasible or list(candidates)
    if feasible:
        return max(
            pool,
            key=lambda name: (
                float(operating_point(name)["recall"]),
                float(candidates[name]["average_precision"]),
                float(operating_point(name)["precision"]),
            ),
        )
    return max(
        pool,
        key=lambda name: (
            float(operating_point(name)["precision"]),
            float(operating_point(name)["recall"]),
            float(candidates[name]["average_precision"]),
        ),
    )


def metrics(
    y_true: pd.Series | np.ndarray, probability: np.ndarray, threshold: float
) -> dict[str, object]:
    prediction = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    unique_targets = np.unique(np.asarray(y_true))
    average_precision = (
        float(average_precision_score(y_true, probability)) if len(unique_targets) == 2 else None
    )
    return {
        "average_precision": average_precision,
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "threshold": float(threshold),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def obfuscate_message(text: str) -> str:
    replacements = {"free": "fr33", "win": "w1n", "claim": "cl4im", "prize": "pr1ze"}
    result = text
    for source, replacement in replacements.items():
        result = re.sub(source, replacement, result, flags=re.IGNORECASE)
    return result


def robustness_report(
    model: Pipeline, spam_messages: pd.Series, threshold: float
) -> dict[str, float | int]:
    if spam_messages.empty:
        return {
            "spam_messages": 0,
            "original_detection_rate": 0.0,
            "obfuscated_detection_rate": 0.0,
        }
    original = model.predict_proba(spam_messages)[:, 1]
    mutated_text = spam_messages.map(obfuscate_message)
    mutated = model.predict_proba(mutated_text)[:, 1]
    return {
        "spam_messages": int(len(spam_messages)),
        "original_detection_rate": float(np.mean(original >= threshold)),
        "obfuscated_detection_rate": float(np.mean(mutated >= threshold)),
        "mean_probability_shift": float(np.mean(mutated - original)),
    }
