"""Run deterministic offline experiments for the SMS spam pipeline."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .data import TARGET, make_smoke_data, validate_schema
from .modeling import (
    KEYWORDS,
    candidate_models,
    choose_precision_threshold,
    keyword_probability,
    metrics,
    obfuscate_message,
)


@dataclass(frozen=True)
class ExperimentConfig:
    """Parameters that uniquely identify one synthetic experiment."""

    label: str
    seed: int
    ham_count: int
    spam_count: int
    min_precision: float
    vectorizer: str = "word_tfidf"
    classifier_c: float = 1.0
    word_ngram_max: int = 2
    char_ngram_min: int = 3
    char_ngram_max: int = 5
    max_features: int = 25_000
    obfuscation: str = "none"


def build_model(config: ExperimentConfig, seed: int):
    """Build a recorded TF-IDF configuration with a balanced logistic classifier."""
    models = candidate_models(seed)
    if config.vectorizer not in models:
        raise ValueError(f"unsupported vectorizer: {config.vectorizer}")
    model = models[config.vectorizer]
    parameters: dict[str, object] = {"classifier__C": config.classifier_c}
    if config.vectorizer == "word_tfidf":
        parameters["tfidf__ngram_range"] = (1, config.word_ngram_max)
    elif config.vectorizer == "char_tfidf":
        parameters["tfidf__ngram_range"] = (config.char_ngram_min, config.char_ngram_max)
        parameters["tfidf__max_features"] = config.max_features
    else:
        parameters["tfidf__word__ngram_range"] = (1, config.word_ngram_max)
        parameters["tfidf__char__ngram_range"] = (
            config.char_ngram_min,
            config.char_ngram_max,
        )
        parameters["tfidf__char__max_features"] = config.max_features
    return model.set_params(**parameters)


def vectorizer_diagnostics(model, messages: pd.Series) -> dict[str, float | int]:
    """Report fitted vocabulary dimensionality and sparse-matrix density."""
    matrix = model.named_steps["tfidf"].transform(messages)
    total_cells = matrix.shape[0] * matrix.shape[1]
    return {
        "documents": matrix.shape[0],
        "features": matrix.shape[1],
        "nonzero_values": int(matrix.nnz),
        "density": float(matrix.nnz / total_cells) if total_cells else 0.0,
    }


def obfuscate_messages(messages: pd.Series, kind: str) -> pd.Series:
    """Apply one deterministic spam-keyword transformation."""
    if kind == "none":
        return messages.copy()
    if kind == "leetspeak":
        return messages.map(obfuscate_message)

    def replace(text: str) -> str:
        result = text
        for keyword in KEYWORDS:
            if kind == "spaced":
                replacement = " ".join(keyword)
            elif kind == "punctuated":
                replacement = ".".join(keyword)
            elif kind == "mixed_case":
                replacement = "".join(
                    character.upper() if index % 2 else character.lower()
                    for index, character in enumerate(keyword)
                )
            else:
                raise ValueError(f"unsupported obfuscation: {kind}")
            result = re.sub(keyword, replacement, result, flags=re.IGNORECASE)
        return result

    return messages.map(replace)


def obfuscation_diagnostics(
    model, spam_messages: pd.Series, threshold: float, kind: str
) -> dict[str, float | int | str]:
    """Measure probability and detection changes after keyword obfuscation."""
    if spam_messages.empty:
        return {"scenario": kind, "messages": 0}
    mutated = obfuscate_messages(spam_messages, kind)
    original_probability = model.predict_proba(spam_messages)[:, 1]
    mutated_probability = model.predict_proba(mutated)[:, 1]
    return {
        "scenario": kind,
        "messages": len(spam_messages),
        "changed_messages": int((mutated != spam_messages).sum()),
        "original_detection_rate": float((original_probability >= threshold).mean()),
        "obfuscated_detection_rate": float((mutated_probability >= threshold).mean()),
        "mean_probability_shift": float((mutated_probability - original_probability).mean()),
    }


def split_messages(
    frame: pd.DataFrame, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create deterministic stratified 60/20/20 partitions."""
    train_validation, test = train_test_split(
        frame, test_size=0.2, stratify=frame[TARGET], random_state=seed
    )
    train, validation = train_test_split(
        train_validation,
        test_size=0.25,
        stratify=train_validation[TARGET],
        random_state=seed,
    )
    return (
        train.reset_index(drop=True),
        validation.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def precision_threshold_diagnostics(
    truth: pd.Series, probability, targets: tuple[float, ...] = (0.8, 0.9, 0.95, 0.99)
) -> dict[str, object]:
    """Measure a fixed threshold curve and feasible points for precision targets."""
    curve = []
    for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        report = metrics(truth, probability, threshold)
        matrix = report["confusion_matrix"]
        curve.append(
            {
                "threshold": threshold,
                "precision": report["precision"],
                "recall": report["recall"],
                "f1": report["f1"],
                "predicted_spam": matrix["tp"] + matrix["fp"],
            }
        )
    return {
        "curve": curve,
        "operating_points": {
            str(target): choose_precision_threshold(truth, probability, target)
            for target in targets
        },
    }


def _class_balance(frame: pd.DataFrame) -> dict[str, float | int]:
    spam = int(frame[TARGET].sum())
    ham = int(len(frame) - spam)
    return {
        "messages": len(frame),
        "ham": ham,
        "spam": spam,
        "spam_prevalence": spam / len(frame),
        "ham_to_spam_ratio": ham / spam,
    }


def imbalance_diagnostics(
    full: pd.DataFrame,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> dict[str, object]:
    """Report class prevalence and its drift across stratified partitions."""
    partitions = {
        "full": _class_balance(full),
        "train": _class_balance(train),
        "validation": _class_balance(validation),
        "test": _class_balance(test),
    }
    full_prevalence = float(partitions["full"]["spam_prevalence"])
    return {
        "partitions": partitions,
        "max_absolute_prevalence_shift": max(
            abs(float(report["spam_prevalence"]) - full_prevalence)
            for name, report in partitions.items()
            if name != "full"
        ),
    }


def run_experiment(config: ExperimentConfig) -> dict[str, object]:
    """Fit word TF-IDF and measure the selected precision operating point."""
    if min(config.ham_count, config.spam_count) < 20:
        raise ValueError("at least 20 messages per class are required")
    if not 0 < config.min_precision <= 1:
        raise ValueError("min_precision must be in (0, 1]")

    frame = validate_schema(make_smoke_data(config.ham_count, config.spam_count))
    train, validation, test = split_messages(frame, config.seed)
    balance = imbalance_diagnostics(frame, train, validation, test)
    balance["requested"] = {"ham": config.ham_count, "spam": config.spam_count}
    balance["deduplicated_messages"] = config.ham_count + config.spam_count - len(frame)
    model = build_model(config, config.seed)
    model.fit(train["text"], train[TARGET])
    validation_probability = model.predict_proba(validation["text"])[:, 1]
    threshold = choose_precision_threshold(
        validation[TARGET], validation_probability, config.min_precision
    )
    threshold_diagnostics = precision_threshold_diagnostics(
        validation[TARGET], validation_probability
    )
    test_probability = model.predict_proba(test["text"])[:, 1]
    test_metrics = metrics(test[TARGET], test_probability, float(threshold["threshold"]))
    obfuscation = obfuscation_diagnostics(
        model,
        test.loc[test[TARGET].eq(1), "text"],
        float(threshold["threshold"]),
        config.obfuscation,
    )
    baseline_metrics = metrics(test[TARGET], keyword_probability(test["text"]), 0.5)
    model_ap = float(test_metrics["average_precision"])
    baseline_ap = float(baseline_metrics["average_precision"])

    return {
        "schema_version": 1,
        "config": asdict(config),
        "split": {"train": len(train), "validation": len(validation), "test": len(test)},
        "model": config.vectorizer,
        "vectorizer_diagnostics": vectorizer_diagnostics(model, train["text"]),
        "obfuscation_diagnostics": obfuscation,
        "precision_constraint": threshold,
        "threshold_diagnostics": threshold_diagnostics,
        "imbalance_diagnostics": balance,
        "validation_metrics": metrics(
            validation[TARGET], validation_probability, float(threshold["threshold"])
        ),
        "test_metrics": test_metrics,
        "keyword_baseline": baseline_metrics,
        "baseline_comparison": {
            "candidate": config.vectorizer,
            "baseline": "keyword_rule",
            "average_precision_lift": model_ap - baseline_ap,
            "recall_lift": float(test_metrics["recall"]) - float(baseline_metrics["recall"]),
        },
    }


def write_result(result: dict[str, object], output: Path) -> None:
    """Write a stable UTF-8 JSON artifact."""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ham-count", type=int, default=120)
    parser.add_argument("--spam-count", type=int, default=48)
    parser.add_argument("--min-precision", type=float, default=0.95)
    parser.add_argument(
        "--vectorizer",
        choices=["word_tfidf", "char_tfidf", "word_char_tfidf"],
        default="word_tfidf",
    )
    parser.add_argument("--classifier-c", type=float, default=1.0)
    parser.add_argument("--word-ngram-max", type=int, default=2)
    parser.add_argument("--char-ngram-min", type=int, default=3)
    parser.add_argument("--char-ngram-max", type=int, default=5)
    parser.add_argument("--max-features", type=int, default=25_000)
    parser.add_argument(
        "--obfuscation",
        choices=["none", "leetspeak", "spaced", "punctuated", "mixed_case"],
        default="none",
    )
    args = parser.parse_args(argv)
    config = ExperimentConfig(
        label=args.label,
        seed=args.seed,
        ham_count=args.ham_count,
        spam_count=args.spam_count,
        min_precision=args.min_precision,
        vectorizer=args.vectorizer,
        classifier_c=args.classifier_c,
        word_ngram_max=args.word_ngram_max,
        char_ngram_min=args.char_ngram_min,
        char_ngram_max=args.char_ngram_max,
        max_features=args.max_features,
        obfuscation=args.obfuscation,
    )
    write_result(run_experiment(config), args.output)
    print(f"experiment {config.label} written to {args.output}")


if __name__ == "__main__":
    main()
