"""Run deterministic offline experiments for the SMS spam pipeline."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .data import TARGET, make_smoke_data, validate_schema
from .modeling import candidate_models, choose_precision_threshold, metrics


@dataclass(frozen=True)
class ExperimentConfig:
    """Parameters that uniquely identify one synthetic experiment."""

    label: str
    seed: int
    ham_count: int
    spam_count: int
    min_precision: float


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


def run_experiment(config: ExperimentConfig) -> dict[str, object]:
    """Fit word TF-IDF and measure the selected precision operating point."""
    if min(config.ham_count, config.spam_count) < 20:
        raise ValueError("at least 20 messages per class are required")
    if not 0 < config.min_precision <= 1:
        raise ValueError("min_precision must be in (0, 1]")

    frame = validate_schema(make_smoke_data(config.ham_count, config.spam_count))
    train, validation, test = split_messages(frame, config.seed)
    model = candidate_models(config.seed)["word_tfidf"]
    model.fit(train["text"], train[TARGET])
    validation_probability = model.predict_proba(validation["text"])[:, 1]
    threshold = choose_precision_threshold(
        validation[TARGET], validation_probability, config.min_precision
    )
    test_probability = model.predict_proba(test["text"])[:, 1]

    return {
        "schema_version": 1,
        "config": asdict(config),
        "split": {"train": len(train), "validation": len(validation), "test": len(test)},
        "model": "word_tfidf",
        "precision_constraint": threshold,
        "validation_metrics": metrics(
            validation[TARGET], validation_probability, float(threshold["threshold"])
        ),
        "test_metrics": metrics(test[TARGET], test_probability, float(threshold["threshold"])),
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
    args = parser.parse_args(argv)
    config = ExperimentConfig(
        label=args.label,
        seed=args.seed,
        ham_count=args.ham_count,
        spam_count=args.spam_count,
        min_precision=args.min_precision,
    )
    write_result(run_experiment(config), args.output)
    print(f"experiment {config.label} written to {args.output}")


if __name__ == "__main__":
    main()
