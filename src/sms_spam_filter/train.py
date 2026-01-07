"""Train and select a precision-oriented SMS spam filter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
from sklearn.model_selection import train_test_split

from .data import TARGET, read_input
from .modeling import (
    candidate_models,
    choose_model_for_precision,
    choose_precision_threshold,
    keyword_probability,
    metrics,
    robustness_report,
)


def train(
    input_path: str | Path,
    output_dir: str | Path,
    min_precision: float = 0.95,
    seed: int = 42,
) -> dict[str, object]:
    data = read_input(input_path)
    train_validation, test = train_test_split(
        data, test_size=0.2, stratify=data[TARGET], random_state=seed
    )
    train_frame, validation = train_test_split(
        train_validation,
        test_size=0.25,
        stratify=train_validation[TARGET],
        random_state=seed,
    )
    models = candidate_models(seed)
    validation_candidates: dict[str, dict[str, object]] = {}
    for name, model in models.items():
        model.fit(train_frame["text"], train_frame[TARGET])
        probability = model.predict_proba(validation["text"])[:, 1]
        report = metrics(validation[TARGET], probability, threshold=0.5)
        report["precision_constraint"] = choose_precision_threshold(
            validation[TARGET], probability, min_precision
        )
        validation_candidates[name] = report
    selected_name = choose_model_for_precision(validation_candidates)
    model = models[selected_name]
    threshold_report = validation_candidates[selected_name]["precision_constraint"]
    threshold = float(threshold_report["threshold"])
    test_probability = model.predict_proba(test["text"])[:, 1]
    test_metrics = metrics(test[TARGET], test_probability, threshold)
    short = test["text"].str.len().lt(40)
    short_metrics = (
        metrics(test.loc[short, TARGET], test_probability[short.to_numpy()], threshold)
        if short.any()
        else None
    )
    result: dict[str, object] = {
        "seed": seed,
        "deduplicated_messages": len(data),
        "split": {"train": len(train_frame), "validation": len(validation), "test": len(test)},
        "validation_candidates": validation_candidates,
        "selected_model": selected_name,
        "precision_constraint": {"requested": min_precision, **threshold_report},
        "test": test_metrics,
        "short_messages_test": short_metrics,
        "keyword_baseline_test": metrics(test[TARGET], keyword_probability(test["text"]), 0.5),
        "obfuscation_test": robustness_report(
            model, test.loc[test[TARGET].eq(1), "text"], threshold
        ),
        "data_note": "Metrics describe the supplied file; smoke data are not real evaluation.",
    }
    artifact = {
        "model": model,
        "threshold": threshold,
        "selected_model": selected_name,
        "min_precision": min_precision,
        "label_semantics": {"0": "ham", "1": "spam"},
        "seed": seed,
    }
    predictions = test[["text", TARGET]].copy()
    predictions["probability"] = test_probability
    predictions["prediction"] = (test_probability >= threshold).astype(int)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output / "model.joblib")
    predictions.to_csv(output / "test_predictions.csv", index=False)
    (output / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a precision-oriented SMS spam filter")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--min-precision", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(
        json.dumps(
            train(args.input, args.output_dir, args.min_precision, args.seed),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
