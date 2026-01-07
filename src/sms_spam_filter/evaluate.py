"""Evaluate a persisted SMS spam filter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib

from .data import TARGET, read_input
from .modeling import metrics, robustness_report, validate_model_bundle


def evaluate(input_path: str | Path, artifact_path: str | Path) -> dict[str, object]:
    data = read_input(input_path)
    artifact = validate_model_bundle(joblib.load(artifact_path))
    probability = artifact["model"].predict_proba(data["text"])[:, 1]
    result = metrics(data[TARGET], probability, artifact["threshold"])
    result["obfuscation"] = robustness_report(
        artifact["model"], data.loc[data[TARGET].eq(1), "text"], artifact["threshold"]
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a persisted SMS spam filter")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.input, args.artifact), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
