"""Batch and single-message SMS scoring CLI."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from .data import read_input
from .modeling import validate_model_bundle


def score_messages(messages: pd.Series, artifact_path: str | Path) -> pd.DataFrame:
    artifact = validate_model_bundle(joblib.load(artifact_path))
    probability = artifact["model"].predict_proba(messages)[:, 1]
    return pd.DataFrame(
        {
            "text": messages.to_numpy(),
            "spam_probability": probability,
            "is_spam": (probability >= artifact["threshold"]).astype(int),
            "threshold": artifact["threshold"],
        }
    )


def predict(input_path: str | Path, artifact_path: str | Path) -> pd.DataFrame:
    data = read_input(input_path, require_target=False)
    return score_messages(data["text"], artifact_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score SMS messages")
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--message", action="append", help="Message to score; may be repeated")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if bool(args.input) == bool(args.message):
        parser.error("provide exactly one of --input or --message")
    result = (
        predict(args.input, args.artifact)
        if args.input
        else score_messages(pd.Series(args.message, dtype="string"), args.artifact)
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.output, index=False)
        print(f"Wrote {len(result)} scores to {args.output}")
    else:
        print(result.to_csv(index=False))


if __name__ == "__main__":
    main()
