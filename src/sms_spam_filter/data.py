"""SMS data parsing, deduplication and deterministic smoke data."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

TARGET = "is_spam"


def fingerprint(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold()).strip()


def validate_schema(frame: pd.DataFrame, require_target: bool = True) -> pd.DataFrame:
    clean = frame.copy()
    if "message" in clean and "text" not in clean:
        clean = clean.rename(columns={"message": "text"})
    required = {"text", "label"} if require_target else {"text"}
    missing = sorted(required.difference(clean.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if clean.empty:
        raise ValueError("SMS data is empty")
    clean["text"] = clean["text"].astype("string").str.strip()
    if clean["text"].isna().any() or clean["text"].eq("").any():
        raise ValueError("text must be non-empty")
    clean["fingerprint"] = clean["text"].map(fingerprint)
    if require_target:
        labels = clean["label"].astype("string").str.casefold().str.strip()
        mapped = labels.map({"ham": 0, "spam": 1, "0": 0, "1": 1})
        if mapped.isna().any():
            raise ValueError("label must contain ham/spam or 0/1")
        clean[TARGET] = mapped.astype(int)
        conflict = clean.groupby("fingerprint", observed=True)[TARGET].nunique().gt(1)
        if conflict.any():
            raise ValueError("Normalized duplicate messages have conflicting labels")
        clean = clean.drop_duplicates(subset="fingerprint", keep="first")
        if clean[TARGET].nunique() != 2:
            raise ValueError("label must contain both classes")
    return clean.reset_index(drop=True)


def read_input(path: str | Path, require_target: bool = True) -> pd.DataFrame:
    path = Path(path)
    try:
        csv_frame = pd.read_csv(path, keep_default_na=False)
    except pd.errors.ParserError:
        csv_frame = pd.DataFrame()
    canonical = {"text", "label"} if require_target else {"text"}
    if canonical.issubset(csv_frame.columns) or "message" in csv_frame.columns:
        return validate_schema(csv_frame, require_target=require_target)
    if not require_target:
        raise ValueError("Prediction CSV must contain a text column")
    raw = pd.read_csv(
        path,
        sep="\t",
        names=["label", "text"],
        header=None,
        keep_default_na=False,
        engine="python",
    )
    return validate_schema(raw, require_target=True)


def make_smoke_data(ham_count: int = 120, spam_count: int = 48) -> pd.DataFrame:
    """Create deterministic English SMS-like examples and a few exact duplicates."""
    if min(ham_count, spam_count) < 20:
        raise ValueError("At least 20 messages per class are required")
    ham_templates = (
        "Meeting moved to {hour}:00, see you in room {room}",
        "Can you call me after work about ticket {index}?",
        "Your appointment is confirmed for day {day}",
        "Please bring milk and bread when you come home {index}",
        "Thanks for your help today, reference {index}",
        "Train arrives at platform {room} around {hour}:15",
    )
    spam_templates = (
        "FREE prize {index}! Call 0900{code} now to claim",
        "URGENT winner: text WIN {code} for cash reward {index}",
        "Exclusive offer {index}, click example.invalid/{code} today",
        "You have won voucher {index}; claim by calling 0800{code}",
    )
    rows: list[dict[str, str]] = []
    for index in range(ham_count):
        template = ham_templates[index % len(ham_templates)]
        rows.append(
            {
                "label": "ham",
                "text": template.format(
                    index=index, hour=8 + index % 10, room=1 + index % 20, day=1 + index % 28
                ),
            }
        )
    for index in range(spam_count):
        template = spam_templates[index % len(spam_templates)]
        rows.append({"label": "spam", "text": template.format(index=index, code=10_000 + index)})
    rows.extend([rows[3].copy(), rows[ham_count + 2].copy()])
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create deterministic SMS smoke data")
    parser.add_argument("--output", type=Path, default=Path("data/smoke.csv"))
    parser.add_argument("--ham-count", type=int, default=120)
    parser.add_argument("--spam-count", type=int, default=48)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    make_smoke_data(args.ham_count, args.spam_count).to_csv(args.output, index=False)
    print(f"Wrote synthetic SMS data to {args.output}")


if __name__ == "__main__":
    main()
