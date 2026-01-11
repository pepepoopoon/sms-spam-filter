from __future__ import annotations

import json

import numpy as np
import pandas as pd

from sms_spam_filter.experiments import (
    ExperimentConfig,
    main,
    precision_threshold_diagnostics,
    run_experiment,
)


def test_experiment_is_reproducible() -> None:
    config = ExperimentConfig(
        label="deterministic",
        seed=17,
        ham_count=60,
        spam_count=20,
        min_precision=0.9,
    )

    first = run_experiment(config)
    second = run_experiment(config)

    assert first == second
    assert first["split"] == {"train": 48, "validation": 16, "test": 16}
    assert first["test_metrics"]["average_precision"] is not None


def test_experiment_compares_model_with_keyword_rule() -> None:
    result = run_experiment(
        ExperimentConfig(
            label="baseline",
            seed=19,
            ham_count=80,
            spam_count=30,
            min_precision=0.9,
        )
    )

    assert result["baseline_comparison"]["baseline"] == "keyword_rule"
    assert result["baseline_comparison"]["candidate"] == "word_tfidf"
    assert (
        result["test_metrics"]["average_precision"]
        >= result["keyword_baseline"]["average_precision"]
    )


def test_precision_threshold_curve_tracks_alert_volume() -> None:
    result = run_experiment(
        ExperimentConfig(
            label="thresholds",
            seed=29,
            ham_count=90,
            spam_count=35,
            min_precision=0.95,
        )
    )

    diagnostics = result["threshold_diagnostics"]
    assert [point["threshold"] for point in diagnostics["curve"]] == [
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
    ]
    assert diagnostics["curve"][0]["predicted_spam"] >= diagnostics["curve"][-1]["predicted_spam"]
    assert set(diagnostics["operating_points"]) == {"0.8", "0.9", "0.95", "0.99"}

    direct = precision_threshold_diagnostics(
        pd.Series([0, 0, 1, 1]), np.array([0.1, 0.4, 0.6, 0.9])
    )
    assert direct["curve"][4]["predicted_spam"] == 2


def test_experiment_cli_writes_json(tmp_path) -> None:
    output = tmp_path / "result.json"

    main(
        [
            "--output",
            str(output),
            "--label",
            "cli-check",
            "--seed",
            "23",
            "--ham-count",
            "70",
            "--spam-count",
            "25",
        ]
    )

    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["config"]["label"] == "cli-check"
    assert result["config"]["seed"] == 23
