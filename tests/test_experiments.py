from __future__ import annotations

import json

import numpy as np
import pandas as pd

from sms_spam_filter.experiments import (
    ExperimentConfig,
    main,
    measure_seed_stability,
    obfuscate_messages,
    precision_threshold_diagnostics,
    run_experiment,
    transform_message_length,
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


def test_imbalance_diagnostics_measure_split_prevalence() -> None:
    result = run_experiment(
        ExperimentConfig(
            label="imbalance",
            seed=31,
            ham_count=100,
            spam_count=20,
            min_precision=0.9,
        )
    )

    balance = result["imbalance_diagnostics"]
    assert balance["requested"] == {"ham": 100, "spam": 20}
    assert balance["partitions"]["full"]["ham"] == 84
    assert balance["partitions"]["full"]["spam"] == 20
    assert balance["deduplicated_messages"] == 16
    assert balance["partitions"]["full"]["ham_to_spam_ratio"] == 4.2
    assert balance["max_absolute_prevalence_shift"] < 0.03


def test_vectorizer_configuration_reports_sparse_features() -> None:
    result = run_experiment(
        ExperimentConfig(
            label="characters",
            seed=37,
            ham_count=80,
            spam_count=30,
            min_precision=0.9,
            vectorizer="char_tfidf",
            classifier_c=0.5,
            char_ngram_min=2,
            char_ngram_max=4,
            max_features=700,
        )
    )

    diagnostics = result["vectorizer_diagnostics"]
    assert result["model"] == "char_tfidf"
    assert result["baseline_comparison"]["candidate"] == "char_tfidf"
    assert 0 < diagnostics["features"] <= 700
    assert 0 < diagnostics["density"] < 1


def test_obfuscation_diagnostics_measure_probability_shift() -> None:
    result = run_experiment(
        ExperimentConfig(
            label="spaced-keywords",
            seed=41,
            ham_count=90,
            spam_count=36,
            min_precision=0.9,
            vectorizer="char_tfidf",
            obfuscation="spaced",
        )
    )

    diagnostics = result["obfuscation_diagnostics"]
    changed = obfuscate_messages(pd.Series(["FREE prize now"]), "spaced")

    assert changed.iloc[0] == "f r e e p r i z e now"
    assert diagnostics["scenario"] == "spaced"
    assert diagnostics["changed_messages"] > 0
    assert -1 <= diagnostics["mean_probability_shift"] <= 1


def test_length_diagnostics_cover_test_and_measure_truncation() -> None:
    result = run_experiment(
        ExperimentConfig(
            label="short-messages",
            seed=43,
            ham_count=100,
            spam_count=40,
            min_precision=0.9,
            length_stress="truncate_24",
        )
    )

    diagnostics = result["length_diagnostics"]
    transformed = transform_message_length(pd.Series(["a" * 50, "short"]), "truncate_24")

    assert transformed.str.len().max() == 24
    assert (
        sum(bucket["messages"] for bucket in diagnostics["buckets"].values())
        == result["split"]["test"]
    )
    assert diagnostics["stress"]["scenario"] == "truncate_24"
    assert diagnostics["stress"]["mean_length_after"] <= 24


def test_seed_stability_repeats_threshold_selection() -> None:
    config = ExperimentConfig(
        label="stable",
        seed=47,
        ham_count=100,
        spam_count=40,
        min_precision=0.9,
        vectorizer="word_char_tfidf",
        max_features=1_000,
        repeat_seeds=(101, 103, 107),
    )

    stability = measure_seed_stability(config)
    result = run_experiment(config)

    assert [run["seed"] for run in stability["runs"]] == [101, 103, 107]
    assert stability["summary"]["run_count"] == 3
    assert 0 <= stability["summary"]["recall_mean"] <= 1
    assert result["seed_stability"] == stability


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
