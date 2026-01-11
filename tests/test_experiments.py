from __future__ import annotations

import json

from sms_spam_filter.experiments import ExperimentConfig, main, run_experiment


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
