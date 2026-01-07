from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from sms_spam_filter.data import make_smoke_data, read_input, validate_schema
from sms_spam_filter.evaluate import evaluate
from sms_spam_filter.modeling import choose_model_for_precision
from sms_spam_filter.predict import predict, score_messages
from sms_spam_filter.train import train


class SMSPipelineTest(unittest.TestCase):
    def test_model_selection_prefers_feasible_precision_constraint(self) -> None:
        candidates = {
            "high_ap_but_infeasible": {
                "average_precision": 0.99,
                "precision_constraint": {
                    "constraint_met": False,
                    "precision": 0.80,
                    "recall": 0.95,
                },
            },
            "feasible": {
                "average_precision": 0.90,
                "precision_constraint": {
                    "constraint_met": True,
                    "precision": 0.96,
                    "recall": 0.70,
                },
            },
        }

        self.assertEqual(choose_model_for_precision(candidates), "feasible")

    def test_deduplication_precedes_split(self) -> None:
        raw = make_smoke_data(40, 20)
        clean = validate_schema(raw)
        self.assertEqual(len(clean), len(raw) - 2)

    def test_conflicting_duplicate_is_rejected(self) -> None:
        frame = pd.DataFrame(
            {"label": ["ham", "spam"], "text": ["Call me later", "  call ME later  "]}
        )
        with self.assertRaisesRegex(ValueError, "conflicting"):
            validate_schema(frame)

    def test_official_tab_format_with_commas_is_supported(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "SMSSpamCollection"
            path.write_text(
                "ham\tHello, are you coming later?\nspam\tFREE, claim prize now!\n",
                encoding="utf-8",
            )
            loaded = read_input(path)
            self.assertEqual(loaded["is_spam"].tolist(), [0, 1])

    def test_train_evaluate_and_predict(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "sms.csv"
            prediction_path = root / "messages.csv"
            output = root / "artifacts"
            make_smoke_data(90, 36).to_csv(input_path, index=False)
            pd.DataFrame({"text": ["Lunch at noon", "FREE prize call now"]}).to_csv(
                prediction_path, index=False
            )

            result = train(input_path, output, min_precision=0.9)
            evaluation = evaluate(input_path, output / "model.joblib")
            scored = predict(prediction_path, output / "model.joblib")
            direct = score_messages(pd.Series(["hello"]), output / "model.joblib")

            self.assertTrue((output / "test_predictions.csv").exists())
            self.assertIn(result["selected_model"], {"word_tfidf", "char_tfidf", "word_char_tfidf"})
            self.assertIn("constraint_met", result["precision_constraint"])
            self.assertIn("obfuscation", evaluation)
            self.assertEqual(len(scored), 2)
            self.assertEqual(len(direct), 1)


if __name__ == "__main__":
    unittest.main()
