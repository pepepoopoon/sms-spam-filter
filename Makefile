PYTHON ?= python3.11

.PHONY: install lint test smoke train evaluate predict

install:
	$(PYTHON) -m pip install -e ".[dev]"

lint:
	$(PYTHON) -m ruff check src tests

test:
	$(PYTHON) -m pytest

smoke:
	PYTHONPATH=src $(PYTHON) -m sms_spam_filter.data --output data/smoke.csv

train: smoke
	PYTHONPATH=src $(PYTHON) -m sms_spam_filter.train --input data/smoke.csv --output-dir artifacts --min-precision 0.95

evaluate:
	PYTHONPATH=src $(PYTHON) -m sms_spam_filter.evaluate --input data/smoke.csv --artifact artifacts/model.joblib

predict:
	PYTHONPATH=src $(PYTHON) -m sms_spam_filter.predict --artifact artifacts/model.joblib --message "FREE prize, call now!"
