.PHONY: setup lint format typecheck test fixture quality check all
PY ?= python

setup:
	$(PY) -m pip install -e ".[dev]"
	pre-commit install

lint:
	ruff check src tests
	black --check src tests

format:
	black src tests
	ruff check --fix src tests

typecheck:
	mypy

test:
	pytest --cov

fixture:
	sentinel data make-fixture --out data/fixtures

quality: fixture
	sentinel data quality --fixture data/fixtures --out results
	sentinel signals research --fixture data/fixtures --out results
	sentinel backtest run --fixture data/fixtures --out results
	sentinel signals model-cards
	sentinel docs-check

check: lint typecheck test

all: check quality
