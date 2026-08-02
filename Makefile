# VIS — common developer tasks. Run `make help` for the list.
PY ?= python

.PHONY: help install test lint harness experiments smoke check all

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:    ## Install the package + dev tools (editable)
	$(PY) -m pip install -e ".[dev]"

test:       ## Run the unit + integration tests
	pytest -q

lint:       ## Lint with ruff
	ruff check .

harness:    ## Smoke run: detectors vs. synthetic/simulated traffic
	$(PY) eval/harness.py

experiments: ## Run the E1-E7 + Section 8 evaluation suite
	$(PY) eval/experiments.py

smoke: harness experiments  ## Run both runnable demos

check: lint test            ## What CI runs: lint + tests

all: lint test smoke        ## Everything: lint, tests, and both demos
