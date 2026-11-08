.PHONY: install test lint verify propositions tables notebook demo assets check clean

PY ?= python

install:            ## install the package with the development tools
	$(PY) -m pip install -e ".[dev]"

test:               ## run the test suite
	$(PY) -m pytest

lint:               ## static checks
	ruff check .

verify:             ## every number of the paper from results/v6.6
	$(PY) scripts/verify_paper_numbers.py

propositions:       ## re-run the numerical checks of the propositions
	$(PY) scripts/check_propositions.py

tables:             ## write the tables of the paper to build/paper_tables/
	$(PY) scripts/make_paper_tables.py

notebook:           ## rebuild the notebook from its cell sources and compare with the shipped one
	$(PY) notebook/build_notebook.py --check

demo:               ## the probe on three predictors with known answers
	$(PY) scripts/demo_probe.py

assets:             ## redraw the README figures and badges from results/v6.6 (needs .[assets])
	$(PY) scripts/make_readme_assets.py

check: lint test verify propositions notebook   ## everything the continuous integration runs

clean:
	rm -rf build notebook/build .pytest_cache .ruff_cache src/*.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
