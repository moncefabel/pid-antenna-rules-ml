export PYTHONPATH := .

all: structures measurements rules layout

structures:
	python scripts/01_generate_structures.py

measurements:
	python scripts/02_simulate_measurements.py

rules:
	python scripts/03_extract_rules.py

layout:
	python scripts/04_plot_layout.py

test:
	python -m pytest -q tests

.PHONY: all structures measurements rules layout test
