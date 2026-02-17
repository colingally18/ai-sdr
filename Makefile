.PHONY: install run test lint clean

install:
	pip install -e ".[dev]"

run:
	python -m sdr.main

test:
	pytest -v

lint:
	ruff check sdr/ tests/
	ruff format --check sdr/ tests/

format:
	ruff format sdr/ tests/

clean:
	rm -rf __pycache__ .pytest_cache *.egg-info dist build
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
