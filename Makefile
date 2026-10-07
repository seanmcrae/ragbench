.PHONY: install lint format typecheck test demo data ci

install:
	uv sync

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .
	uv run ruff check --fix .

typecheck:
	uv run mypy

test:
	uv run pytest

data:
	uv run python scripts/generate_synthetic.py

demo:
	uv run rag-eval run configs/example.yaml
	uv run rag-eval report --img-dir docs/img
	uv run rag-eval compare bm25-k5 hybrid-rerank-k5

ci: lint typecheck test
