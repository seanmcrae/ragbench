.PHONY: install lint format typecheck test demo data scifact ci

install:
	uv sync --frozen

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
	uv run rag-eval compare bm25-k5 hybrid-rerank-agent3-k5

scifact:
	uv run python scripts/download_scifact.py
	uv run rag-eval run configs/scifact.yaml
	uv run rag-eval report runs/scifact

ci: lint typecheck test
