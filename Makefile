.PHONY: install lint format typecheck test demo data scifact site diagram ci

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

# Static docs site for GitHub Pages: a fresh run of the example matrix rendered into site/.
site:
	uv run rag-eval run configs/example.yaml --quiet
	uv run python scripts/build_site.py --run runs/example --out site

# Re-render the architecture SVG after editing docs/architecture.mmd (needs Node 20+ and a
# Chromium that Puppeteer can launch; the rendered file is committed, so CI does not need this).
diagram:
	npx -y @mermaid-js/mermaid-cli@12 -c docs/mermaid.json -i docs/architecture.mmd \
		-o docs/img/architecture.svg -b white

ci: lint typecheck test
