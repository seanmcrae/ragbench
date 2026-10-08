"""The static docs site builds offline from a run and the repo's own docs."""

import importlib.util
import re
import sys
from pathlib import Path
from types import ModuleType

import pytest
import yaml
from typer.testing import CliRunner

from rag_eval.cli import app

ROOT = Path(__file__).resolve().parent.parent


def _load_builder() -> ModuleType:
    spec = importlib.util.spec_from_file_location("build_site", ROOT / "scripts" / "build_site.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(module)
    return module


build_site = _load_builder()


@pytest.fixture(scope="module")
def site(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("site")
    config = {
        "name": "site-smoke",
        "dataset": {"kind": "synthetic", "limit": 12},
        "prices": str(ROOT / "configs" / "prices.yaml"),
        "output_dir": str(root / "runs"),
        "cache_dir": str(root / "cache"),
        "baseline": "bm25-k3",
        "budget": {"max_cost_per_1k_usd": 1.0, "max_p95_latency_ms": 1500},
        "bootstrap_samples": 200,
        "defaults": {"top_k": 3},
        "pipelines": [
            {"name": "bm25-k3", "retriever": {"kind": "bm25"}},
            {"name": "agent-k3", "retriever": {"kind": "bm25"}, "max_steps": 3},
        ],
    }
    path = root / "experiment.yaml"
    path.write_text(yaml.safe_dump(config))
    result = CliRunner().invoke(app, ["run", str(path), "--quiet"])
    assert result.exit_code == 0, result.output
    out: Path = build_site.build_site(root / "runs" / "site-smoke", root / "site")
    return out


def test_site_has_every_page_and_asset(site: Path) -> None:
    names = {p.relative_to(site).as_posix() for p in site.rglob("*") if p.is_file()}
    assert names == {
        ".nojekyll",
        "index.html",
        "product.html",
        "report.html",
        "img/quality_vs_cost.png",
        "img/architecture.svg",
    }


def test_landing_page_is_generated_from_the_run(site: Path) -> None:
    page = (site / "index.html").read_text(encoding="utf-8")
    assert "site-smoke" in page and "over 12 queries" in page
    assert "<h3>Recommendation</h3>" in page and "<h3>All pipelines</h3>" in page
    assert "Versus the baseline (bm25-k3)" in page and "bm25-k3" in page
    for anchor in (
        "quickstart",
        "results",
        "how-evaluation-works",
        "architecture",
        "where-it-fails",
        "limitations",
    ):
        assert f'id="{anchor}"' in page
    assert 'src="img/architecture.svg"' in page and "```" not in page


def test_site_needs_no_network(site: Path) -> None:
    for page in ("index.html", "product.html"):
        html = (site / page).read_text(encoding="utf-8")
        assert "<script" not in html
        assert not re.search(r'<img[^>]+src="https?://', html)
        assert not re.search(r'<link[^>]+rel="stylesheet"', html)


def test_product_brief_has_a_table_of_contents(site: Path) -> None:
    page = (site / "product.html").read_text(encoding="utf-8")
    assert '<a href="#problem">Problem</a>' in page
    assert '<h2 id="roadmap">Roadmap</h2>' in page


def test_split_sections_ignores_headings_inside_code() -> None:
    text = "# T\n\n## A\none\n```\n## not a heading\n```\n## B\ntwo\n"
    sections = build_site.split_sections(text)
    assert list(sections) == ["A", "B"]
    assert "## not a heading" in sections["A"]


def test_rewrite_links_targets_site_and_repo() -> None:
    text = "[brief](docs/PRODUCT.md) [lic](LICENSE) ![c](docs/img/x.png) [ext](https://a.b)"
    assert build_site.rewrite_links(text, "https://g/r") == (
        "[brief](product.html) [lic](https://g/r/blob/main/LICENSE) ![c](img/x.png) "
        "[ext](https://a.b)"
    )
