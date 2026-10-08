"""Build the static documentation site (GitHub Pages) from a finished run and the repo docs.

Every number on the site comes from the run directory: the report model, the paired
comparison and the chart are recomputed here from ``per_query.jsonl`` and ``manifest.json``.
Prose comes from README.md and docs/PRODUCT.md, so the site never drifts from the repo.
The output is plain HTML and CSS with no network dependencies.

    uv run python scripts/build_site.py --run runs/example --out site
"""

from __future__ import annotations

import argparse
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import markdown
from jinja2 import Environment, FileSystemLoader, select_autoescape

from rag_eval.report import ReportData, build_report, write_report
from rag_eval.report.compare import Comparison, compare_pipelines, comparison_table
from rag_eval.report.render import (
    Table,
    baseline_table,
    by_type_table,
    context_paragraphs,
    recommendation_paragraphs,
    summary_table,
)
from rag_eval.results import load_manifest, load_records

ROOT = Path(__file__).resolve().parent.parent
TEMPLATES = ROOT / "docs" / "site"
REPO_URL = "https://github.com/seanmcrae/ragbench"
PAGES_URL = "https://seanmcrae.github.io/ragbench/"

# README sections reused verbatim on the landing page, in display order.
README_SECTIONS = (
    "Quickstart",
    "How evaluation works",
    "Architecture",
    "Design decisions",
    "Data",
    "Configuration",
    "Where it fails",
    "Limitations",
)
MERMAID_BLOCK = re.compile(r"```mermaid\n(.*?)```", flags=re.DOTALL)
MD_EXTENSIONS = ["tables", "fenced_code", "sane_lists", "attr_list", "toc"]


@dataclass(frozen=True)
class Stat:
    label: str
    value: str
    detail: str


@dataclass(frozen=True)
class HtmlTable:
    title: str
    intro: str
    headers: list[str]
    rows: list[list[str]]


def split_sections(text: str) -> dict[str, str]:
    """Markdown body of each ``## `` section, keyed by heading text."""
    sections: dict[str, str] = {}
    current: str | None = None
    lines: list[str] = []
    in_fence = False
    for line in text.splitlines():
        if line.startswith("```"):
            in_fence = not in_fence
        if not in_fence and line.startswith("## "):
            if current is not None:
                sections[current] = "\n".join(lines).strip()
            current, lines = line[3:].strip(), []
        elif current is not None:
            lines.append(line)
    if current is not None:
        sections[current] = "\n".join(lines).strip()
    return sections


def rewrite_links(text: str, repo_url: str) -> str:
    """Point README-relative links at their site or GitHub equivalents."""
    text = text.replace("](docs/PRODUCT.md)", "](product.html)")
    text = text.replace("](docs/img/", "](img/")
    text = re.sub(
        r"\]\((?!https?://|#|img/|product\.html)([^)]+)\)", rf"]({repo_url}/blob/main/\1)", text
    )
    return text


def render_markdown(text: str) -> str:
    """Markdown to HTML; Mermaid blocks become the committed SVG plus their source."""

    def diagram(match: re.Match[str]) -> str:
        source = match.group(1).replace("<", "&lt;").replace(">", "&gt;")
        return (
            '\n<figure class="diagram"><img src="img/architecture.svg" '
            'alt="Architecture: runner, retrievers, reranker, generator, cache, scoring and '
            'reports"></figure>\n<details><summary>Mermaid source</summary>'
            f"<pre><code>{source}</code></pre></details>\n"
        )

    return markdown.markdown(MERMAID_BLOCK.sub(diagram, text), extensions=MD_EXTENSIONS)


def bold_to_html(text: str) -> str:
    """Report paragraphs mark emphasis with ``**``; render them through Markdown."""
    return markdown.markdown(text)


def html_table(title: str, intro: str, table: Table) -> HtmlTable:
    return HtmlTable(title, intro, table.headers, table.rows)


def headline_stats(data: ReportData, best_vs_baseline: Comparison | None) -> list[Stat]:
    ds = data.manifest["dataset"]
    stats = [
        Stat(
            "Pipelines compared",
            str(len(data.summaries)),
            f"over {ds['queries']} queries, {ds['documents']} documents",
        )
    ]
    chosen = data.recommendation.chosen
    if chosen is not None:
        stats.append(
            Stat(
                "Recommended under budget",
                chosen.name,
                f"{data.quality_metric} {chosen.quality:.3f}, ${chosen.cost_per_1k:.3f} per 1k "
                f"queries, p95 {chosen.p95_ms:,.0f} ms",
            )
        )
    if best_vs_baseline is not None and best_vs_baseline.gate is not None:
        ci = best_vs_baseline.gate
        (ca, cb), (pa, pb) = best_vs_baseline.cost_per_1k, best_vs_baseline.p95_ms
        verdict = "significant" if ci.significant else "not significant"
        stats.append(
            Stat(
                f"Best {data.quality_metric} vs baseline",
                f"{ci.delta:+.3f}",
                f"{best_vs_baseline.b} vs {best_vs_baseline.a}, 95% CI "
                f"[{ci.low:+.3f}, {ci.high:+.3f}] ({verdict}), at {cb / ca:.1f}x the cost and "
                f"{pb / pa:.1f}x the p95",
            )
        )
    return stats


def best_quality(data: ReportData) -> str:
    best = max(data.candidates, key=lambda c: (c.quality, -c.cost_per_1k, -c.p95_ms))
    return best.name


def build_site(run_dir: Path, out_dir: Path, repo_url: str = REPO_URL) -> Path:
    records = load_records(run_dir)
    manifest = load_manifest(run_dir)
    data = build_report(records, manifest)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    img_dir = out_dir / "img"
    written = write_report(data, out_dir, img_dir)
    written.markdown.unlink()  # the site links the self-contained HTML report instead
    written.chart.unlink()
    shutil.copyfile(ROOT / "docs" / "img" / "architecture.svg", img_dir / "architecture.svg")

    best = best_quality(data)
    comparison = (
        compare_pipelines(records, manifest["config"], data.baseline, best)
        if best != data.baseline
        else None
    )
    tables = [
        html_table("All pipelines", "", summary_table(data)),
        html_table(
            f"Versus the baseline ({data.baseline})",
            f"Paired bootstrap over queries: difference in mean {data.quality_metric} with a 95% "
            "percentile interval.",
            baseline_table(data),
        ),
        html_table(f"{data.quality_metric} by query type", "", by_type_table(data)),
    ]
    if comparison is not None:
        tables.append(
            html_table(
                f"{comparison.b} vs {comparison.a}, every metric",
                "The highest-quality configuration regardless of budget, against the baseline. "
                f"Same output as <code>rag-eval compare {comparison.a} {comparison.b}</code>.",
                comparison_table(comparison),
            )
        )

    readme = rewrite_links((ROOT / "README.md").read_text(encoding="utf-8"), repo_url)
    readme_sections = split_sections(readme)
    missing = [s for s in README_SECTIONS if s not in readme_sections]
    if missing:
        raise KeyError(f"README.md is missing sections the site renders: {missing}")
    sections = [
        (title, re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-"), render_markdown(body))
        for title, body in ((t, readme_sections[t]) for t in README_SECTIONS)
    ]

    product = markdown.Markdown(extensions=MD_EXTENSIONS)
    product_md = (ROOT / "docs" / "PRODUCT.md").read_text(encoding="utf-8")
    product_md = rewrite_links(product_md.split("\n", 1)[1], repo_url)  # drop the H1
    product_html = product.convert(product_md)

    env = Environment(loader=FileSystemLoader(TEMPLATES), autoescape=select_autoescape(["html"]))
    common = {
        "repo_url": repo_url,
        "pages_url": PAGES_URL,
        "css": (TEMPLATES / "style.css").read_text(encoding="utf-8"),
        "version": manifest["rag_eval_version"],
    }
    index = env.get_template("index.html").render(
        **common,
        title="ragbench: evaluate RAG and agentic search on quality, latency and cost",
        stats=headline_stats(data, comparison),
        context=[bold_to_html(p) for p in context_paragraphs(data)],
        recommendation=[bold_to_html(p) for p in recommendation_paragraphs(data)],
        tables=tables,
        sections=sections,
        experiment=data.experiment,
        quality_metric=data.quality_metric,
    )
    (out_dir / "index.html").write_text(index, encoding="utf-8")
    product_page = env.get_template("product.html").render(
        **common,
        title="ragbench: product brief",
        body=product_html,
        toc=product.toc_tokens,  # type: ignore[attr-defined]
    )
    (out_dir / "product.html").write_text(product_page, encoding="utf-8")
    (out_dir / ".nojekyll").write_text("", encoding="utf-8")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--run", type=Path, default=Path("runs/example"), help="Run directory.")
    parser.add_argument("--out", type=Path, default=Path("site"), help="Output directory.")
    parser.add_argument("--repo-url", default=REPO_URL, help="Repository URL for links.")
    args = parser.parse_args()
    out = build_site(args.run, args.out, args.repo_url)
    print(f"wrote {out}/index.html, {out}/product.html, {out}/report.html")


if __name__ == "__main__":
    main()
