"""Render a ReportData as Markdown, a self-contained HTML page, or a plain-text table."""

from __future__ import annotations

import base64
import html
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from rag_eval.report.model import ReportData


@dataclass(frozen=True)
class Table:
    headers: list[str]
    rows: list[list[str]]


@dataclass(frozen=True)
class Section:
    title: str
    paragraphs: list[str]
    table: Table | None = None


def _ms(value: float) -> str:
    return f"{value:,.0f}"


def _confidence_level(data: ReportData) -> int:
    return round(data.deltas[0].ci.confidence * 100) if data.deltas else 95


def _verdict(significant: bool, delta: float) -> str:
    if not significant:
        return "no significant difference"
    return "better" if delta > 0 else "worse"


def summary_table(data: ReportData) -> Table:
    m = data.quality_metric
    secondary = [k for k in ("exact_match", "judge_score", "groundedness") if k != m]
    headers = [
        "pipeline",
        m,
        *secondary,
        "nDCG@5",
        "recall@5",
        "MRR",
        "p50 ms",
        "p95 ms",
        "$/1k q",
        "steps",
    ]
    rows = []
    for s in data.summaries:
        rows.append(
            [
                s.name,
                f"{s.metrics[m]:.3f}",
                *(f"{s.metrics[k]:.3f}" if k in s.metrics else "n/a" for k in secondary),
                f"{s.metrics.get('ndcg@5', float('nan')):.3f}",
                f"{s.metrics.get('recall@5', float('nan')):.3f}",
                f"{s.metrics['mrr']:.3f}",
                _ms(s.latency_ms["total_p50"]),
                _ms(s.latency_ms["total_p95"]),
                f"{s.cost_per_1k_usd:.3f}",
                f"{s.mean_steps:.2f}",
            ]
        )
    return Table(headers, rows)


def baseline_table(data: ReportData) -> Table:
    level = _confidence_level(data)
    headers = ["pipeline", f"delta {data.quality_metric}", f"{level}% CI", "p", "verdict"]
    rows = [
        [
            d.pipeline,
            f"{d.ci.delta:+.3f}",
            f"[{d.ci.low:+.3f}, {d.ci.high:+.3f}]",
            f"{d.ci.p_value:.3f}",
            _verdict(d.ci.significant, d.ci.delta),
        ]
        for d in data.deltas
    ]
    return Table(headers, rows)


def by_type_table(data: ReportData) -> Table:
    types = data.query_types
    rows = [
        [s.name, *(f"{s.quality_by_type[t]:.2f}" if t in s.quality_by_type else "-" for t in types)]
        for s in data.summaries
    ]
    return Table(["pipeline", *types], rows)


def recommendation_paragraphs(data: ReportData) -> list[str]:
    rec = data.recommendation
    limits = []
    if rec.budget.max_cost_per_1k_usd is not None:
        limits.append(f"cost <= ${rec.budget.max_cost_per_1k_usd:.2f} per 1k queries")
    if rec.budget.max_p95_latency_ms is not None:
        limits.append(f"p95 latency <= {rec.budget.max_p95_latency_ms:,.0f} ms")
    budget = " and ".join(limits) or "no budget"
    out = [
        f"Budget: {budget}. Selection: highest {data.quality_metric} on the in-budget Pareto frontier."
    ]
    if rec.chosen is None:
        out.append("No configuration fits the budget; relax it or add cheaper configurations.")
    else:
        c = rec.chosen
        out.append(
            f"Recommended: **{c.name}** ({data.quality_metric} {c.quality:.3f}, "
            f"${c.cost_per_1k:.3f} per 1k queries, p95 {_ms(c.p95_ms)} ms)."
        )
    best = max(data.candidates, key=lambda c: (c.quality, -c.cost_per_1k, -c.p95_ms))
    if rec.chosen is not None and best.quality > rec.chosen.quality:
        c = rec.chosen
        out.append(
            f"Best {data.quality_metric} regardless of budget: {best.name} ({best.quality:.3f}, "
            f"{best.quality - c.quality:+.3f} vs the recommendation) at "
            f"{best.cost_per_1k / c.cost_per_1k:.1f}x the cost and "
            f"{best.p95_ms / c.p95_ms:.1f}x the p95 latency."
        )
    frontier = ", ".join(c.name for c in rec.frontier) or "none"
    out.append(f"In-budget Pareto frontier (cheapest first): {frontier}.")
    on_frontier = {c.name for c in rec.frontier}
    dominated = [
        c.name for c in data.candidates if c.name not in on_frontier and c.name not in rec.rejected
    ]
    if dominated:
        out.append(f"In budget but dominated: {', '.join(dominated)}.")
    if rec.rejected:
        out.append(
            "Over budget: " + "; ".join(f"{n} ({why})" for n, why in rec.rejected.items()) + "."
        )
    return out


def context_paragraphs(data: ReportData) -> list[str]:
    ds = data.manifest["dataset"]
    out = [
        f"Dataset: {ds['name']} ({ds['documents']} documents, {ds['queries']} queries, "
        f"fingerprint {ds['fingerprint']}). Prices: illustrative table dated "
        f"{data.manifest['prices_as_of']}. Run created {data.manifest['created_at']}."
    ]
    if data.latency_modeled:
        out.append(
            "Generation latency comes from the offline mock's token-based latency model, not "
            "from a live API; retrieval latency is measured wall-clock."
        )
    if not ds["has_reference_answers"]:
        out.append(
            "This dataset has no reference answers, so answer F1 and exact match are skipped."
        )
    return out


def sections(data: ReportData) -> list[Section]:
    level = _confidence_level(data)
    return [
        Section("Recommendation", recommendation_paragraphs(data)),
        Section("Results", [], summary_table(data)),
        Section(
            f"Versus baseline ({data.baseline})",
            [
                f"Paired bootstrap over queries, {level}% percentile intervals on the difference "
                f"in mean {data.quality_metric}."
            ],
            baseline_table(data),
        ),
        Section(f"{data.quality_metric} by query type", [], by_type_table(data)),
    ]


def _md_table(table: Table) -> list[str]:
    lines = ["| " + " | ".join(table.headers) + " |"]
    lines.append("|" + "|".join("---" for _ in table.headers) + "|")
    lines.extend("| " + " | ".join(row) + " |" for row in table.rows)
    return lines


def render_markdown(data: ReportData, chart_name: str) -> str:
    lines = [f"# Evaluation report: {data.experiment}", ""]
    for paragraph in context_paragraphs(data):
        lines += [paragraph, ""]
    for section in sections(data):
        lines += [f"## {section.title}", ""]
        for paragraph in section.paragraphs:
            lines += [paragraph, ""]
        if section.table is not None:
            lines += [*_md_table(section.table), ""]
        if section.title == "Recommendation":
            lines += [f"![Quality vs cost]({chart_name})", ""]
    return "\n".join(lines)


_CSS = """
body{font-family:system-ui,-apple-system,sans-serif;max-width:1100px;margin:2rem auto;
padding:0 1rem;color:#1f2933;line-height:1.5}
table{border-collapse:collapse;margin:1rem 0;font-size:.9rem}
th,td{border:1px solid #d9e2ec;padding:.35rem .6rem;text-align:right}
th:first-child,td:first-child{text-align:left}
th{background:#f0f4f8}img{max-width:100%;border:1px solid #d9e2ec}
"""


def _inline(text: str) -> str:
    escaped = html.escape(text)
    parts = escaped.split("**")
    return "".join(f"<strong>{p}</strong>" if i % 2 else p for i, p in enumerate(parts))


def _html_table(table: Table) -> str:
    head = "".join(f"<th>{html.escape(h)}</th>" for h in table.headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in row) + "</tr>" for row in table.rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def render_html(data: ReportData, chart_path: Path) -> str:
    """Single-file HTML: the chart is embedded as a data URI so the page travels alone."""
    encoded = base64.b64encode(chart_path.read_bytes()).decode("ascii")
    parts = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        f"<title>Evaluation report: {html.escape(data.experiment)}</title>",
        f"<style>{_CSS}</style></head><body>",
        f"<h1>Evaluation report: {html.escape(data.experiment)}</h1>",
        *(f"<p>{_inline(p)}</p>" for p in context_paragraphs(data)),
    ]
    for section in sections(data):
        parts.append(f"<h2>{html.escape(section.title)}</h2>")
        parts.extend(f"<p>{_inline(p)}</p>" for p in section.paragraphs)
        if section.table is not None:
            parts.append(_html_table(section.table))
        if section.title == "Recommendation":
            parts.append(f"<img alt='Quality vs cost' src='data:image/png;base64,{encoded}'>")
    parts.append("</body></html>")
    return "\n".join(parts)


def render_text_table(table: Table) -> str:
    widths = [
        max(len(r[i]) for r in [table.headers, *table.rows]) for i in range(len(table.headers))
    ]

    def line(cells: Sequence[str]) -> str:
        first, *rest = cells
        return "  ".join(
            [first.ljust(widths[0])] + [c.rjust(w) for c, w in zip(rest, widths[1:], strict=True)]
        )

    return "\n".join([line(table.headers), *(line(r) for r in table.rows)])
