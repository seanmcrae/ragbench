"""Report generation: model -> Markdown / HTML / chart."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from rag_eval.report.chart import plot_quality_vs_cost
from rag_eval.report.model import BaselineDelta, ReportData, build_report
from rag_eval.report.render import render_html, render_markdown

CHART_FILE = "quality_vs_cost.png"

__all__ = ["BaselineDelta", "ReportData", "WrittenReport", "build_report", "write_report"]


@dataclass(frozen=True)
class WrittenReport:
    markdown: Path
    html: Path
    chart: Path


def write_report(data: ReportData, run_dir: Path, img_dir: Path | None = None) -> WrittenReport:
    chart = plot_quality_vs_cost(data, run_dir / CHART_FILE)
    markdown = run_dir / "report.md"
    markdown.write_text(render_markdown(data, CHART_FILE), encoding="utf-8")
    page = run_dir / "report.html"
    page.write_text(render_html(data, chart), encoding="utf-8")
    if img_dir is not None:
        img_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(chart, img_dir / CHART_FILE)
    return WrittenReport(markdown, page, chart)
