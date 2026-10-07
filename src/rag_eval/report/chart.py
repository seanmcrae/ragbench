"""Quality-versus-cost scatter with the Pareto frontier, the budget line and the pick."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: reports are written in CI and containers

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from rag_eval.pareto import Candidate, pareto_frontier
from rag_eval.report.model import ReportData

CHOSEN, FRONTIER, OTHER = "#c0392b", "#2c3e50", "#95a5a6"


def _clusters(ax: Axes, candidates: list[Candidate], radius_px: float = 14.0) -> list[list[int]]:
    """Group candidates whose markers would overlap on screen, so they share one label."""
    points = ax.transData.transform([(c.cost_per_1k, c.quality) for c in candidates])
    groups: list[list[int]] = []
    for i, (x, y) in enumerate(points):
        for group in groups:
            gx, gy = points[group[0]]
            if abs(gx - x) <= radius_px and abs(gy - y) <= radius_px:
                group.append(i)
                break
        else:
            groups.append([i])
    return groups


def plot_quality_vs_cost(data: ReportData, path: Path) -> Path:
    rec = data.recommendation
    candidates = data.candidates
    frontier = pareto_frontier(candidates)
    on_frontier = {c.name for c in frontier}
    chosen = rec.chosen.name if rec.chosen else None

    fig, (ax, side) = plt.subplots(
        1, 2, figsize=(10, 5), dpi=120, gridspec_kw={"width_ratios": [2.4, 1]}
    )
    for c in candidates:
        if c.name == chosen:
            color, marker, size, z = CHOSEN, "*", 280, 4
        elif c.name in on_frontier:
            color, marker, size, z = FRONTIER, "o", 70, 3
        else:
            color, marker, size, z = OTHER, "o", 55, 2
        ax.scatter(c.cost_per_1k, c.quality, color=color, marker=marker, s=size, zorder=z)

    if len(frontier) > 1:
        ax.plot(
            [c.cost_per_1k for c in frontier],
            [c.quality for c in frontier],
            color=FRONTIER,
            linewidth=1,
            linestyle="--",
            zorder=1,
            label="Pareto frontier (all configs)",
        )
    max_cost = rec.budget.max_cost_per_1k_usd
    if max_cost is not None:
        ax.axvline(max_cost, color=CHOSEN, linewidth=1, alpha=0.6, label="cost budget")

    xs = [c.cost_per_1k for c in candidates]
    ys = [c.quality for c in candidates]
    ax.set_xlim(0, max([*xs, max_cost or 0]) * 1.15 or 1)
    pad = max((max(ys) - min(ys)) * 0.12, 0.01)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    ax.set_xlabel("Cost per 1,000 queries (USD, illustrative prices)")
    ax.set_ylabel(data.quality_metric)
    ax.grid(alpha=0.3)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="lower right", fontsize=8)

    fig.canvas.draw()  # finalise limits so data->pixel transforms are accurate
    for group in _clusters(ax, candidates):
        anchor = candidates[group[0]]
        ax.annotate(
            ",".join(str(i + 1) for i in group),
            (anchor.cost_per_1k, anchor.quality),
            textcoords="offset points",
            xytext=(8, 5),
            fontsize=8,
            color="#34495e",
        )

    lines = [
        f"{i + 1}. {c.name}\n    {c.quality:.3f} | ${c.cost_per_1k:.2f} | {c.p95_ms:,.0f} ms"
        for i, c in enumerate(candidates)
    ]
    side.axis("off")
    side.text(
        0.0,
        1.0,
        f"{data.quality_metric} | $/1k | p95\n\n" + "\n".join(lines),
        va="top",
        family="monospace",
        fontsize=7.5,
        color="#34495e",
    )
    modeled = " (generation latency modeled)" if data.latency_modeled else ""
    fig.suptitle(f"{data.experiment}: quality vs cost{modeled}", fontsize=11)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path
