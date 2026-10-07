"""Keep the hand-maintained docs consistent with their single sources."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_readme_architecture_matches_diagram_source() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    blocks = re.findall(r"```mermaid\n(.*?)```", readme, flags=re.DOTALL)
    source = (ROOT / "docs" / "architecture.mmd").read_text(encoding="utf-8")
    assert blocks == [source]


def test_rendered_architecture_svg_is_committed() -> None:
    svg = (ROOT / "docs" / "img" / "architecture.svg").read_text(encoding="utf-8")
    assert svg.lstrip().startswith("<svg")
    assert "Reranker" in svg
