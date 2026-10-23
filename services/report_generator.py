"""Self-contained HTML report (inline styles and logo; prints cleanly to PDF)."""
from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

BASE = Path(__file__).resolve().parent.parent
_env = Environment(loader=FileSystemLoader(BASE / "templates"), autoescape=select_autoescape(["html"]))


def render_report(result: dict) -> str:
    logo = (BASE / "assets" / "hallmark-logo.svg").read_text()
    return _env.get_template("report.html").render(r=result, logo_svg=logo)
