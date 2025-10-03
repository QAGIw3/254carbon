"""
PDF generation utilities for report-service with Jinja2 rendering and WeasyPrint.
"""

from __future__ import annotations

from typing import Any, Dict
from jinja2 import Environment, FileSystemLoader


class PDFGenerator:
    def __init__(self, template_dir: str = "templates") -> None:
        self.env = Environment(loader=FileSystemLoader(template_dir))

    def render_html(self, template_name: str, context: Dict[str, Any]) -> str:
        template = self.env.get_template(template_name)
        return template.render(**context)

    def to_pdf(self, html_content: str) -> bytes:
        try:
            from weasyprint import HTML, CSS  # type: ignore
            from weasyprint.text.fonts import FontConfiguration  # type: ignore
        except Exception as exc:  # pragma: no cover
            # Fallback to HTML bytes when system libs are missing
            return html_content.encode("utf-8")

        font_config = FontConfiguration()
        html_doc = HTML(string=html_content)
        css = CSS(
            string=
            """
            @page { size: A4; margin: 1in; }
            body { font-size: 12px; line-height: 1.4; }
            .section { page-break-inside: avoid; margin-bottom: 20px; }
            """
        )
        return html_doc.write_pdf(stylesheets=[css], font_config=font_config)


