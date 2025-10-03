"""
Advanced PDF generation utilities for report-service with Jinja2 rendering and WeasyPrint.
Supports multi-column layouts, TOC generation, headers/footers, and vector graphics.
"""

from __future__ import annotations

import base64
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple
from jinja2 import Environment, FileSystemLoader


class PDFGenerator:
    def __init__(self, template_dir: str = "templates") -> None:
        self.env = Environment(loader=FileSystemLoader(template_dir))
        self.toc_items: List[Dict[str, Any]] = []
        self.page_count = 0

    def render_html(self, template_name: str, context: Dict[str, Any]) -> str:
        template = self.env.get_template(template_name)
        return template.render(**context)

    def add_toc_item(self, title: str, level: int = 1, page: int = 1) -> None:
        """Add an item to the table of contents."""
        self.toc_items.append({
            "title": title,
            "level": level,
            "page": page
        })

    def generate_toc_html(self) -> str:
        """Generate HTML for table of contents."""
        if not self.toc_items:
            return ""

        toc_html = '<div class="toc">\n<h2>Table of Contents</h2>\n<ul>\n'
        for item in self.toc_items:
            indent = "  " * (item["level"] - 1)
            toc_html += f'{indent}<li><a href="#page-{item["page"]}">{item["title"]}</a></li>\n'
        toc_html += '</ul>\n</div>'
        return toc_html

    def to_pdf(self, html_content: str, options: Optional[Dict[str, Any]] = None) -> bytes:
        """Generate PDF with advanced features."""
        try:
            from weasyprint import HTML, CSS  # type: ignore
            from weasyprint.text.fonts import FontConfiguration  # type: ignore
        except Exception as exc:  # pragma: no cover
            # Fallback to HTML bytes when system libs are missing
            return html_content.encode("utf-8")

        font_config = FontConfiguration()

        # Enhanced CSS for advanced PDF features
        css_content = self._generate_pdf_css()

        # Add TOC if items exist
        if self.toc_items:
            toc_html = self.generate_toc_html()
            # Insert TOC after title page or at beginning
            html_content = html_content.replace('</head>', f'<style>{css_content}</style></head>')
            # Simple insertion - in production would need more sophisticated placement
            html_content = html_content.replace('<body>', f'<body>{toc_html}', 1)

        html_doc = HTML(string=html_content)

        # Default options
        pdf_options = {
            "stylesheets": [CSS(string=css_content)],
            "font_config": font_config,
        }

        # Merge with provided options
        if options:
            pdf_options.update(options)

        return html_doc.write_pdf(**pdf_options)

    def _generate_pdf_css(self) -> str:
        """Generate comprehensive CSS for PDF formatting."""
        return """
        @page {
            size: A4;
            margin: 1in;
            @bottom-right {
                content: "Page " counter(page) " of " counter(pages);
                font-size: 10px;
                color: #666;
            }
            @top-center {
                content: "254Carbon Market Intelligence";
                font-size: 10px;
                color: #666;
            }
        }

        @page :first {
            @top-center { content: ""; }
            @bottom-center {
                content: "Confidential - Internal Use Only";
                font-size: 8px;
                color: #999;
            }
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            font-size: 12px;
            line-height: 1.5;
            color: #333;
        }

        /* Multi-column layout support */
        .columns-2 {
            column-count: 2;
            column-gap: 20px;
            column-rule: 1px solid #ddd;
        }

        .columns-3 {
            column-count: 3;
            column-gap: 15px;
        }

        /* Header styles */
        .header {
            text-align: center;
            border-bottom: 3px solid #2c3e50;
            padding-bottom: 20px;
            margin-bottom: 30px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
            padding: 30px;
            page-break-after: avoid;
        }

        .header h1 {
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }

        .header .subtitle {
            margin: 10px 0 0 0;
            font-size: 1.2em;
            opacity: 0.9;
        }

        /* Section styles */
        .section {
            margin: 40px 0;
            page-break-inside: avoid;
        }

        .section h2 {
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
            margin-bottom: 20px;
            page-break-after: avoid;
        }

        /* Chart containers */
        .chart-container {
            margin: 30px 0;
            text-align: center;
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            page-break-inside: avoid;
        }

        /* Table styles */
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }

        th, td {
            border: 1px solid #dee2e6;
            padding: 15px;
            text-align: left;
        }

        th {
            background: #2c3e50;
            color: white;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.85em;
        }

        tr:nth-child(even) {
            background: #f8f9fa;
        }

        /* Footer */
        .footer {
            margin-top: 50px;
            padding-top: 20px;
            border-top: 1px solid #dee2e6;
            text-align: center;
            color: #6c757d;
            font-size: 0.9em;
        }

        /* Summary cards */
        .summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }

        .summary-card {
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            transition: transform 0.2s ease;
        }

        .summary-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        }

        .summary-value {
            font-size: 2.5em;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 5px;
        }

        .summary-label {
            color: #6c757d;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        /* Metadata box */
        .metadata {
            background: #f8f9fa;
            border-left: 4px solid #3498db;
            padding: 15px;
            margin: 20px 0;
            border-radius: 5px;
        }

        .metadata strong {
            color: #2c3e50;
        }

        /* Price highlighting */
        .price-positive {
            color: #28a745;
            font-weight: bold;
        }

        .price-negative {
            color: #dc3545;
            font-weight: bold;
        }

        .highlight {
            background: linear-gradient(90deg, #ffeaa7 0%, #fab1a0 100%);
            padding: 2px 6px;
            border-radius: 4px;
            font-weight: bold;
        }

        /* TOC styles */
        .toc {
            page-break-after: always;
            margin: 40px 0;
        }

        .toc h2 {
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }

        .toc ul {
            list-style: none;
            padding: 0;
        }

        .toc li {
            margin: 8px 0;
            padding-left: 20px;
            position: relative;
        }

        .toc li:before {
            content: "•";
            color: #3498db;
            position: absolute;
            left: 0;
        }

        .toc a {
            color: #2c3e50;
            text-decoration: none;
        }

        .toc a:hover {
            color: #3498db;
            text-decoration: underline;
        }

        /* Watermark */
        .watermark {
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%) rotate(-45deg);
            font-size: 72px;
            color: rgba(200, 200, 200, 0.3);
            z-index: -1;
            pointer-events: none;
        }
        """


