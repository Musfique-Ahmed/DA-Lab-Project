"""Render docs/PROJECT_DOCUMENTATION.md to a styled PDF.

Uses `markdown` to convert MD -> HTML, then `reportlab` to render a
multi-page PDF with title page, headings, tables, and code blocks.

Run:
    py docs/_build_pdf.py
"""
from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path

import markdown
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    PageBreak,
    PageTemplate,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
MD_PATH = REPO_ROOT / "docs" / "PROJECT_DOCUMENTATION.md"
PDF_PATH = REPO_ROOT / "docs" / "PROJECT_DOCUMENTATION.pdf"


# ---------------------------------------------------------------------------
# Minimal HTML parser — converts the HTML stream into a list of
# reportlab Flowables (Paragraph / Table / Spacer). Deliberately small:
# handles h1-h6, p, strong, em, code, ul/ol/li, table, thead, tbody, tr,
# th, td, hr, br. Anything else is rendered as inline runs.
# ---------------------------------------------------------------------------


class FlowBuilder(HTMLParser):
    def __init__(self, styles):
        super().__init__(convert_charrefs=True)
        self.styles = styles
        self.flowables = []
        self.stack = []           # list of dicts: {tag, attrs, buf}
        self.text_buf = ""
        self.in_table = None      # current table being built
        self.in_tr = None         # current row
        self.list_stack = []      # list of (kind, counter)
        self.paragraph_style = None  # current Paragraph style override

    # -- helpers ------------------------------------------------------------
    def _flush_text(self, into=None):
        """Flush accumulated text buffer into the topmost inline container."""
        if not self.text_buf.strip():
            self.text_buf = ""
            return
        target = self.stack[-1] if self.stack else None
        if target is not None and "buf" in target:
            target["buf"].append(self.text_buf)
        elif into is not None:
            into.append(self.text_buf)
        self.text_buf = ""

    def _new_style(self, style_name, parent="BodyText", **overrides):
        s = ParagraphStyle(
            name=style_name,
            parent=self.styles[parent],
            **overrides,
        )
        self.styles.add(s) if style_name not in self.styles.byName else None
        return self.styles[style_name]

    # -- handlers ------------------------------------------------------------
    def handle_starttag(self, tag, attrs):
        attrs_d = dict(attrs)

        # Reset text buffer into the parent before opening a new container
        self._flush_text()

        if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            level = int(tag[1])
            style_name = f"Heading{level}"
            if style_name not in self.styles.byName:
                size = {1: 22, 2: 18, 3: 14, 4: 12, 5: 11, 6: 10}[level]
                space_before = {1: 18, 2: 14, 3: 10, 4: 8, 5: 6, 6: 4}[level]
                space_after = {1: 10, 2: 8, 3: 6, 4: 4, 5: 3, 6: 2}[level]
                self._new_style(
                    style_name,
                    parent="Heading1" if level == 1 else "Heading2",
                    fontSize=size,
                    leading=size * 1.25,
                    spaceBefore=space_before,
                    spaceAfter=space_after,
                    textColor=colors.HexColor("#0A1428"),
                    fontName="Helvetica-Bold",
                )
            self.stack.append({"tag": tag, "style": self.styles[style_name], "buf": []})

        elif tag == "p":
            self.stack.append({"tag": "p", "style": self.styles["BodyText"], "buf": []})

        elif tag == "strong" or tag == "b":
            self.stack.append({"tag": "strong", "buf": []})

        elif tag == "em" or tag == "i":
            self.stack.append({"tag": "em", "buf": []})

        elif tag == "code":
            self.stack.append({"tag": "code", "buf": []})

        elif tag == "br":
            self.text_buf += "\n"

        elif tag == "hr":
            self.flowables.append(Spacer(1, 0.4 * cm))
            self.flowables.append(Table([[""]], colWidths=[16 * cm], rowHeights=[0.02 * cm],
                                         style=TableStyle([("LINEBELOW", (0, 0), (-1, -1), 0.5, colors.grey)])))

        elif tag == "ul":
            self.list_stack.append(("ul", 0))

        elif tag == "ol":
            self.list_stack.append(("ol", 0))

        elif tag == "li":
            kind, _ = self.list_stack[-1] if self.list_stack else ("ul", 0)
            if kind == "ol":
                self.list_stack[-1] = (kind, self.list_stack[-1][1] + 1)
                bullet = f"{self.list_stack[-1][1]}. "
            else:
                bullet = "• "
            self.stack.append({"tag": "li", "bullet": bullet, "buf": []})

        elif tag == "table":
            self.in_table = {"rows": [], "current_row": None, "widths": []}

        elif tag == "thead":
            self.in_table["section"] = "thead"

        elif tag == "tbody":
            self.in_table["section"] = "tbody"

        elif tag == "tr":
            self.in_table["current_row"] = []

        elif tag in ("th", "td"):
            self.stack.append({"tag": tag, "buf": []})

        elif tag == "blockquote":
            self.stack.append({"tag": "blockquote", "buf": []})

    def handle_endtag(self, tag):
        self._flush_text()
        if not self.stack and tag not in ("table", "tr", "th", "td", "thead", "tbody", "ul", "ol", "li"):
            return

        if tag in ("h1", "h2", "h3", "h4", "h5", "h6", "p"):
            node = self.stack.pop()
            text = "".join(node["buf"]).strip()
            if text:
                style = node["style"]
                self.flowables.append(Paragraph(text, style))

        elif tag in ("strong", "b", "em", "i", "code"):
            node = self.stack.pop()
            inner = "".join(node["buf"])
            if tag in ("strong", "b"):
                self.text_buf += f"<b>{inner}</b>"
            elif tag in ("em", "i"):
                self.text_buf += f"<i>{inner}</i>"
            elif tag == "code":
                self.text_buf += f'<font face="Courier" backColor="#F4F4F4">{inner}</font>'

        elif tag == "li":
            node = self.stack.pop()
            text = "".join(node["buf"]).strip()
            if text:
                bullet = node.get("bullet", "• ")
                self.flowables.append(Paragraph(
                    f"{bullet}{text}",
                    ParagraphStyle(
                        name="ListItem",
                        parent=self.styles["BodyText"],
                        leftIndent=14,
                        bulletIndent=2,
                        spaceBefore=2,
                        spaceAfter=2,
                    ),
                ))

        elif tag in ("ul", "ol"):
            if self.list_stack:
                self.list_stack.pop()
            self.flowables.append(Spacer(1, 0.15 * cm))

        elif tag in ("th", "td"):
            node = self.stack.pop()
            cell_text = "".join(node["buf"]).strip()
            if self.in_table and self.in_table["current_row"] is not None:
                self.in_table["current_row"].append(cell_text)

        elif tag == "tr":
            if self.in_table and self.in_table["current_row"] is not None:
                self.in_table["rows"].append(self.in_table["current_row"])
                self.in_table["current_row"] = None

        elif tag in ("thead", "tbody"):
            if self.in_table:
                self.in_table["section"] = None

        elif tag == "table":
            tbl = self.in_table
            self.in_table = None
            if tbl and tbl["rows"]:
                # Build Table
                data = tbl["rows"]
                # Equal-width columns
                col_w = (16 * cm) / max(len(data[0]), 1)
                t = Table(data, colWidths=[col_w] * len(data[0]), repeatRows=1)
                t.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0A1428")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, 0), 9),
                    ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                    ("FONTSIZE", (0, 1), (-1, -1), 8),
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 4),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                    ("TOPPADDING", (0, 0), (-1, -1), 3),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#CBD5E1")),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                     [colors.white, colors.HexColor("#F8FAFC")]),
                ]))
                self.flowables.append(Spacer(1, 0.2 * cm))
                self.flowables.append(t)
                self.flowables.append(Spacer(1, 0.3 * cm))

    def handle_data(self, data):
        if self.in_table is not None and self.stack:
            # buffer into the latest cell/row
            node = self.stack[-1]
            if "buf" in node:
                node["buf"].append(data)
            return
        if self.list_stack and self.stack and self.stack[-1].get("tag") == "li":
            self.stack[-1]["buf"].append(data)
            return
        self.text_buf += data

    def handle_entityref(self, name):
        # Convert a few common entities
        mapping = {"amp": "&", "lt": "<", "gt": ">", "quot": '"', "apos": "'", "nbsp": " "}
        self.text_buf += mapping.get(name, f"&{name};")

    def handle_charref(self, name):
        try:
            if name.startswith(("x", "X")):
                self.text_buf += chr(int(name[1:], 16))
            else:
                self.text_buf += chr(int(name))
        except ValueError:
            pass


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

def add_page_number(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#64748B"))
    canvas.drawRightString(20.5 * cm, 1 * cm, f"Page {doc.page}")
    canvas.drawString(2 * cm, 1 * cm, "Credit Risk Intelligence — Project Documentation")
    canvas.restoreState()


def add_cover(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(colors.HexColor("#0A1428"))
    canvas.rect(0, 0, 21 * cm, 29.7 * cm, fill=1, stroke=0)
    canvas.setFillColor(colors.HexColor("#00D9B5"))
    canvas.setFont("Helvetica-Bold", 32)
    canvas.drawString(2 * cm, 22 * cm, "Credit Risk")
    canvas.drawString(2 * cm, 20.5 * cm, "Intelligence")
    canvas.setFillColor(colors.whitesmoke)
    canvas.setFont("Helvetica", 16)
    canvas.drawString(2 * cm, 18 * cm, "Project Documentation")
    canvas.setFont("Helvetica", 12)
    canvas.setFillColor(colors.HexColor("#94A3B8"))
    canvas.drawString(2 * cm, 16 * cm, "Group DomainRange")
    canvas.drawString(2 * cm, 15 * cm, "UIU Data Analytics Laboratory")
    canvas.drawString(2 * cm, 14 * cm, "Musfique Ahmed · Tasfiya Binte Karim")
    canvas.setFillColor(colors.HexColor("#3B82F6"))
    canvas.rect(2 * cm, 13 * cm, 4 * cm, 0.05 * cm, fill=1, stroke=0)
    canvas.setFillColor(colors.HexColor("#94A3B8"))
    canvas.setFont("Helvetica", 10)
    canvas.drawString(2 * cm, 4 * cm, "Home Credit Default Risk · 307,511 loans · XGBoost AUC 0.7578")
    canvas.restoreState()


def build_pdf():
    md_text = MD_PATH.read_text(encoding="utf-8")

    # Convert markdown -> HTML
    html = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "sane_lists"],
    )

    styles = getSampleStyleSheet()
    # Tweak body text
    body = styles["BodyText"]
    body.fontName = "Helvetica"
    body.fontSize = 10
    body.leading = 13
    body.spaceAfter = 6
    body.textColor = colors.HexColor("#0F172A")

    # Heading 1 (used by "# Credit Risk...")
    h1 = styles["Heading1"]
    h1.fontName = "Helvetica-Bold"
    h1.fontSize = 24
    h1.leading = 28
    h1.spaceBefore = 24
    h1.spaceAfter = 12
    h1.textColor = colors.HexColor("#0A1428")

    h2 = styles["Heading2"]
    h2.fontName = "Helvetica-Bold"
    h2.fontSize = 18
    h2.leading = 22
    h2.spaceBefore = 16
    h2.spaceAfter = 8
    h2.textColor = colors.HexColor("#0A1428")

    # Build flowables
    builder = FlowBuilder(styles)
    builder.feed(html)
    builder._flush_text()

    # Filter out the first H1 ("# Credit Risk Intelligence — Project Documentation")
    # because the cover page already shows it
    flowables = []
    skip_first_h1 = True
    for f in builder.flowables:
        if skip_first_h1 and isinstance(f, Paragraph):
            style_name = f.style.name if hasattr(f.style, "name") else ""
            if style_name == "Heading1":
                skip_first_h1 = False
                continue
        flowables.append(f)

    # Build doc — first page is the cover, subsequent pages have page numbers
    doc = BaseDocTemplate(
        str(PDF_PATH),
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
        title="Credit Risk Intelligence — Project Documentation",
        author="Group DomainRange",
    )
    cover_frame = Frame(0, 0, 21 * cm, 29.7 * cm, id="cover", showBoundary=0)
    body_frame = Frame(
        2 * cm, 2 * cm, 17 * cm, 25.7 * cm, id="body", showBoundary=0,
        leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0,
    )
    cover_template = PageTemplate(id="cover", frames=[cover_frame], onPage=add_cover)
    body_template = PageTemplate(id="body", frames=[body_frame], onPage=add_page_number)
    doc.addPageTemplates([cover_template, body_template])

    story = [PageBreak()] + flowables
    doc.build(story)

    print(f"PDF written: {PDF_PATH}")


if __name__ == "__main__":
    build_pdf()