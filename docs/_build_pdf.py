"""Render a docs/*.md file to a styled PDF.

Strategy: parse the markdown **directly** into reportlab Flowables.
We skip markdown -> HTML entirely so we never have to round-trip through
a fragile HTML parser (the previous version rendered raw tags and
inlined code blocks incorrectly).

Run:
    py docs/_build_pdf.py                          # PROJECT_DOCUMENTATION.md
    py docs/_build_pdf.py --md docs/PROGRESS_PRESENTATION.md \
        --subtitle "Progress Presentation"
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Preformatted,
    Spacer,
    Table,
    TableStyle,
)

# Register Windows system fonts that support the unicode glyphs we use
# (→ ← ≥ ≤ × · χ² ≪). Helvetica (PDF Type 1) doesn't include these.
_FONT_REGULAR = "Helvetica"        # fallback for body if TTF fails
_FONT_BOLD = "Helvetica-Bold"
try:
    pdfmetrics.registerFont(TTFont("AppSans", r"C:\Windows\Fonts\segoeui.ttf"))
    pdfmetrics.registerFont(TTFont("AppSansBold", r"C:\Windows\Fonts\segoeuib.ttf"))
    pdfmetrics.registerFont(TTFont("AppMono", r"C:\Windows\Fonts\consola.ttf"))
    _FONT_REGULAR = "AppSans"
    _FONT_BOLD = "AppSansBold"
    _FONT_MONO = "AppMono"
except Exception:
    # Fall back to Helvetica; we'll lose some glyphs but the PDF still builds.
    _FONT_MONO = "Courier"

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MD_PATH = REPO_ROOT / "docs" / "PROJECT_DOCUMENTATION.md"

FRAME_WIDTH = 17 * cm


# ---------------------------------------------------------------------------
# Markdown -> Flowables
# ---------------------------------------------------------------------------
# Inline emphasis is converted to reportlab mini-tags (`<b>`, `<i>`,
# `<font name="Courier">`) so paragraphs render correctly via Paragraph.
# Code blocks (indented or fenced) become Preformatted.
# GFM tables become Table Flowables.

INLINE_CODE = re.compile(r"`([^`\n]+)`")
BOLD = re.compile(r"\*\*(.+?)\*\*")
ITALIC = re.compile(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)")
LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
CODE_FENCE = re.compile(r"^```([a-zA-Z0-9_-]*)\s*$")
TABLE_DIVIDER = re.compile(r"^\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)+\|?\s*$")
IMAGE = re.compile(r"^!\[([^\]]*)\]\(([^)]+)\)\s*$")

# Diagram languages we render as a caption instead of raw source: the
# markdown viewer draws them, but the source text is noise in a PDF.
DIAGRAM_LANGS = {"mermaid"}


def inline_md_to_rl(text: str) -> str:
    """Convert markdown inline syntax to reportlab inline tags.

    Order matters. We first locate backtick code spans and **mask** them
    so bold/italic regexes don't see `*` characters inside code spans.
    Then apply bold/italic/link. Finally unmask the code spans with the
    Courier font tag.
    """
    # 1. Escape XML-significant chars
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    # 2. Mask inline code spans with placeholders that don't contain
    #    any markdown-significant chars. Use a single sentinel char
    #    unlikely to appear in the doc text.
    sentinels: list[str] = []
    def _mask(m: re.Match) -> str:
        idx = len(sentinels)
        sentinels.append(m.group(1))
        return f"\x00CODE{idx}\x00"
    text = INLINE_CODE.sub(_mask, text)

    # 3. Bold + italic + links
    text = BOLD.sub(r"<b>\1</b>", text)
    text = ITALIC.sub(r"<i>\1</i>", text)
    text = LINK.sub(r"\1", text)

    # 4. Unmask code spans -> monospace font
    def _unmask(m: re.Match) -> str:
        idx = int(m.group(1))
        inner = sentinels[idx]
        return f'<font name="{_FONT_MONO}" color="#0A1428">{inner}</font>'
    text = re.sub(r"\x00CODE(\d+)\x00", _unmask, text)

    return text


def split_table_row(line: str) -> list[str]:
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [c.strip() for c in s.split("|")]


def is_table_header_border(line: str) -> bool:
    return bool(TABLE_DIVIDER.match(line.strip()))


# ---------------------------------------------------------------------------
# Style sheet
# ---------------------------------------------------------------------------


def build_styles():
    s = getSampleStyleSheet()

    body = ParagraphStyle(
        "BodyText",
        parent=s["BodyText"],
        fontName=_FONT_REGULAR,
        fontSize=10,
        leading=14,
        spaceAfter=6,
        textColor=colors.HexColor("#0F172A"),
    )

    def heading(name, size, before, after, color="#0A1428"):
        return ParagraphStyle(
            name,
            parent=s["Heading2"],
            fontName=_FONT_BOLD,
            fontSize=size,
            leading=size * 1.25,
            spaceBefore=before,
            spaceAfter=after,
            textColor=colors.HexColor(color),
        )

    h1 = heading("Heading1", 22, 18, 10)
    h2 = heading("Heading2", 16, 14, 8)
    h3 = heading("Heading3", 13, 10, 6)
    h4 = heading("Heading4", 11, 8, 4)
    code_style = ParagraphStyle(
        "Code",
        parent=body,
        fontName=_FONT_MONO,
        fontSize=9,
        leading=12,
        leftIndent=0,
        backColor=colors.HexColor("#F4F4F4"),
        textColor=colors.HexColor("#0A1428"),
        spaceBefore=4,
        spaceAfter=4,
        borderPadding=4,
    )
    list_item = ParagraphStyle(
        "ListItem",
        parent=body,
        leftIndent=16,
        bulletIndent=4,
        spaceBefore=2,
        spaceAfter=2,
    )
    table_cell = ParagraphStyle(
        "TableCell",
        parent=body,
        fontSize=8,
        leading=10,
        spaceAfter=0,
        spaceBefore=0,
    )
    table_head = ParagraphStyle(
        "TableHead",
        parent=table_cell,
        fontName=_FONT_BOLD,
        textColor=colors.whitesmoke,
    )
    caption = ParagraphStyle(
        "Caption",
        parent=body,
        fontSize=8,
        leading=10,
        alignment=1,
        textColor=colors.HexColor("#64748B"),
        spaceBefore=2,
        spaceAfter=8,
    )
    return {
        "body": body,
        "h1": h1,
        "h2": h2,
        "h3": h3,
        "h4": h4,
        "code": code_style,
        "list": list_item,
        "cell": table_cell,
        "thead": table_head,
        "caption": caption,
    }


# ---------------------------------------------------------------------------
# Build Flowables from markdown text
# ---------------------------------------------------------------------------


def build_flowables(md_text: str, styles: dict, base_dir: Path | None = None) -> list:
    base_dir = base_dir or REPO_ROOT
    lines = md_text.splitlines()
    flowables = []
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]
        stripped = line.strip()

        # --- Skip blank lines between blocks ---
        if not stripped:
            i += 1
            continue

        # --- Standalone image ---
        m_img = IMAGE.match(stripped)
        if m_img:
            flowables.extend(_image(m_img.group(1), m_img.group(2), base_dir, styles))
            i += 1
            continue

        # --- Fenced code block ---
        if CODE_FENCE.match(stripped):
            lang = CODE_FENCE.match(stripped).group(1)
            i += 1
            buf = []
            while i < n and not CODE_FENCE.match(lines[i].strip()):
                buf.append(lines[i])
                i += 1
            i += 1  # consume closing fence
            if lang.lower() in DIAGRAM_LANGS:
                flowables.append(Paragraph(
                    f"[{lang} diagram — see the markdown source for the rendered version]",
                    styles["caption"],
                ))
                continue
            code_text = "\n".join(buf)
            flowables.append(Spacer(1, 0.2 * cm))
            flowables.append(_code_block(code_text, styles))
            flowables.append(Spacer(1, 0.3 * cm))
            continue

        # --- Indented code block (4 spaces) ---
        if line.startswith("    ") or line.startswith("\t"):
            buf = []
            while i < n and (lines[i].startswith("    ") or lines[i].startswith("\t") or lines[i] == ""):
                buf.append(lines[i][4:] if lines[i].startswith("    ") else lines[i][1:])
                i += 1
            code_text = "\n".join(buf).rstrip("\n")
            flowables.append(Spacer(1, 0.2 * cm))
            flowables.append(_code_block(code_text, styles))
            flowables.append(Spacer(1, 0.3 * cm))
            continue

        # --- Headings ---
        m = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if m:
            level = len(m.group(1))
            text = m.group(2).strip()
            style = styles[f"h{min(level, 4)}"]
            flowables.append(Paragraph(inline_md_to_rl(text), style))
            i += 1
            continue

        # --- Horizontal rule ---
        if re.match(r"^(\*\s*){3,}$|^-\s*-\s*-$|^_+\s*$", stripped):
            flowables.append(Spacer(1, 0.3 * cm))
            t = Table([[""]], colWidths=[17 * cm], rowHeights=[0.02 * cm])
            t.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, -1), 0.5, colors.HexColor("#94A3B8"))]))
            flowables.append(t)
            flowables.append(Spacer(1, 0.3 * cm))
            i += 1
            continue

        # --- Table (GFM pipe table) ---
        # Header line + divider + body lines
        if "|" in stripped and i + 1 < n and is_table_header_border(lines[i + 1]):
            tbl_lines = [stripped, lines[i + 1].strip()]
            i += 2
            while i < n and "|" in lines[i].strip() and lines[i].strip():
                tbl_lines.append(lines[i].strip())
                i += 1
            flowables.append(_table(tbl_lines, styles))
            flowables.append(Spacer(1, 0.3 * cm))
            continue

        # --- Blockquote ---
        if stripped.startswith(">"):
            buf = []
            while i < n and lines[i].strip().startswith(">"):
                buf.append(lines[i].strip().lstrip(">").strip())
                i += 1
            quote_text = " ".join(buf)
            flowables.append(Paragraph(
                inline_md_to_rl(quote_text),
                ParagraphStyle(
                    "BlockQuote",
                    parent=styles["body"],
                    leftIndent=16,
                    textColor=colors.HexColor("#475569"),
                    borderPadding=4,
                ),
            ))
            continue

        # --- List items ---
        if re.match(r"^\s*[-*+]\s+", line) or re.match(r"^\s*\d+\.\s+", line):
            list_lines = []
            while i < n and (
                re.match(r"^\s*[-*+]\s+", lines[i])
                or re.match(r"^\s*\d+\.\s+", lines[i])
                or (lines[i].strip() and list_lines and lines[i].startswith(" "))
            ):
                list_lines.append(lines[i])
                i += 1
            flowables.extend(_list(list_lines, styles))
            continue

        # --- Paragraph (collect continuation lines) ---
        para = [stripped]
        i += 1
        while i < n:
            nxt = lines[i]
            if not nxt.strip():
                break
            if (
                CODE_FENCE.match(nxt.strip())
                or nxt.startswith("    ")
                or nxt.startswith("\t")
                or re.match(r"^(#{1,6})\s+", nxt.strip())
                or re.match(r"^\s*[-*+]\s+", nxt)
                or re.match(r"^\s*\d+\.\s+", nxt)
                or ("|" in nxt and i + 1 < n and is_table_header_border(lines[i + 1]))
                or nxt.strip().startswith(">")
                or IMAGE.match(nxt.strip())
                or re.match(r"^(\*\s*){3,}$|^-\s*-\s*-$", nxt.strip())
            ):
                break
            para.append(nxt.strip())
            i += 1
        text = " ".join(para)
        flowables.append(Paragraph(inline_md_to_rl(text), styles["body"]))

    return flowables


# ---------------------------------------------------------------------------
# Block builders
# ---------------------------------------------------------------------------


def _code_block(code_text: str, styles: dict):
    # Use Preformatted for true verbatim rendering with monospace font.
    return Preformatted(
        code_text,
        ParagraphStyle(
            "CodeBlock",
            parent=styles["body"],
            fontName=_FONT_MONO,
            fontSize=9,
            leading=12,
            backColor=colors.HexColor("#F4F4F4"),
            textColor=colors.HexColor("#0A1428"),
            borderPadding=6,
            leftIndent=0,
            rightIndent=0,
            spaceBefore=0,
            spaceAfter=0,
        ),
    )


def _image(alt: str, src: str, base_dir: Path, styles: dict) -> list:
    """Scale a figure to the frame width, capped so it fits on one page."""
    path = (base_dir / src).resolve()
    if not path.exists():
        return [Paragraph(f"[missing figure: {src}]", styles["caption"])]

    px_w, px_h = ImageReader(str(path)).getSize()
    width = min(FRAME_WIDTH, 15 * cm)
    height = width * px_h / px_w
    max_height = 11 * cm
    if height > max_height:
        width *= max_height / height
        height = max_height

    img = Image(str(path), width=width, height=height)
    img.hAlign = "CENTER"
    block = [Spacer(1, 0.2 * cm), img]
    if alt:
        block.append(Paragraph(alt, styles["caption"]))
    else:
        block.append(Spacer(1, 0.3 * cm))
    return [KeepTogether(block)]


def _table(tbl_lines: list, styles: dict) -> Table:
    header = split_table_row(tbl_lines[0])
    body_rows = [split_table_row(r) for r in tbl_lines[2:]]

    # Wrap each cell in a Paragraph so long text wraps inside the cell
    data = []
    head_row = [Paragraph(inline_md_to_rl(c), styles["thead"]) for c in header]
    data.append(head_row)
    for row in body_rows:
        # Pad / trim to header width
        cells = row + [""] * (len(header) - len(row))
        cells = cells[: len(header)]
        data.append([Paragraph(inline_md_to_rl(c), styles["cell"]) for c in cells])

    n_cols = len(header)
    page_w = 17 * cm
    col_w = page_w / n_cols

    t = Table(data, colWidths=[col_w] * n_cols, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0A1428")),
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
    return t


def _list(list_lines: list, styles: dict) -> list:
    """Render markdown list lines as a sequence of bulleted Paragraphs."""
    out = []
    for ln in list_lines:
        s = ln.strip()
        # Determine bullet
        if re.match(r"^[-*+]\s+", s):
            text = re.sub(r"^[-*+]\s+", "", s)
            bullet = "• "
            style = styles["list"]
        elif re.match(r"^\d+\.\s+", s):
            text = re.sub(r"^\d+\.\s+", "", s)
            # Pull the number from the original raw line for proper numbering
            m = re.match(r"^\s*(\d+)\.\s+", ln)
            num = m.group(1) if m else "1"
            bullet = f"{num}. "
            style = styles["list"]
        else:
            # Continuation line — append to previous (rare in our doc)
            if out:
                last = out[-1]
                last_text = last.text if hasattr(last, "text") else ""
                # best effort: just create a continuation paragraph
                out.append(Paragraph(inline_md_to_rl(s), styles["body"]))
            continue
        out.append(Paragraph(bullet + inline_md_to_rl(text), style))
    return out


# ---------------------------------------------------------------------------
# Page templates (cover + body)
# ---------------------------------------------------------------------------


def make_page_footer(subtitle: str):
    def add_page_number(canvas, doc):
        canvas.saveState()
        canvas.setFont(_FONT_REGULAR, 8)
        canvas.setFillColor(colors.HexColor("#64748B"))
        canvas.drawRightString(19 * cm, 1 * cm, f"Page {doc.page}")
        canvas.drawString(2 * cm, 1 * cm, f"Credit Risk Intelligence — {subtitle}")
        canvas.restoreState()

    return add_page_number


def make_cover(subtitle: str):
    def draw_cover(canvas, doc):
        canvas.saveState()
        W, H = A4
        # Background
        canvas.setFillColor(colors.HexColor("#0A1428"))
        canvas.rect(0, 0, W, H, fill=1, stroke=0)

        # Subtle accent rule near the top
        canvas.setFillColor(colors.HexColor("#00D9B5"))
        canvas.rect(2 * cm, H - 4 * cm, 3 * cm, 0.06 * cm, fill=1, stroke=0)

        # Title (two lines)
        canvas.setFillColor(colors.whitesmoke)
        canvas.setFont(_FONT_BOLD, 36)
        canvas.drawString(2 * cm, H - 6 * cm, "Credit Risk")
        canvas.drawString(2 * cm, H - 7.5 * cm, "Intelligence")

        # Subtitle
        canvas.setFillColor(colors.HexColor("#94A3B8"))
        canvas.setFont(_FONT_REGULAR, 18)
        canvas.drawString(2 * cm, H - 9.5 * cm, subtitle)

        # Authors
        canvas.setFont(_FONT_REGULAR, 12)
        canvas.drawString(2 * cm, H - 12 * cm, "Group DomainRange")
        canvas.drawString(2 * cm, H - 13 * cm, "UIU Data Analytics Laboratory")
        canvas.setFont(_FONT_REGULAR, 11)
        canvas.drawString(2 * cm, H - 14 * cm, "Musfique Ahmed   ·   Tasfiya Binte Karim")

        # Footer accent + dataset line
        canvas.setFillColor(colors.HexColor("#3B82F6"))
        canvas.rect(2 * cm, 4 * cm, 5 * cm, 0.05 * cm, fill=1, stroke=0)
        canvas.setFillColor(colors.HexColor("#94A3B8"))
        canvas.setFont(_FONT_REGULAR, 10)
        canvas.drawString(2 * cm, 3.3 * cm, "Home Credit Default Risk   ·   307,511 loans   ·   XGBoost AUC 0.7578")

        canvas.restoreState()

    return draw_cover


# ---------------------------------------------------------------------------
# Build the PDF
# ---------------------------------------------------------------------------


def build_pdf(md_path: Path = DEFAULT_MD_PATH, pdf_path: Path | None = None,
              subtitle: str = "Project Documentation"):
    md_path = Path(md_path)
    pdf_path = Path(pdf_path) if pdf_path else md_path.with_suffix(".pdf")
    md_text = md_path.read_text(encoding="utf-8")
    styles = build_styles()

    # Skip the first H1 (the doc title) since the cover page already shows it.
    # Also skip the very first horizontal rule immediately after it.
    all_flowables = build_flowables(md_text, styles, base_dir=md_path.parent)
    flowables = []
    skip_next_hr = False
    skipped_first_h1 = False
    for f in all_flowables:
        if (
            not skipped_first_h1
            and isinstance(f, Paragraph)
            and getattr(f.style, "name", "") == "Heading1"
        ):
            skipped_first_h1 = True
            skip_next_hr = True
            continue
        if skip_next_hr and isinstance(f, Table) and len(f._cellvalues) == 1:
            # HR is a single-cell, single-row Table
            skip_next_hr = False
            continue
        flowables.append(f)

    doc = BaseDocTemplate(
        str(pdf_path),
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
        title=f"Credit Risk Intelligence — {subtitle}",
        author="Group DomainRange",
    )

    cover_frame = Frame(0, 0, A4[0], A4[1], id="cover", showBoundary=0,
                        leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    body_frame = Frame(
        2 * cm, 2 * cm, FRAME_WIDTH, 24.5 * cm, id="body", showBoundary=0,
        leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0,
    )

    cover_template = PageTemplate(id="cover", frames=[cover_frame],
                                  onPage=make_cover(subtitle))
    body_template = PageTemplate(id="body", frames=[body_frame],
                                 onPage=make_page_footer(subtitle))
    doc.addPageTemplates([cover_template, body_template])

    # Use the cover template for page 1, then switch to the body
    # template for every subsequent page.
    story = [NextPageTemplate("body"), PageBreak()] + flowables
    doc.build(story)
    print(f"PDF written: {pdf_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--md", default=str(DEFAULT_MD_PATH),
                   help="markdown source to render")
    p.add_argument("--out", default=None,
                   help="output PDF path (default: same name as --md)")
    p.add_argument("--subtitle", default="Project Documentation",
                   help="cover subtitle and page-footer label")
    args = p.parse_args()
    build_pdf(Path(args.md), Path(args.out) if args.out else None, args.subtitle)


if __name__ == "__main__":
    main()