#!/usr/bin/env python3
"""Convert softmax.md to softmax.pdf using reportlab."""

from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Preformatted,
    Table, TableStyle, HRFlowable,
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

BASE = Path("/Volumes/4T_OFFICE/_Claude/MDLA7_Claude")
SRC  = BASE / "softmax.md"
OUT  = BASE / "softmax.pdf"

# ── register CJK font ─────────────────────────────────────────────────────────
pdfmetrics.registerFont(TTFont("CJK",      "/System/Library/Fonts/STHeiti Medium.ttc", subfontIndex=0))
pdfmetrics.registerFont(TTFont("CJK-Bold", "/System/Library/Fonts/STHeiti Medium.ttc", subfontIndex=0))
from reportlab.pdfbase.pdfmetrics import registerFontFamily
registerFontFamily("CJK", normal="CJK", bold="CJK-Bold", italic="CJK", boldItalic="CJK-Bold")

FONT_BODY = "CJK"
FONT_CODE = "Courier"

# ── styles ────────────────────────────────────────────────────────────────────
styles = getSampleStyleSheet()

S_H1 = ParagraphStyle("h1", fontName=FONT_BODY,
                       fontSize=16, leading=22, spaceAfter=6,
                       textColor=colors.HexColor("#1a1a2e"))
S_H2 = ParagraphStyle("h2", fontName=FONT_BODY,
                       fontSize=13, leading=18, spaceBefore=14, spaceAfter=4,
                       textColor=colors.HexColor("#16213e"))
S_H3 = ParagraphStyle("h3", fontName=FONT_BODY,
                       fontSize=11, leading=16, spaceBefore=10, spaceAfter=3,
                       textColor=colors.HexColor("#0f3460"))
S_BODY = ParagraphStyle("body", fontName=FONT_BODY,
                         fontSize=9.5, leading=16, spaceAfter=4)
S_BULLET = ParagraphStyle("bullet", fontName=FONT_BODY,
                           fontSize=9.5, leading=15,
                           leftIndent=14, bulletIndent=4, spaceAfter=2)
S_CODE = ParagraphStyle("code", fontName=FONT_BODY,
                         fontSize=8.5, leading=13,
                         backColor=colors.HexColor("#f4f4f4"),
                         leftIndent=8, rightIndent=8,
                         spaceAfter=6)
S_CAPTION = ParagraphStyle("caption", fontName=FONT_BODY,
                            fontSize=8, leading=12,
                            textColor=colors.grey, spaceAfter=6)

TBL_STYLE = TableStyle([
    ("BACKGROUND",  (0, 0), (-1, 0),  colors.HexColor("#16213e")),
    ("TEXTCOLOR",   (0, 0), (-1, 0),  colors.white),
    ("FONTNAME",    (0, 0), (-1, 0),  "CJK-Bold"),
    ("FONTNAME",    (0, 1), (-1, -1), "CJK"),
    ("FONTSIZE",    (0, 0), (-1, -1), 8),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1),
     [colors.HexColor("#f9f9f9"), colors.white]),
    ("GRID",        (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
    ("LEFTPADDING",  (0, 0), (-1, -1), 6),
    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ("TOPPADDING",   (0, 0), (-1, -1), 4),
    ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
    ("VALIGN",       (0, 0), (-1, -1), "TOP"),
])

# ── markdown parser ───────────────────────────────────────────────────────────

def escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def parse_inline(s: str) -> str:
    """Bold, code spans."""
    import re
    s = escape(s)
    s = re.sub(r"`([^`]+)`", r'<font name="Courier" size="8" color="#c0392b">\1</font>', s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"\*(.+?)\*", r"<i>\1</i>", s)
    return s

def md_to_flowables(md: str) -> list:
    lines = md.splitlines()
    story = []
    i = 0
    W = A4[0] - 40*mm   # usable width

    while i < len(lines):
        line = lines[i]

        # headings
        if line.startswith("### "):
            story.append(Paragraph(parse_inline(line[4:]), S_H3))
            i += 1; continue
        if line.startswith("## "):
            story.append(HRFlowable(width="100%", thickness=0.5,
                                     color=colors.HexColor("#16213e"), spaceAfter=4))
            story.append(Paragraph(parse_inline(line[3:]), S_H2))
            i += 1; continue
        if line.startswith("# "):
            story.append(Paragraph(parse_inline(line[2:]), S_H1))
            i += 1; continue

        # horizontal rule
        if line.strip() in ("---", "***", "___"):
            story.append(HRFlowable(width="100%", thickness=0.5,
                                     color=colors.HexColor("#cccccc"),
                                     spaceBefore=4, spaceAfter=4))
            i += 1; continue

        # fenced code block
        if line.strip().startswith("```"):
            i += 1
            code_lines = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            i += 1  # closing ```
            # Use Paragraph with explicit line breaks so CJK font renders correctly
            import re as _re
            def _code_escape(s):
                return s.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;").replace(" ","&#160;")
            parts = []
            for ci, cl in enumerate(code_lines):
                parts.append(_code_escape(cl))
            xml = "<br/>".join(parts)
            story.append(Paragraph(xml, S_CODE))
            continue

        # table
        if line.strip().startswith("|") and i + 1 < len(lines):
            rows_raw = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows_raw.append(lines[i])
                i += 1
            # filter separator rows
            rows = [r for r in rows_raw
                    if not all(c in "|-: " for c in r.replace("|", ""))]
            if rows:
                data = []
                for row in rows:
                    cells = [c.strip() for c in row.strip().strip("|").split("|")]
                    data.append([Paragraph(parse_inline(c), S_BODY) for c in cells])
                col_n = max(len(r) for r in data)
                col_w = W / col_n
                tbl = Table(data, colWidths=[col_w] * col_n, repeatRows=1)
                tbl.setStyle(TBL_STYLE)
                story.append(tbl)
                story.append(Spacer(1, 6))
            continue

        # blockquote
        if line.startswith("> "):
            story.append(Paragraph(
                parse_inline(line[2:]),
                ParagraphStyle("bq", fontName=FONT_BODY, fontSize=9.5, leading=15,
                               leftIndent=12, textColor=colors.HexColor("#555555"))
            ))
            i += 1; continue

        # todo / bullet
        if line.strip().startswith("- [ ] "):
            story.append(Paragraph("☐  " + parse_inline(line.strip()[6:]), S_BULLET))
            i += 1; continue
        if line.strip().startswith("- [x] "):
            story.append(Paragraph("☑  " + parse_inline(line.strip()[6:]), S_BULLET))
            i += 1; continue
        if line.strip().startswith("- "):
            story.append(Paragraph("•  " + parse_inline(line.strip()[2:]), S_BULLET))
            i += 1; continue

        # blank line
        if line.strip() == "":
            story.append(Spacer(1, 4))
            i += 1; continue

        # normal paragraph
        story.append(Paragraph(parse_inline(line), S_BODY))
        i += 1

    return story


# ── build PDF ─────────────────────────────────────────────────────────────────

def main() -> None:
    md = SRC.read_text(encoding="utf-8")
    story = md_to_flowables(md)

    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=A4,
        leftMargin=20*mm, rightMargin=20*mm,
        topMargin=20*mm,  bottomMargin=20*mm,
        title="Softmax MB Chain – L1Mesh Stress Test",
        author="MDLA7",
    )
    doc.build(story)
    print(f"Written: {OUT}  ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
