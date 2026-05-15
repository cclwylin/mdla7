#!/usr/bin/env python3
"""
transformer.md → transformer.pdf
Format follows notebook.md spec:
  A4, margin 20/16/18/16mm, STHeiti CJK font
  Cover → TOC → Chapters (H1 new page)
"""
from __future__ import annotations
import re
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.platypus import (
    BaseDocTemplate, PageTemplate, Frame,
    Paragraph, Spacer, HRFlowable, Table, TableStyle,
    PageBreak, NextPageTemplate, KeepTogether, Image,
)
from reportlab.platypus.flowables import Flowable
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.pdfmetrics import registerFontFamily

# ── paths ──────────────────────────────────────────────────────────────────────
BASE = Path("/Volumes/4T_OFFICE/_Claude/MDLA7_Claude")
SRC  = BASE / "transformer.md"
OUT  = BASE / "pdf" / "transformer.pdf"
OUT.parent.mkdir(parents=True, exist_ok=True)

# ── fonts ──────────────────────────────────────────────────────────────────────
pdfmetrics.registerFont(TTFont("CJK",  "/System/Library/Fonts/STHeiti Medium.ttc", subfontIndex=0))
pdfmetrics.registerFont(TTFont("CJKB", "/System/Library/Fonts/STHeiti Medium.ttc", subfontIndex=0))
registerFontFamily("CJK", normal="CJK", bold="CJKB", italic="CJK", boldItalic="CJKB")
MONO = "Courier"

# ── colours ────────────────────────────────────────────────────────────────────
NAVY   = colors.HexColor("#1E2761")
MID    = colors.HexColor("#2D3E8A")
ICE    = colors.HexColor("#CADCFC")
ACCENT = colors.HexColor("#4FC3F7")
GREEN  = colors.HexColor("#1B7B4E")
WHITE  = colors.white
OFFWH  = colors.HexColor("#F7F9FF")
LGRAY  = colors.HexColor("#E8EEF7")
DGRAY  = colors.HexColor("#2D3748")
MGRAY  = colors.HexColor("#64748B")
CBKG   = colors.HexColor("#F4F4F8")
CBRD   = colors.HexColor("#D0D8F0")

# ── page geometry (notebook.md spec) ──────────────────────────────────────────
PW, PH = A4
ML, MR, MT, MB = 20*mm, 16*mm, 18*mm, 16*mm
TW = PW - ML - MR   # text width

# ── styles ─────────────────────────────────────────────────────────────────────
def S(name, **kw):
    return ParagraphStyle(name, fontName="CJK", **kw)

sH1   = S("h1", fontSize=20, leading=26, spaceBefore=18, spaceAfter=8,
           textColor=NAVY)
sH2   = S("h2", fontSize=15, leading=21, spaceBefore=14, spaceAfter=5,
           textColor=MID)
sH3   = S("h3", fontSize=11, leading=16, spaceBefore=10, spaceAfter=3,
           textColor=colors.HexColor("#0f3460"))
sBODY = S("body", fontSize=10, leading=16, spaceAfter=4, textColor=DGRAY)
sBULL = S("bull", fontSize=10, leading=15, leftIndent=14, spaceAfter=2,
           textColor=DGRAY)
sCODE = ParagraphStyle("code", fontName="CJK", fontSize=8.5, leading=13,
                        backColor=CBKG, leftIndent=6, rightIndent=6, spaceAfter=6,
                        textColor=colors.HexColor("#1A1F36"))
sCAPT = S("capt", fontSize=8, leading=12, textColor=MGRAY, spaceAfter=4)
sTOC1 = S("toc1", fontSize=11, leading=16, spaceAfter=2, textColor=NAVY)
sTOC2 = S("toc2", fontSize=9.5, leading=14, leftIndent=14, spaceAfter=1,
           textColor=DGRAY)

TBL_STYLE = TableStyle([
    ("BACKGROUND",   (0,0), (-1,0),  NAVY),
    ("TEXTCOLOR",    (0,0), (-1,0),  WHITE),
    ("FONTNAME",     (0,0), (-1,0),  "CJKB"),
    ("FONTNAME",     (0,1), (-1,-1), "CJK"),
    ("FONTSIZE",     (0,0), (-1,-1), 8.5),
    ("LEADING",      (0,0), (-1,-1), 13),
    ("ROWBACKGROUNDS",(0,1),(-1,-1), [OFFWH, WHITE]),
    ("GRID",         (0,0), (-1,-1), 0.4, CBRD),
    ("LEFTPADDING",  (0,0), (-1,-1), 6),
    ("RIGHTPADDING", (0,0), (-1,-1), 6),
    ("TOPPADDING",   (0,0), (-1,-1), 4),
    ("BOTTOMPADDING",(0,0), (-1,-1), 4),
    ("VALIGN",       (0,0), (-1,-1), "TOP"),
])

# ── inline markup ──────────────────────────────────────────────────────────────
def esc(s):
    return s.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")

def inline(s):
    s = esc(s)
    s = re.sub(r"`([^`]+)`",
               r'<font name="Courier" size="8" color="#c0392b">\1</font>', s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"\*(.+?)\*",     r"<i>\1</i>",  s)
    return s

# ── page templates ─────────────────────────────────────────────────────────────

class HeaderCanvas:
    def __init__(self, title="Transformer 入門筆記"):
        self.title = title

    def __call__(self, canvas, doc):
        if doc.page == 1:          # cover — no header/footer
            return
        canvas.saveState()
        # header line
        canvas.setStrokeColor(NAVY); canvas.setLineWidth(0.5)
        canvas.line(ML, PH-MT+4*mm, PW-MR, PH-MT+4*mm)
        canvas.setFont("CJK", 8); canvas.setFillColor(MGRAY)
        canvas.drawString(ML, PH-MT+5*mm, self.title)
        canvas.drawRightString(PW-MR, PH-MT+5*mm, "MDLA7")
        # footer
        canvas.line(ML, MB-4*mm, PW-MR, MB-4*mm)
        canvas.setFont("CJK", 8)
        canvas.drawCentredString(PW/2, MB-8*mm,
                                 f"— {doc.page - 1} —")
        canvas.restoreState()


def make_doc():
    hdr = HeaderCanvas()
    content_frame = Frame(ML, MB, TW, PH-MT-MB, id="content")
    cover_frame   = Frame(0, 0, PW, PH, id="cover",
                          leftPadding=0, rightPadding=0,
                          topPadding=0,  bottomPadding=0)
    doc = BaseDocTemplate(
        str(OUT), pagesize=A4,
        leftMargin=ML, rightMargin=MR,
        topMargin=MT,  bottomMargin=MB,
        title="Transformer 入門筆記",
        author="MDLA7",
    )
    doc.addPageTemplates([
        PageTemplate(id="Cover",   frames=[cover_frame],   onPage=hdr),
        PageTemplate(id="Content", frames=[content_frame], onPage=hdr),
    ])
    return doc

# ── cover ──────────────────────────────────────────────────────────────────────

class CoverPage(Flowable):
    def draw(self):
        c = self.canv
        w, h = PW, PH
        # background
        c.setFillColor(NAVY); c.rect(0, 0, w, h, fill=1, stroke=0)
        # accent strip
        c.setFillColor(MID);  c.rect(w*0.6, 0, w*0.4, h, fill=1, stroke=0)
        c.setFillColor(colors.HexColor("#253070"))
        c.rect(w*0.78, 0, w*0.22, h, fill=1, stroke=0)
        # green bar
        c.setFillColor(GREEN); c.rect(ML, h*0.52, 60*mm, 2*mm, fill=1, stroke=0)
        # title
        c.setFont("CJKB", 36); c.setFillColor(WHITE)
        c.drawString(ML, h*0.60, "Transformer")
        c.setFont("CJK",  30); c.setFillColor(ACCENT)
        c.drawString(ML, h*0.52, "入門筆記")
        # subtitle
        c.setFont("CJK", 13); c.setFillColor(ICE)
        c.drawString(ML, h*0.44, "從 Q K V 到 GPT / LLaMA / Qwen3.5")
        # detail
        c.setFont("CJK", 10); c.setFillColor(MGRAY)
        c.drawString(ML, h*0.12, "MDLA7 Architecture Team")
        c.setFont("CJK",  9)
        c.drawString(ML, h*0.08, "2026")

    def wrap(self, aw, ah): return PW, PH


# ── markdown → flowables ───────────────────────────────────────────────────────

def code_para(lines):
    """Code block: escape and join with <br/>."""
    parts = []
    for ln in lines:
        parts.append(ln.replace("&","&amp;")
                       .replace("<","&lt;")
                       .replace(">","&gt;")
                       .replace(" ","&#160;"))
    return Paragraph("<br/>".join(parts), sCODE)


def parse_table(rows_raw):
    data = []
    for row in rows_raw:
        if re.match(r"^\s*\|[\s\-:]+\|", row):
            continue
        cells = [c.strip() for c in row.strip().strip("|").split("|")]
        data.append([Paragraph(inline(c), sBODY) for c in cells])
    if not data:
        return None
    ncols = max(len(r) for r in data)
    cw    = TW / ncols
    tbl   = Table(data, colWidths=[cw]*ncols, repeatRows=1)
    tbl.setStyle(TBL_STYLE)
    return tbl


def md_to_flowables(md: str):
    lines   = md.splitlines()
    story   = []
    i       = 0
    toc_items = []   # (level, text)

    while i < len(lines):
        ln = lines[i]

        # H1
        if ln.startswith("# ") and not ln.startswith("## "):
            txt = ln[2:].strip()
            toc_items.append((1, txt))
            story.append(PageBreak())
            story.append(Paragraph(txt, sH1))
            story.append(HRFlowable(width="100%", thickness=1.5,
                                     color=NAVY, spaceAfter=6))
            i += 1; continue

        # H2
        if ln.startswith("## "):
            txt = ln[3:].strip()
            toc_items.append((2, txt))
            story.append(Spacer(1, 4))
            story.append(Paragraph(txt, sH2))
            story.append(HRFlowable(width="100%", thickness=0.5,
                                     color=ICE, spaceAfter=3))
            i += 1; continue

        # H3
        if ln.startswith("### "):
            txt = ln[4:].strip()
            story.append(Paragraph(txt, sH3))
            i += 1; continue

        # HR
        if ln.strip() in ("---", "***", "___"):
            story.append(HRFlowable(width="100%", thickness=0.4,
                                     color=CBRD, spaceBefore=4, spaceAfter=4))
            i += 1; continue

        # fenced code block
        if ln.strip().startswith("```"):
            i += 1
            code_lines = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i]); i += 1
            i += 1
            story.append(code_para(code_lines))
            continue

        # table
        if ln.strip().startswith("|"):
            rows_raw = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows_raw.append(lines[i]); i += 1
            tbl = parse_table(rows_raw)
            if tbl:
                story.append(tbl)
                story.append(Spacer(1, 6))
            continue

        # image  ![alt](path)
        img_m = re.match(r"!\[([^\]]*)\]\(([^)]+)\)", ln.strip())
        if img_m:
            alt, path = img_m.group(1), img_m.group(2)
            img_path = BASE / path
            if img_path.exists():
                from PIL import Image as PILImage
                with PILImage.open(img_path) as im:
                    iw, ih = im.size
                max_w = TW
                scale = max_w / iw
                disp_w = max_w
                disp_h = ih * scale
                # cap height at 18cm
                if disp_h > 180*mm:
                    scale  = 180*mm / ih
                    disp_w = iw * scale
                    disp_h = 180*mm
                story.append(Spacer(1, 4))
                story.append(Image(str(img_path), width=disp_w, height=disp_h))
                if alt:
                    story.append(Paragraph(alt, sCAPT))
                story.append(Spacer(1, 6))
            else:
                story.append(Paragraph(f"[圖片未找到: {path}]", sCAPT))
            i += 1; continue

        # blockquote
        if ln.startswith("> "):
            story.append(Paragraph(inline(ln[2:]),
                ParagraphStyle("bq", fontName="CJK", fontSize=9.5, leading=15,
                               leftIndent=10, textColor=MGRAY,
                               borderPad=0, spaceAfter=4)))
            i += 1; continue

        # bullet
        if re.match(r"^\s*[-*]\s", ln):
            txt = re.sub(r"^\s*[-*]\s", "", ln)
            story.append(Paragraph("•  " + inline(txt), sBULL))
            i += 1; continue

        # numbered list
        if re.match(r"^\s*\d+\.\s", ln):
            txt = re.sub(r"^\s*\d+\.\s", "", ln)
            story.append(Paragraph(inline(txt), sBULL))
            i += 1; continue

        # blank
        if ln.strip() == "":
            story.append(Spacer(1, 4))
            i += 1; continue

        # normal paragraph
        story.append(Paragraph(inline(ln), sBODY))
        i += 1

    return story, toc_items


def make_toc(toc_items):
    items = []
    items.append(Paragraph("目　錄", S("toch", fontSize=16, leading=22,
                                       textColor=NAVY, spaceAfter=10)))
    items.append(HRFlowable(width="100%", thickness=1.5, color=NAVY,
                             spaceAfter=10))
    for level, txt in toc_items:
        if level == 1:
            items.append(Paragraph(txt, sTOC1))
        else:
            items.append(Paragraph("  " + txt, sTOC2))
    return items


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    md = SRC.read_text(encoding="utf-8")
    content, toc_items = md_to_flowables(md)

    doc   = make_doc()
    story = []

    # 1. Cover (rendered on Cover template, then switch to Content)
    story.append(CoverPage())
    story.append(NextPageTemplate("Content"))
    story.append(PageBreak())

    # 2. TOC
    story += make_toc(toc_items)
    story.append(PageBreak())

    # 3. Content
    story += content

    doc.build(story)
    sz = OUT.stat().st_size
    print(f"Written: {OUT}  ({sz//1024} KB)")


if __name__ == "__main__":
    main()
