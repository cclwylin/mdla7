#!/usr/bin/env python3
import sys, re, markdown
from pathlib import Path
from pygments.formatters import HtmlFormatter

src = Path(sys.argv[1]).read_text(encoding="utf-8")
src = re.sub(r'(?m)^\\newpage\s*$', '<div class="page-break"></div>', src)

md = markdown.Markdown(extensions=[
    'extra', 'toc', 'fenced_code', 'codehilite', 'tables', 'sane_lists'
], extension_configs={
    'codehilite': {'guess_lang': False, 'noclasses': False, 'pygments_style': 'friendly'},
    'toc': {'title': '完整目錄', 'permalink': False, 'toc_depth': '1-3'},
})
body = md.convert(src)
toc = md.toc

# Part 分界
PARTS = [
    ("Part I — 入門與地圖", 0, 1),
    ("Part II — HW Spec", 2, 5),
    ("Part III — Compiler", 6, 8),
    ("Part IV — SystemC Architecture", 9, 13),
    ("Part V — Scheduler & Performance", 14, 16),
    ("Part VI — Verification", 17, 18),
    ("Part VII — 實戰與 Roadmap", 19, 21),
]

# 從 body 抓 chapter H1（split_h1 之前的原始形式）
ch_pattern = re.compile(
    r'<h1 id="([^"]+)">(第\s*(\d+)\s*章)\s*[—–-]\s*([^<]+)</h1>'
)
chapters = {}
for m in ch_pattern.finditer(body):
    anchor_id = m.group(1)
    ch_num = int(m.group(3))
    title_only = m.group(4).strip()
    chapters[ch_num] = (anchor_id, title_only)

# 建章節速覽 HTML
overview_parts = ['<div class="ch-overview"><h1 id="ch-overview">章節速覽</h1>']
for part_name, start, end in PARTS:
    overview_parts.append(f'<h2>{part_name}</h2><ul>')
    for ch_num in range(start, end + 1):
        if ch_num in chapters:
            anchor, name = chapters[ch_num]
            overview_parts.append(
                f'<li><a href="#{anchor}">第 {ch_num} 章 — {name}</a></li>'
            )
    overview_parts.append('</ul>')
overview_parts.append('</div>')
overview_html = ''.join(overview_parts)

# split_h1：把 H1 拆成 chnum/chsep/chname 三個 span（給頁首抓字串）
def split_h1(m):
    open_tag, num, sep, name, close = (m.group(1), m.group(2), m.group(3),
                                        m.group(4), m.group(5))
    return (f'{open_tag}<span class="chnum">{num}</span>'
            f'<span class="chsep">{sep}</span>'
            f'<span class="chname">{name}</span>{close}')
body = re.sub(
    r'(<h1[^>]*>)(第\s*\d+\s*章)(\s*[—–-]\s*)(.+?)(</h1>)',
    split_h1, body)

# 章首注入「回章節速覽」link（在 H1 之後立刻插）
top_nav = '<p class="ch-nav-top"><a href="#ch-overview">← 回章節速覽</a></p>'
body = re.sub(
    r'(<h1 id="[^"]+"><span class="chnum">第\s*\d+\s*章</span>'
    r'<span class="chsep">[^<]*</span>'
    r'<span class="chname">[^<]+</span></h1>)',
    r'\1' + top_nav,
    body
)

# 章末注入：每個 page-break 前面加 bottom nav；最後一章加在 body 結尾
bottom_nav = '<p class="ch-nav-bot"><a href="#ch-overview">← 回章節速覽</a></p>'
body = body.replace(
    '<div class="page-break"></div>',
    bottom_nav + '\n<div class="page-break"></div>'
)
body += '\n' + bottom_nav

pyg_css = HtmlFormatter(style='friendly').get_style_defs('.codehilite')

html = f"""<!doctype html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<title>SystemC 教材</title>
<style>
@page {{
  size: A4; margin: 20mm 16mm 18mm 16mm;
  @top-left {{
    content: string(chnum);
    font-family: "PingFang TC", "Heiti TC", sans-serif;
    font-size: 9pt; color: #666;
  }}
  @top-right {{
    content: string(chapter);
    font-family: "PingFang TC", "Heiti TC", sans-serif;
    font-size: 9pt; color: #666;
  }}
  @bottom-center {{
    content: counter(page);
    font-family: "PingFang TC", "Heiti TC", sans-serif;
    font-size: 9.5pt; color: #555;
  }}
}}
@page :first {{ @top-left {{ content: ""; }} @top-right {{ content: ""; }} @bottom-center {{ content: ""; }} }}
@page cover       {{ @top-left {{ content: ""; }} @top-right {{ content: ""; }} @bottom-center {{ content: ""; }} }}
@page ch_overview {{ @top-left {{ content: ""; }} @top-right {{ content: "章節速覽"; }} }}
@page toc         {{ @top-left {{ content: ""; }} @top-right {{ content: "完整目錄"; }} }}
.cover       {{ page: cover; }}
.ch-overview {{ page: ch_overview; }}
.toc         {{ page: toc; }}
html, body {{ -webkit-print-color-adjust: exact; print-color-adjust: exact; }}
body {{
  font-family: "PingFang TC", "Heiti TC", "Hiragino Sans GB", "Microsoft JhengHei", sans-serif;
  font-size: 10pt; line-height: 1.6; color: #222; margin: 0;
}}
h1, h2, h3, h4 {{ font-weight: 700; line-height: 1.3; break-inside: avoid; }}
h1 {{ font-size: 20pt; border-bottom: 2px solid #333; padding-bottom: 6px; margin-top: 0; break-before: page; }}
h1 .chnum  {{ string-set: chnum content(); }}
h1 .chname {{ string-set: chapter content(); }}
.cover h1, .toc h1, .ch-overview h1 {{ break-before: auto; string-set: chnum "", chapter ""; }}
h2 {{ font-size: 15pt; border-bottom: 1px solid #bbb; padding-bottom: 4px; margin-top: 1.6em; }}
h3 {{ font-size: 12pt; margin-top: 1.4em; }}
h4 {{ font-size: 11pt; }}
p {{ margin: 0.6em 0; }}
code {{
  font-family: "JetBrains Mono", "Menlo", "Consolas", monospace;
  font-size: 8.5pt; background: #f3f3f3; padding: 1px 4px; border-radius: 3px;
}}
pre {{
  background: #f7f7f7; border: 1px solid #e1e1e1; border-radius: 5px;
  padding: 10px 12px; overflow: hidden; white-space: pre-wrap; word-wrap: break-word;
}}
pre code {{ background: none; padding: 0; font-size: 8pt; line-height: 1.45; }}
table {{ border-collapse: collapse; margin: 0.8em 0; font-size: 9pt; }}
th, td {{ border: 1px solid #bbb; padding: 5px 9px; }}
th {{ background: #eee; }}
blockquote {{
  border-left: 4px solid #888; margin: 0.8em 0; padding: 0.2em 1em;
  color: #555; background: #fafafa;
}}
a {{ color: #1554a4; text-decoration: none; }}
img {{
  display: block; max-width: 70%; height: auto;
  margin: 1em auto; border: 1px solid #ddd; border-radius: 4px;
  background: #fff; padding: 6px;
  break-inside: avoid;
}}
p > img {{ margin: 1em auto; }}
img[src*="/eq/"] {{
  max-width: 45%;
  border: none; padding: 0; background: none;
  margin: 0.6em auto;
}}
img[src*="relu_curves"], img[src*="roofline"] {{
  max-width: 75%;
}}
img[src*="model_scale"], img[src*="model_scale_zoom"] {{
  max-width: 100%;
}}
.page-break {{ break-after: page; }}

/* 章節速覽 */
.ch-overview {{
  break-after: page;
  border: 1px solid #ddd;
  padding: 18px 24px;
  border-radius: 6px;
  background: #fbfbfb;
}}
.ch-overview h1 {{
  border: none;
  font-size: 22pt;
  text-align: center;
  margin: 0 0 0.6em 0;
}}
.ch-overview h2 {{
  margin: 0.9em 0 0.3em;
  border: none;
  font-size: 13pt;
  color: #1554a4;
  padding-bottom: 0;
}}
.ch-overview ul {{
  list-style: none;
  padding-left: 0.4em;
  margin: 0 0 0.4em 0;
  column-count: 1;
}}
.ch-overview li {{
  margin: 2px 0;
  font-size: 10pt;
  break-inside: avoid;
}}
.ch-overview a {{ color: #222; }}

/* 完整目錄 */
.toc {{
  break-after: page;
  border: 1px solid #ddd; padding: 16px 22px; border-radius: 6px;
  background: #fbfbfb;
}}
.toc > ul {{ list-style: none; padding-left: 0; }}
.toc ul ul {{ padding-left: 1.4em; }}
.toc li {{ margin: 3px 0; }}
.toc a {{ color: #222; }}

/* 章首章末 nav */
.ch-nav-top {{
  text-align: right;
  font-size: 9pt;
  margin: 0.3em 0 0.8em 0;
}}
.ch-nav-bot {{
  text-align: right;
  font-size: 9pt;
  margin: 1.6em 0 0.4em 0;
}}
.ch-nav-top a, .ch-nav-bot a {{
  color: #666;
  text-decoration: none;
  border: 1px solid #ccc;
  padding: 2px 8px;
  border-radius: 3px;
  background: #fafafa;
}}

/* 封面 */
.cover {{ text-align: center; padding-top: 110px; height: calc(297mm - 36mm - 110px); break-after: page; }}
.cover h1 {{ border: none; font-size: 26pt; break-before: auto; margin-bottom: 4px; }}
.cover .sub {{ color: #555; font-size: 13pt; margin-top: 8px; margin-bottom: 90px; }}
.cover-arch {{ max-width: 75%; margin: 30px auto 0 auto; }}
.cover-arch img {{ width: 100%; height: auto; display: block; margin: 0; border: 1px solid #ddd; border-radius: 6px; padding: 0; background: #fff; }}
{pyg_css}
</style>
</head>
<body>
<div class="cover">
  <h1>SystemC 原理與實作走讀</h1>
  <div class="sub">Kernel · Channel · TLM · AMS · Bus · Memory · Pipeline · NoC · Cache</div>
</div>
{overview_html}
<div class="toc"><h1 style="page-break-before: avoid;">完整目錄</h1>{toc}</div>
{body}
</body>
</html>
"""
Path(sys.argv[2]).write_text(html, encoding="utf-8")
print(f"wrote {sys.argv[2]} ({len(html)} bytes)")
