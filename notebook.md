# 走讀教材製作 — 通用格式與 Build 備忘

這份文件只保留可跨 project 重用的經驗。各別 project 的章節規劃、路徑、狀態、實驗結果、handoff 細節，應放在該 project 自己的 `handoff.md`、`plan.md`、`md/*` 或 spec 文件裡。

目標：從一個能跑的軟體 / 硬體專案，整理成一套可閱讀、可重生 PDF 的中文教材。

---

## 1. 核心流程

```text
repo 能 build / test
        ↓
規劃章節大綱
        ↓
撰寫 md/NN_topic.md
        ↓
補 drawio 圖與公式圖
        ↓
合併 MD → HTML → PDF
        ↓
掃 stale refs / 檢查 PDF
```

原則：

- 先讓專案能跑，再寫教材。
- 先讓 PDF pipeline 能一鍵重生，再大量寫章節。
- `md/NN_*.md` 是 source of truth；combined md、HTML、PDF 都是產物。
- 每改 2~3 章就 rebuild 一次，不要等全部寫完才看版面。

---

## 2. 建議目錄

```text
<project>/
├── notebook.md              # 通用製作規格，可由本檔複製
├── handoff.md               # project 當前狀態，不進教材
├── md/
│   ├── 00_intro.md
│   ├── 01_*.md
│   ├── ...
│   ├── <project>_textbook_combined.md   # build 產物
│   └── <project>_textbook.html          # build 產物
├── pdf/
│   └── <project>_textbook.pdf
├── drawio/
│   ├── topic.drawio
│   └── topic.drawio.png
├── eq/                      # optional: LaTeX / plot PNG
├── png/                     # optional: cover or screenshots
├── scripts/
│   ├── build_pdf.sh
│   ├── md2html.py
│   └── render_eq.py
├── samples/ or model/       # test inputs, usually ignored
└── <source-repo>/
```

Source / generated 分層：

| 層 | 例子 | 是否手改 |
|---|---|---|
| Source chapters | `md/00_intro.md` | 是 |
| Drawio source | `drawio/foo.drawio` | 是 |
| Formula source | `scripts/render_eq.py` | 是 |
| Combined MD | `md/*_combined.md` | 否，除非臨時同步 |
| HTML | `md/*.html` | 否 |
| PDF | `pdf/*.pdf` | 否 |

---

## 3. 章節規劃

### 通用 9 章模板

| # | 章節 | 內容 |
|---|---|---|
| 0 | Intro | 這個專案是什麼、學習地圖、最短跑通路徑 |
| 1 | Build & Run | 環境、編譯、測試、常用命令 |
| 2 | Domain Basics A | 第一塊必要背景 |
| 3 | Domain Basics B | 第二塊必要背景 |
| 4 | Architecture | 模組分層、資料流、控制流 |
| 5 | Core Data Structures | 主要 struct/class、生命週期、ownership |
| 6 | Main Flow A | 第一條核心路徑走讀 |
| 7 | Main Flow B | 第二條核心路徑或反向路徑 |
| 8 | Performance / Debug | 效能模型、debug playbook、練習 |

### Bottom-Up 模板

硬體、compiler、模擬器、runtime 類專案常適合 bottom-up：

| # | 章節 | 內容 |
|---|---|---|
| 0 | Intro | 整體地圖 |
| 1 | Smallest Unit | 最小可理解元件 |
| 2 | Composition | 多個元件如何組成模組 |
| 3 | Control | FSM、scheduler、dispatch |
| 4 | Data Path | buffer、memory、format |
| 5 | Program Format | descriptor、binary、IR、command |
| 6 | Frontend | parser、loader、compiler |
| 7 | Runtime | execution、simulation、verification |
| 8 | Optimization | tiling、fusion、cache、parallelism |
| 9 | Debug | regression、profile、常見失敗 |

章數可以調整；重點是每章只回答一組問題。

---

## 4. MD 寫作格式

### H1 / 導覽

每章 H1 固定：

```markdown
# 第 N 章 — Title
```

建議章首：

```markdown
> 上一章：[第 N-1 章 — ...](0N-1_*.md)

本章你會學到什麼：

- ...
- ...
```

建議章末：

```markdown
## 小結

...

> 下一章 → [第 N+1 章 — ...](0N+1_*.md)
```

PDF 頁首、章節速覽、TOC 會依賴 H1 格式；不要隨意改成其他標題樣式。

### 文字風格

| 規則 | 建議 |
|---|---|
| 目標讀者 | 假設讀者聰明但第一次碰這個 codebase |
| 語言 | 中文敘述 + 英文術語並存 |
| 節奏 | 先講目的，再講資料結構，再講流程 |
| 篇幅 | 每章 200~500 行；太長拆章 |
| 表格 | 多用「概念 / 位置 / 用途」三欄 |
| code link | 用 markdown link 指到檔案或行號 |
| code block | fenced block 並加語言標籤 |

### 禁忌

- 不要把 project 狀態、待辦、commit 記錄塞進教材章節。
- 不要用 box-drawing 字元畫大架構圖；改用 drawio。
- 不要讓章節變成 API dump；每段都要回答「為什麼讀者需要知道」。
- 不要在 source md 裡手寫自動 nav，例如「回章節速覽」；讓 `md2html.py` 注入。

---

## 5. Drawio 圖規格

### 檔案與引用

```text
drawio/<topic>.drawio
drawio/<topic>.drawio.png
```

MD 引用：

```markdown
![圖說](../drawio/<topic>.drawio.png)
```

渲染：

```bash
DRAWIO="/Applications/draw.io.app/Contents/MacOS/draw.io"
cd drawio
for f in *.drawio; do
  "$DRAWIO" -x -f png -b 10 -s 2 -o "${f}.png" "$f"
done
```

### 視覺規則

| 項目 | 規則 |
|---|---|
| 背景 | 白底，`background="#FFFFFF"` |
| 線段 | 一律 orthogonal，不走斜線 |
| 圓角 | 保守使用；工程圖通常 4~8px 以內 |
| 字體 | 10~12pt，block 要足夠容納換行 |
| 圖密度 | 一張圖只講一件事 |
| 顏色 | 用顏色區分資料、控制、memory、compute、warning |

建議配色：

| 顏色 | 用途 |
|---|---|
| 藍 | 資料、I/O、buffer |
| 綠 | 正常狀態、完成、safe path |
| 紫 | API、入口、控制 |
| 黃 | 中介層、設定、queue |
| 橘 | warning、特殊 case |
| 紅 | 錯誤、瓶頸、critical block |
| 灰 | optional、disabled、background |

### Orthogonal Edge

每條 edge 加：

```xml
style="edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;
       exitX=0.5;exitY=1;entryX=0.5;entryY=0;
       endArrow=classic;strokeColor=#666;strokeWidth=2;"
```

需要避開 block 時，用 waypoint 走 row / column gap：

```xml
<Array as="points">
  <mxPoint x="520" y="250" />
  <mxPoint x="120" y="250" />
</Array>
```

踩坑：

| 問題 | 解法 |
|---|---|
| PNG 變黑底 | `mxGraphModel` 設白色 background |
| 匯出 116-byte 空 PNG | cell value 避免內嵌 HTML 標籤 |
| edge 斜線 | 加 `edgeStyle=orthogonalEdgeStyle` |
| 線交叉或壓 block | 加 waypoint 走空白 gap |
| 文字被裁切 | 手動加大 block，高度至少覆蓋換行數 |
| ID 衝突 | 每個 cell ID 全檔唯一 |

---

## 6. 公式與圖表

不需要完整 TeX toolchain。簡單公式可用 matplotlib mathtext 渲染成 PNG。

流程：

1. 在 `scripts/render_eq.py` 記錄 `(name, latex)`。
2. 產生 `eq/<name>.png`。
3. MD 用 `![alt](../eq/<name>.png)` 引用。
4. `build_pdf.sh` 的第一步自動重生公式圖。

mathtext 常見限制：

| 寫不出來 | 替代 |
|---|---|
| `\begin{cases}` | 拆成多張圖或用 `\min` / `\max` |
| `align` 多行 | 拆多張圖 |
| `\bm{}` | 用 `\mathbf{}` |
| `\text{中文}` | 中文放在 markdown，不放公式 |

CSS 建議：

```css
img[src*="/eq/"] {
  max-width: 45%;
  border: none;
}
```

曲線圖、profiling 圖、散點圖若資訊密度高，可放寬到 75%~100%。

---

## 7. Build Pipeline

一鍵 build：

```bash
bash scripts/build_pdf.sh
```

典型步驟：

| 階段 | 輸入 | 輸出 |
|---|---|---|
| render eq | `scripts/render_eq.py` | `eq/*.png` |
| render drawio | `drawio/*.drawio` | `drawio/*.drawio.png` |
| concatenate | `md/[0-9][0-9]_*.md` | `md/*_combined.md` |
| markdown to HTML | combined md | `md/*.html` |
| HTML to PDF | html | `pdf/*.pdf` |

Chrome headless 建議參數：

```bash
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
  --headless=new --disable-gpu --no-pdf-header-footer \
  --print-to-pdf="pdf/<project>_textbook.pdf" \
  --virtual-time-budget=30000 --run-all-compositor-stages-before-draw \
  "file://$PWD/md/<project>_textbook.html"
```

`--virtual-time-budget=30000` 和 `--run-all-compositor-stages-before-draw` 很重要，否則圖片可能還沒載完 PDF 就輸出。

---

## 8. PDF / HTML 規格

### 基本樣式

| 項目 | 建議 |
|---|---|
| Page | A4 |
| Margin | 20mm / 16mm / 18mm / 16mm |
| Body font | PingFang TC 或系統中文字體 fallback |
| Body size | 10pt，line-height 1.55~1.65 |
| H1 | 20pt，章節新頁 |
| H2 | 15~16pt |
| Code font | 8~8.5pt |
| Code theme | Pygments friendly |
| Image | drawio 70%，公式 45%，高密度圖 75~100% |

### TOC 模式

小書可以只有完整目錄；大書建議兩層：

| 區塊 | 用途 |
|---|---|
| Cover | 書名、副標、版本或日期 |
| Chapter Overview | Part 分組 + 每章一行 |
| Full TOC | H1/H2/H3 完整目錄 |
| Chapters | 正文 |

兩層 TOC 的好處：

- Overview 用來找大方向。
- Full TOC 用來找細節主題。
- 每章可以自動插「回章節速覽」link。

### `md2html.py` 實作要點

- 先用 markdown 產 HTML body。
- 在改寫 H1 前，先掃 `第 N 章 — Title` 建 chapter list。
- 依 `PARTS` 產生 chapter overview。
- 把 H1 拆成 `chnum` / `chname` span，給 `@page string-set` 使用。
- 在 H1 後和 page break 前自動注入 chapter nav。
- 把 `\newpage` 換成 `<div class="page-break"></div>`。

---

## 9. 新增 / 刪除章節 Checklist

新增章節：

```text
1. 新增 md/NN_topic.md
2. 補上一章 / 下一章 link
3. 更新前後章的導覽 link
4. 若使用 PARTS，更新 md2html.py 的章節範圍
5. 補 drawio / eq / screenshot
6. bash scripts/build_pdf.sh
7. 檢查 PDF 目錄、頁首、章節速覽
```

刪除章節：

```text
1. 刪除 md/NN_topic.md
2. 重新編號後續章節或明確保留缺號
3. 更新所有上一章 / 下一章 link
4. 更新 PARTS
5. rg 掃舊章名與舊檔名
6. rebuild PDF
```

搬移檔案或改 CLI 後：

```bash
rg -n "old/path|old_command|old_filename" md notebook.md handoff.md
git diff --check -- md/*.md notebook.md scripts/*.py scripts/*.sh
```

---

## 10. PDF 前檢查清單

```bash
bash scripts/build_pdf.sh
git diff --check -- md/*.md notebook.md scripts/*.py scripts/*.sh
rg -n "TODO|FIXME|old/path|old_command|old_filename" md notebook.md handoff.md
```

人工檢查：

- 封面正常。
- Chapter overview 有列到所有章。
- Full TOC 沒有奇怪的空章節。
- 頁首顯示正確章名。
- drawio 圖沒有黑底、斜線、線壓字、文字裁切。
- code block 沒有超出頁面太多。
- 公式大小一致。
- PDF 頁數合理，檔案大小沒有暴增。

可抽頁檢查：

```bash
pdftoppm -r 120 -f <start> -l <end> pdf/<project>_textbook.pdf /tmp/textbook_page -png
```

---

## 11. 常見踩坑

| 問題 | 解法 |
|---|---|
| 改了 source md，但 PDF 沒變 | 確認有跑 `scripts/build_pdf.sh`，且章節檔名符合 `[0-9][0-9]_*.md` |
| 新章有正文但 overview 沒列 | 更新 `md2html.py` 的 `PARTS` |
| 頁首章名錯 | H1 必須是 `# 第 N 章 — Title`，特殊頁 H1 要清 `string-set` |
| 圖片沒進 PDF | Chrome 加 `--virtual-time-budget`，路徑用相對路徑 |
| drawio 圖黑底 | 設 `background="#FFFFFF"` |
| drawio 線交叉 | 使用 orthogonal edge + waypoint |
| 公式 render 失敗 | 避免 mathtext 不支援的 LaTeX，拆成多張 |
| combined md 被手改後又消失 | combined md 是產物；要改 source chapters |
| 舊命令殘留 | `rg` 掃 md / handoff / notebook |
| code block 太寬 | 縮短行、拆段、或 CSS 降低 code font size |

---

## 12. 給 AI 助手的標準請求

啟動新教材：

> 我要做 `<project>` 走讀教材。repo 在 `<path>`，已經能 build/test。請先依通用 9 章模板列教材大綱，標出每章要讀的核心檔案、需要的 drawio 圖、需要的實驗命令。

撰寫章節：

> 請寫 `md/NN_topic.md`。讀者是第一次接觸這個 codebase 的工程師。請對照 `<files>` 走讀，使用 `# 第 N 章 — Title`，章首列「本章你會學到什麼」，章末有小結和下一章導覽。

重生 PDF：

> 請跑 `bash scripts/build_pdf.sh`，若失敗請修 build script / md / drawio，最後檢查 PDF 產物存在。

改圖：

> 請修改 `drawio/<file>.drawio` 的 `<diagram/page>`。線段全部用 orthogonal，不要斜線，不要交叉，不要壓到文字，改完重生 PNG。

同步文件：

> 我改了 `<path/command/name>`，請掃所有 md / notebook / handoff，把舊引用更新，然後跑 `git diff --check`。

---

## 13. 最短速查

```text
寫章節:  md/NN_topic.md
畫圖:    drawio/foo.drawio -> drawio/foo.drawio.png
公式:    scripts/render_eq.py -> eq/foo.png
重生:    bash scripts/build_pdf.sh
檢查:    git diff --check; rg old/path md notebook.md handoff.md
原則:    source md 可手改；combined/html/pdf 不手改
```
