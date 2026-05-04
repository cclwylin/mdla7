# 走讀教材製作 — 寫作 + Build 流程備忘

從「找一個軟體 / 硬體專案」到「產出一本含目錄、頁碼、章節頁首、drawio 圖、LaTeX 公式的中文 PDF 教材」的完整流程。本檔合併兩份來源：

- **通用 SW 走讀流程**（FFmpeg、SQLite、emulator 都用過）
- **TPU v6e 專屬**（13 章 bottom-up + LaTeX 公式 PNG，本目錄是 reference impl）

整個流程不依賴 pandoc / TeX，純 Python + Chrome + draw.io CLI。

---

## 1. 流程概覽

```
GitHub 下載 → 編譯 / 跑起來 → 規劃章節大綱
              ↓
        撰寫章節 MD（md/NN_topic.md）
              ↓
        畫方塊圖（drawio/*.drawio → *.drawio.png）
              ↓
        渲染公式（LaTeX → eq/*.png，optional）
              ↓
        合併 MD → HTML → PDF（Chrome headless）
```

build 端一條龍：`bash scripts/build_pdf.sh`。

---

## 2. 階段 1 — 從 GitHub 取得專案

```bash
cd /Volumes/4T_OFFICE/_Claude
mkdir -p <ProjectName>          # FFmpeg / SQLite / Linux / TPU_v6e ...
cd <ProjectName>
git clone <repo-url> <subdir>   # 通常 ./<repo>/ 子目錄
```

**選 repo 的條件**（學習用優先）：

- C / C++ / Rust / Go / Python（讀程式比較順）
- 有 README、有官方範例（`examples/`、`tests/`）、能本地編譯
- 程式碼乾淨、模組分明
- 有持續更新（避免太古老、被棄置）
- 中等規模：**幾萬到一兩百萬行**最理想；太大就只走「核心子集」

**不同領域推薦的起點**：

| 領域 | 範例 repo |
|---|---|
| 多媒體 | FFmpeg、GStreamer、libvpx、libaom |
| 編譯器 | TCC（Bellard）、Lua、CPython、QuickJS |
| 資料庫 | SQLite、Redis、LevelDB |
| 作業系統 | xv6、Linux subsystem、seL4 |
| 圖形 / 遊戲引擎 | tinyrenderer、bgfx、Doom3、SDL |
| 機器學習 | tinygrad、llama.cpp、ggml、micrograd |
| 模擬器 | NES / GB / PS1 / NDS |
| 瀏覽器引擎 | Servo subsystems、Chromium V8 |
| 硬體 / 加速器 | TPU v6e（本專案）、Gemmini、cocotb 範例 |

---

## 3. 階段 2 — 編譯 / 跑起來

```bash
# CMake
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j8
# Autotools
./configure && make -j8
# Cargo / Go / Python
cargo build --release  ;  go build ./...  ;  python -m pytest

# SystemC（TPU 用）
cd systemc && make all && for tb in build/tb_*; do "$tb"; done
```

**先確定能執行**（或 `make test` 通過），否則寫教材沒意義。寫到程式碼走讀章節時，要實際對著程式碼說「跑起來會看到 X」，編不過根本沒法驗證。

測試輸入（音檔、影片、ROM、`.tflite`、…）放 `samples/` 或 `model/`，**不要 commit**。

---

## 4. 階段 3 — 規劃章節大綱

兩種模板擇一：

### A) 通用 9 章模板（軟體走讀）

| # | 章節 | 內容 |
|---|---|---|
| 1 | 專案總覽 | 它是什麼、解決什麼問題、整體架構、術語、學習路徑 |
| 2 | 領域基礎 A | 第一塊基礎概念（FFmpeg：影像格式；DB：SQL/B-tree；編譯器：文法） |
| 3 | 領域基礎 B | 第二塊（FFmpeg：音訊；DB：transaction；編譯器：AST） |
| 4 | 模組架構 | 程式碼分層、檔案組織、外掛 / vtable、依賴關係 |
| 5 | 核心資料結構 | 主要 struct/class、生命週期、慣例（記憶體、refcount、錯誤碼） |
| 6 | 主流程走讀 A | 第一條核心路徑（FFmpeg：解碼；DB：query plan） |
| 7 | 主流程走讀 B | 第二條對偶路徑（FFmpeg：編碼；DB：write） |
| 8 | 演算法與效能 | 內部關鍵演算法、SIMD、平行化、HW 加速 |
| 9 | 編譯與動手實驗 | configure / make、CLI 一條龍、9 道由淺入深的修改作業 |

### B) Bottom-up 模板（HW / 加速器，TPU v6e 用這套，13 章）

| # | 章節 | 內容 |
|---|---|---|
| 0 | intro | 整體架構 + 學習地圖 |
| 1 | PE | 最小單元（1-cycle MAC） |
| 2 | systolic array | N×N PE 串成陣列 |
| 3 | MXU | + FSM、auto-skew |
| 4 | BF16 datapath | template + bf16_t |
| 5 | TensorCore | + VPU(ReLU)，matmul + activation 整合 |
| 6 | Conv2D | im2col → tpu_matmul wrapping |
| 7 | TFLite loader | flatc + 真實 .tflite 檔 |
| 8 | Depthwise + FC | block-diagonal W、weight transpose |
| 9 | quantization | INT8 對稱、scale/zero_point |
| 10 | multi-tile | M/K/N 拆分 |
| 11 | MobileNet block | 端到端 5 層 pipeline |
| 12 | extra ops | host op (HARD_SWISH、SE、BN fold...) |

不夠就 6 章、需要更多就 12+ 章——**結構與節奏比章數重要**。

---

## 5. 階段 4 — 撰寫 MD

### 寫作風格約定

| 規則 | 細節 |
|---|---|
| 目標讀者 | **大一新生**——背景假設只有微積分 + 數位邏輯 |
| 語言 | 中文敘述 + 英文術語並存（systolic / im2col / quantization 不翻） |
| 邊做邊寫 | 每跑通一個 testbench 就寫一章，不集中事後補 |
| 章節長度 | 每章 200~400 行；超過要拆 |
| H1 格式 | `# 第 N 章 — Title`，破折號 `—`（觸發 PDF 頁首抓 chnum/chapter） |
| 章間導覽 | 開頭 `> 上一章：[第 N-1 章 — ...](0N-1_*.md)`、結尾 `> 下一章 →` |
| 章首/章末 | 開頭一段「本章你會學到什麼」、結尾「小結 + 下一章預告」 |
| 程式碼引用 | `[file.h:42](../path/file.h#L42)` markdown link 格式 |
| Code block | fenced + lang（` ```c ` / ` ```cpp ` / ` ```bash ` / ` ```text `） |
| 表格密度 | 多用「術語 / 規格 / 對應 API」三欄表，初學者最受用 |
| 數學式 | LaTeX → PNG（見 §7），**不要**直接寫 `Σ_k a_ik · w_kj` 這種 ASCII |
| 重要術語 | 第一次出現用粗體 + 中文括號：**systolic（脈動陣列）** |

### 禁忌

- **不要 box-drawing 字元方塊圖**：`├─┤` 那種一律用 drawio。純位址 / 純 dataflow ASCII 才保留 fenced block
- 不要敬語（請、歡迎、希望）；直接說明
- 不要寫太短的小節（< 5 行）；要嘛合併、要嘛展開

### 對 Claude 的標準話術（撰寫章節）

> 「我在做 `<專案>` 的走讀教材，repo 在 `<路徑>`。請參考 TPU_v6e 教材（`/Volumes/4T_OFFICE/_Claude/TPU_v6e/md/`）的風格與深度，幫我寫第 X 章 `<章名>`，重點是：（1）對照 `<檔案>` 的程式碼走讀；（2）每節結尾接下一節；（3）程式碼引用用 markdown link 格式。」

---

## 6. 階段 5 — drawio 方塊圖

### 規則

- 原始檔 `drawio/<topic>.drawio`，匯出 PNG 同名 `<topic>.drawio.png` 同目錄
- MD 引用：`![圖說](../drawio/<topic>.drawio.png)`
- 圖數量參考：FFmpeg 9 章用 19 張、TPU v6e 13 章用 28 張——平均一章 2~3 張

### 配色準則（白底）

| 顏色 | fillColor / strokeColor | 用途 |
|---|---|---|
| 藍 | `#dae8fc` / `#6c8ebf` | 主要資料結構、I/O、buffer |
| 綠 | `#d5e8d4` / `#82b366` | 安全狀態、ROM、未壓縮資料 |
| 紫 | `#e1d5e7` / `#9673a6` | API 入口、函式 |
| 黃 | `#fff2cc` / `#d6b656` | 中介層、設定、I/O |
| 橘 | `#ffe6cc` / `#d79b00` | 特殊用、warning |
| 紅 | `#f8cecc` / `#b85450` | 壓縮資料、CPU 核心、中斷、重點 |
| 灰虛線 | `#f5f5f5` / `#999999` | 鏡像、未使用、可選 |

### drawio CLI 渲染

```bash
DRAWIO="/Applications/draw.io.app/Contents/MacOS/draw.io"
cd drawio
for f in *.drawio; do
  "$DRAWIO" -x -f png -b 10 -s 2 -o "${f}.png" "$f"
done
```

`-s 2` 是 2× scale，PNG 解析度高一點 PDF 列印才不糊。

### 線段走線：全部用 orthogonal，禁止任何斜線

**所有 edge 一律加 `edgeStyle=orthogonalEdgeStyle`。** 不要任何斜線——即使「短斜線看起來還好」也不行。

理由：

1. **塊圖閱讀友善**：人腦看橫直線比斜線快、不易誤判方向
2. **大小變動 robust**：未來改 block 寬高，orthogonal 自動重排成 L 形；斜線會偏移、可能撞到其他 block
3. **配色一致**：跨列、同列、轉折、直下 — 統一風格，不會「有些斜有些直」混雜

判定不再分 case：

- 同列同欄、center 對齊 → orthogonal 自動畫成直橫線（**結果跟直線一樣，但 style 統一**）
- 跨列／跨欄、center 不齊 → orthogonal 自動畫成 L 形或 Z 形
- 斜的 → 永不存在

drawio 改法：每條 edge 的 style 加三段：

```xml
<mxCell id="lm_e5" 
        style="edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;
               exitX=0.5;exitY=1;entryX=0.5;entryY=0;
               endArrow=classic;strokeColor=#666;strokeWidth=2;" 
        edge="1" parent="1" source="lm_ch4" target="lm_ch5">
  <mxGeometry relative="1" as="geometry">
    <Array as="points">
      <mxPoint x="520" y="250" />   <!-- 走列間 gap -->
      <mxPoint x="120" y="250" />
    </Array>
  </mxGeometry>
</mxCell>
```

三件事：

| Style key | 用途 |
|---|---|
| `edgeStyle=orthogonalEdgeStyle` | 強制橫直 90° 轉角，不走斜線 |
| `exitX/exitY` + `entryX/entryY` | 指定從 source/target 哪個邊出/入（0.5,1 = 底中、0.5,0 = 上中） |
| `<Array as="points">` waypoint | 強制路徑經過某些座標。**走 row 之間 gap、column 之間 gap** |

**waypoint 座標選擇**：找 row 之間的 y gap（建議 ≥ 30~40 px 寬），把橫向走線放那裡，就不會跟 block 重疊。

#### 配套規則 1：轉折點到箭頭至少 20 px

waypoint 不能太靠近 target — 否則箭頭擠在轉折處、看不出方向。**bend → 箭頭末端的線段至少 20 px**（PNG 渲染後仍清楚）。

實作：row gap 設 ≥ 30 px、waypoint y 偏向 source 端、留給 target 進入段足夠長度：

```
Source row bottom y = 250
Target row top    y = 290        ← gap = 40 px
Waypoint (橫向)   y = 265        ← 偏 source 端 15 px
Bend → arrow 距離  = 290 - 265 = 25 px ✓
```

#### 配套規則 2：block 大小要裝得下文字

drawio 的 `whiteSpace=wrap` 不會自動加大 block — 文字超出就被裁切。**手動量文字行數**：

| 文字行數 | 建議高度（fontSize 11） |
|---|---|
| 2 行（標題 + 1 行） | 60 |
| 3 行 | 70 |
| 4 行 | 80 |

寬度看最長那行：12-char 中文約 100 px、英文約 80 px、長 identifier（如 `METHOD/THREAD/CTHREAD`）約 140 px。

範本見：

| 範例 | 用途 |
|---|---|
| [SystemC drawio/ch00_learning_map.drawio](/Volumes/4T_OFFICE/_Claude/SystemC/drawio/ch00_learning_map.drawio) `lm_e5` / `lm_e9` / `lm_e12` | 跨列回繞 orthogonal + waypoint 走 row gap |
| 同檔 `lm_e1` ~ `lm_e14` 全部 | 即使是「同列短橫線」也加 `edgeStyle=orthogonalEdgeStyle` 統一風格 |
| 同檔 `lm_e2` (1→2 跨列) | exitX/entryX 不對齊時用 `exitX=1; exitY=0.5` + `entryX=0.5; entryY=0` 走 right→down 的 L 形 |
| 同檔 `lm_ch6`（4 行 → 80 高） / `lm_ch3` / `lm_ch7`（長英文 → 140 寬） | block 加大裝下文字 |
| [drawio/ch01_build_flow.drawio](/Volumes/4T_OFFICE/_Claude/SystemC/drawio/ch01_build_flow.drawio) `bf_es1` / `bf_es2` | 多輸入 source → 同 target，exitX/entryX 對齊讓線直下、不會斜 |
| 同檔 `bf_es3` / `bf_es4` | x 不能對齊時走「列間 gap 的橫向 lane」收斂到 target 上邊 |
| 同檔 `bf_e8`（lib→link 跨 phase dashed） | 跨大段 distance、避開中間 label 的 orthogonal 走線 |

### drawio 已知踩坑

| 問題 | 解法 |
|---|---|
| Cell value 內含 `<b>` 等 HTML 標籤 → 116-byte 空 PNG | 改純文字，粗體用 `fontStyle=1` |
| `id="filter"` 是 drawio 保留字 → `Export failed` | 改名 `filt_node` 等。其他可疑保留字：`arrow`、`box`、`group`、`edge` |
| 頁面背景沒設白底 → PNG 變黑底 | `<mxGraphModel ... background="#FFFFFF">` 一定要加 |
| 同檔多個 cell 用同 ID → silently 壞掉 | 每個 ID 全域唯一 |
| value 換行 | 用 `&#10;`（不是真的 `\n`）；`&` 寫成 `&amp;` |
| 跨列線 / 跨欄長線壓到中間 block | 用 `edgeStyle=orthogonalEdgeStyle` + waypoint 走 row gap，見上「線段走線原則」 |

---

## 7. 階段 6 — 數學公式：LaTeX → PNG（optional）

TPU v6e 用了 11 條公式，HW / ML 性質的書都需要。**不依賴 pdflatex**，用 matplotlib mathtext，原因：環境乾淨、build 快。

### 加新公式的流程

1. 編輯 `scripts/render_eq.py` 的 `EQUATIONS` list，加一筆 `(name, latex, where)`
2. MD 裡用 `![alt](../eq/<name>.png)` 引用
3. 跑 `bash scripts/build_pdf.sh`——`[0/4]` 步驟會自動重渲染所有公式

### matplotlib mathtext 不支援的 LaTeX

| 寫不出來 | 替代方案 |
|---|---|
| `\begin{cases} ... \end{cases}` | 用 `\min`/`\max`；或拆成兩個 PNG |
| `\bm{}`、`\boldsymbol{}` | `\mathbf{}` 或省略 |
| `\text{...}` | `\mathrm{...}` |
| 多行對齊 (`align`) | 一律單行；要分段拆多個 PNG |
| 公式裡放中文 | 不要——外面用 markdown 描述 |

支援得好的：`\sum_{k=0}^{N-1}`、`\dfrac{a}{b}`、上下標、Greek `\gamma \sigma \mu \beta`、`\mathrm{round}`、`\min`/`\max`、`\approx`、`\ge`、`'`(prime)。

### 函數曲線圖（不只公式）

ReLU/ReLU6 那種 piecewise-linear 函數適合畫圖。`render_eq.py` 裡 `render_relu_curves()` 用 numpy + matplotlib 畫雙 panel 圖，跟公式 PNG 一起放 `eq/`。要加新曲線圖就照這個 pattern 寫一個 helper function，在 `main()` 裡呼叫。

---

## 8. 階段 7 — Build pipeline

### 一條龍

```bash
bash scripts/build_pdf.sh
```

四階段（[0/4] ~ [3/4]，外加 [4/4]）：

| 階段 | 工具 | 輸入 → 輸出 |
|---|---|---|
| 0 | `render_eq.py` | LaTeX list → `eq/*.png` |
| 1 | `draw.io -x -f png` | `drawio/*.drawio` → `drawio/*.drawio.png` |
| 2 | bash for-loop | `md/0[0-9]_*.md` 串接，章節間插 `\newpage` → `*_combined.md` |
| 3 | `md2html.py` | combined.md → HTML（內嵌 CSS、Pygments、TOC） |
| 4 | Chrome headless | HTML → `pdf/*.pdf`（A4、PingFang TC、頁碼/章名頁首） |

最終 PDF 輸出在 **`pdf/`** 子目錄；中間產物（`*_combined.md`、`*.html`）留在 `md/`。

### 手動 build（不用 build_pdf.sh）

```bash
HOME_DIR=/Volumes/4T_OFFICE/_Claude/<專案>
cd "$HOME_DIR/md"
MAIN=<專案>_textbook

# 合併章節
for f in 0[0-9]_*.md 1[0-9]_*.md; do
  cat "$f"; printf '\n\n\\newpage\n\n'
done > "${MAIN}_combined.md"

# MD → HTML
/tmp/mdpdf_venv/bin/python "$HOME_DIR/scripts/md2html.py" \
  "${MAIN}_combined.md" "${MAIN}.html"

# HTML → PDF（輸出到 pdf/）
mkdir -p "$HOME_DIR/pdf"
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
  --headless=new --disable-gpu --no-pdf-header-footer \
  --print-to-pdf="$HOME_DIR/pdf/${MAIN}.pdf" \
  --virtual-time-budget=30000 --run-all-compositor-stages-before-draw \
  "file://$(pwd)/${MAIN}.html"
```

`--virtual-time-budget=30000` 給足 30s 讓 Chrome 等所有圖載入；`--run-all-compositor-stages-before-draw` 確保最後一幀完整 render。**兩個都不能省**。

### PDF 規格

#### 頁面樣式

- A4，邊距 20/16/18/16mm
- PingFang TC 內文 10pt / 行高 1.6
- H1 20pt → H4 11pt
- Pygments `friendly` 主題語法高亮

#### PDF 開頭固定佈局（兩層 TOC）

| 頁 | 區塊 | `@page` rule | 頁首 |
|---|---|---|---|
| 1 | **封面**（書名 + 副標 + 選用架構圖） | `cover` | (無) |
| 2~3 | **章節速覽** chapter overview（Part 分組 + 章名一行） | `ch_overview` | 右「章節速覽」+ 頁碼 |
| 4~N | **完整目錄** full TOC（H1/H2/H3 三層） | `toc` | 右「完整目錄」+ 頁碼 |
| N+1 ~ | **各章** | (default) | 左「第 N 章」+ 右 章名 + 頁碼 |

兩層 TOC 不是 optional ── **這是預設 PDF 規格**：

- **章節速覽**只列 Part 分組與章名、每章一行、2-3 頁可看完——找「我要去哪個 Part 哪一章」用
- **完整目錄**含每章內所有 H2/H3——找「特定主題在哪裡」用
- 章節速覽 H1 用 `id="ch-overview"` 當錨點，給章節 nav link 跳回

#### 章間 nav（每章自動）

每章首末各一個「← 回章節速覽」link 框：

- **章首**：`H1` → `← 回章節速覽` (右上小框) → `> 上一章：[...]`（既有 blockquote）
- **章末**：`> 下一章 → [...]`（既有 blockquote）→ `← 回章節速覽` (右下小框)

兩個 link 都到 `#ch-overview` 錨點。

**自動化原則**：章節速覽 + 章間 nav 都靠 md2html.py post-process 注入，**章節 MD 檔不放 nav**——章節 MD 只保留 `> 上一章 / 下一章` blockquote。增刪章節、改 Part 分組只改 md2html.py 的 `PARTS` list，53 個 MD 檔不必碰。

#### 字串設定（@page header 防漏）

`.cover h1, .toc h1, .ch-overview h1` 三個 H1 都必須設 `string-set: chnum "", chapter ""`，否則它們的標題（「目錄」/「章節速覽」）會把章名 string 蓋掉、後面章節頁首抓不到字。

實作完整程式碼與 CSS 在 §16，或直接複製 [SystemC/scripts/md2html.py](/Volumes/4T_OFFICE/_Claude/SystemC/scripts/md2html.py)。

#### 教材規模對照

| 教材 | 章 | MD 行 | PDF | 模式 |
|---|---|---|---|---|
| FFmpeg | 9 | ~3000 | ~80 頁 | 單 TOC（§9） |
| TPU v6e | 17 | 4161 | 14 MB / 111 頁 | 單 TOC（§9） |
| **SystemC** | **53 / 7 Part** | **20157** | **18 MB / 385 頁** | **大書模式：兩層 TOC + 章間 nav（§16）** |

50 章左右是單 TOC 與大書模式的分水嶺。詳見 §16。

---

## 9. md2html.py 的關鍵 CSS

| Selector | 規則 | 用途 |
|---|---|---|
| `body` | PingFang TC 10pt | 內文字型 |
| `h1` | 20pt + `break-before: page` | 每章自動新頁 |
| `h1 .chnum` / `.chname` | `string-set` | 抓「第 N 章」+ 章名給 `@page` 頁首 |
| `code` | JetBrains Mono 8.5pt + 灰底 | 內聯 code |
| `pre code` | 8pt + Pygments friendly | code block |
| `img` | `max-width: 70%` + 圓角邊框 | 大圖（drawio） |
| `img[src*="/eq/"]` | `max-width: 45%` + 無邊框 | 公式 + 曲線圖 |
| `img[src*="relu_curves"]` | `max-width: 75%` | 雙 panel 函數圖 |
| `img[src*="model_scale"]` | `max-width: 100%` | label 多的散點圖（中國 LLM 加進去後 25 個點，不縮看不清） |
| `.page-break` | `break-after: page` | 對應 MD 的 `\newpage` |
| `.cover` / `.toc` | `@page cover` / `@page toc` | 封面/目錄不顯示頁首頁碼 |

公式圖寬度 **45%** 是試出來的：18% 看不清楚、35% 還是有點小、45% 剛好；70% 跟大型 drawio 一樣大太搶戲。**Dense plot（label 多的散點 / 多 panel 曲線）要 cascade 到 75%~100%**——`model_scale` 因 25+ 個 model label 必須吃滿頁寬。

換新專案要改 `scripts/md2html.py` 的：

1. `<title>` → `<專案> 教材`
2. 封面 `<h1>` → `<專案> <主題>原理與實作`
3. 副標 → 列出該專案的核心關鍵字
4. 想加封面圖：`<div class="cover-arch"><img src="../png/<arch>.png"></div>`

---

## 10. 目錄結構（新專案模板）

```
<專案>/
├── notebook.md              ← 本檔（從 TPU_v6e 複製過來改）
├── README.md                ← 專案說明
├── md/                      ← 教材 MD 來源
│   ├── 00_intro.md
│   ├── 01_*.md ... NN_*.md
│   ├── <專案>_textbook_combined.md  ← build 中間產物
│   └── <專案>_textbook.html         ← build 中間產物
├── pdf/                      ← 最終 PDF 產出目錄
│   └── <專案>_textbook.pdf
├── drawio/                  ← *.drawio + 同名 *.drawio.png
├── eq/                      ← LaTeX 公式 PNG（HW/ML 專案才需要）
├── png/                     ← 封面圖（選用）
├── scripts/
│   ├── md2html.py           ← MD → HTML（含全部 CSS）
│   ├── render_eq.py         ← LaTeX → PNG（HW/ML 才需要）
│   └── build_pdf.sh         ← 一條龍 build
├── samples/ or model/       ← 測試輸入，不 commit
└── <repo-name>/             ← git clone 的原始碼（subdir）
```

---

## 11. 環境依賴

```bash
# Python venv
python3 -m venv /tmp/mdpdf_venv
/tmp/mdpdf_venv/bin/pip install markdown pygments matplotlib numpy pypdf pillow

# draw.io Desktop（提供 CLI）
brew install --cask drawio
```

Chrome 路徑在 `build_pdf.sh` 裡寫死：`/Applications/Google Chrome.app/Contents/MacOS/Google Chrome`。

從現有專案複製 scripts：

```bash
mkdir -p /Volumes/4T_OFFICE/_Claude/<新專案>/scripts
cp /Volumes/4T_OFFICE/_Claude/TPU_v6e/scripts/{md2html.py,render_eq.py,build_pdf.sh} \
   /Volumes/4T_OFFICE/_Claude/<新專案>/scripts/
```

---

## 12. 踩坑紀錄（彙總）

| 問題 | 解法 |
|---|---|
| matplotlib mathtext 看到 `\begin{cases}` 直接 throw | 拆成 `\min/\max` 形式或多個 PNG |
| Chrome PDF 圖片沒載完就出檔 | `--virtual-time-budget=30000 --run-all-compositor-stages-before-draw` |
| draw.io CLI 對 `<b>HTML</b>` 內嵌標籤回 116-byte 空 PNG | cell value 純文字，粗體用 `fontStyle=1` |
| `id="filter"` 是 drawio 保留字 | 改名 `filt_node` 等 |
| drawio PNG 變黑底 | `<mxGraphModel background="#FFFFFF">` |
| 公式圖太大、佔半頁 | CSS `img[src*="/eq/"] { max-width: 45% }` |
| 中文字型在 PDF 變方塊 | `font-family: "PingFang TC", "Heiti TC", "Hiragino Sans GB", "Microsoft JhengHei"` 串好 fallback |
| C++ template instance 名字 | TC instance 用 `_BF16` / `_FP16` 後綴避免跟 INT8 default 撞（TPU 經驗） |
| MD 內 `\newpage` 沒處理 | `md2html.py` 把 `^\\newpage$` regex 替換成 `<div class="page-break"></div>` |
| 章節 H1 沒被 PDF 頁首抓到 | H1 必須是「第 N 章 — Title」格式，正則 `第\s*\d+\s*章` |

---

## 13. 對 Claude 的標準話術

**啟動新專案教材**：

> 「我要做 `<專案>` 走讀教材，repo 已經在 `/Volumes/4T_OFFICE/_Claude/<專案>/`，編譯能跑。請參考 `/Volumes/4T_OFFICE/_Claude/TPU_v6e/notebook.md` 的流程，按 9 章模板（或 bottom-up 模板）幫我寫教材。先列章節大綱讓我看，再依序寫。」

**重生 PDF**：

> 「重生 `<專案>` 教材 PDF」

**改某張 drawio**：

> 「`<專案>/drawio/<檔名>` 的 `<某條線/某個方塊>` 不對，改成 `<期望樣子>`」

**新增一章**：

> 「在 `<專案>` 的第 X 章與第 X+1 章中間插一章 `<章名>`，重點是 `<…>`，重編 PDF」

**加公式**：

> 「§X.Y 那個 `<公式描述>` 公式幫我用 LaTeX→PNG 渲染加進去」

**換封面**：

> 「`<專案>` 封面改成純文字（標題 + 副標），不放圖」  
> 或  
> 「`<專案>` 封面加架構圖：`png/<arch>.png`」

**調圖大小**：

> 「公式圖太大/太小，調成 X%」  
> （改 `md2html.py` 的 `img[src*="/eq/"]` max-width）

---

## 14. 速查表（cheat sheet）

```
新專案啟動: cp -r TPU_v6e/scripts <新專案>/scripts; 改 md2html.py title
寫新章:    md/NN_topic.md → 加 drawio 圖 → 加公式 → bash scripts/build_pdf.sh
加公式:    scripts/render_eq.py 加一筆 → MD 用 ![](../eq/x.png) → rebuild
加圖:      drawio/topic.drawio → MD 用 ![](../drawio/topic.drawio.png) → rebuild
改 CSS:    scripts/md2html.py → rebuild
查斷面:    pdftoppm -r 100 -f N -l M textbook.pdf out -png（render 第 N~M 頁）
踩坑:      §12（drawio 保留字、Chrome 載圖、mathtext 限制）
```

---

## 15. 範本檔案位置

### 單 TOC 模式（小於 ~25 章）

| 用途 | 路徑 |
|---|---|
| 本流程文件 | `/Volumes/4T_OFFICE/_Claude/TPU_v6e/notebook.md` |
| 範本 build_pdf.sh / md2html.py / render_eq.py | `/Volumes/4T_OFFICE/_Claude/TPU_v6e/scripts/` |
| 範本章節 MD（HW/bottom-up） | `/Volumes/4T_OFFICE/_Claude/TPU_v6e/md/` |
| 範本 drawio | `/Volumes/4T_OFFICE/_Claude/TPU_v6e/drawio/` |
| 範本 LaTeX 公式 | `/Volumes/4T_OFFICE/_Claude/TPU_v6e/eq/` |
| 範本 PDF | `/Volumes/4T_OFFICE/_Claude/TPU_v6e/md/TPU_v6e_textbook.pdf` |
| FFmpeg 教材（純軟體走讀範本） | `/Volumes/4T_OFFICE/_Claude/FFmpeg/textbook/` |
| 模擬器專用流程（NES/GB） | `/Volumes/4T_OFFICE/_Claude/EMULATOR/EMULATOR_TEXTBOOK_WORKFLOW.md` |

### 大書模式（25+ 章 / 多 Part）

| 用途 | 路徑 |
|---|---|
| 大書模式範本 md2html.py（含章節速覽 + 章間 nav） | `/Volumes/4T_OFFICE/_Claude/SystemC/scripts/md2html.py` |
| 範本 plan.md（多 Part 設計，53 章 / 7 Part） | `/Volumes/4T_OFFICE/_Claude/SystemC/plan.md` |
| 範本章節 MD | `/Volumes/4T_OFFICE/_Claude/SystemC/md/` |
| 範本 PDF（385 頁 / 7 Part） | `/Volumes/4T_OFFICE/_Claude/SystemC/pdf/SystemC_textbook.pdf` |

---

## 16. 大書模式：章節速覽 + 完整目錄 + 章間 nav

當教材 25+ 章 / 多 Part 時，單一 TOC 的 10+ 頁就讓人迷路。SystemC 教材（53 章 / 7 Part / 385 頁）的踩坑與解法。

### 16.1 PDF 開頭固定佈局（兩層 TOC）

| 頁 | 內容 | 頁首 |
|---|---|---|
| 1 | 封面 | (無) |
| 2-3 | **章節速覽**（chapter overview） | 章節速覽 |
| 4-22 | **完整目錄**（full TOC，含 H1/H2/H3） | 完整目錄 |
| 23+ | 各章 | 第 N 章 / 章名 |

- **章節速覽**：只列 Part 分組與章名，每章一行，2-3 頁可看完；給「我要找哪個 Part 哪一章」用
- **完整目錄**：H1/H2/H3 三層，包含每章內所有節；給「我要找特定主題在哪裡」用

### 16.2 章間 nav

每章自動 inject 兩個「← 回章節速覽」link：

- **章首**：`# 第 N 章 — Title` → `← 回章節速覽` (右上小框) → `> 上一章：[...]` (既有 blockquote)
- **章末**：`> 下一章 → [...]` (既有 blockquote) → `← 回章節速覽` (右下小框)

「← 回章節速覽」link 到 `#ch-overview`（章節速覽 H1 的 anchor）。

**設計要點**：所有自動化在 md2html.py 內，**完全不改章節 MD 檔**——之後增刪章節、改 Part 分組，只需改 md2html.py 的 `PARTS` list。

### 16.3 在 md2html.py 內實作（5 步）

#### (a) PARTS 配置

```python
PARTS = [
    ("Part I — 使用 SystemC", 0, 14),
    ("Part II — Library 內部走讀", 15, 24),
    ("Part III — SoC Bus 通訊協定", 25, 30),
    # ... (部別名, 起始章, 結束章)
]
```

#### (b) 從 body 抓 chapter H1（split_h1 改寫之前）

```python
ch_pattern = re.compile(
    r'<h1 id="([^"]+)">(第\s*(\d+)\s*章)\s*[—–-]\s*([^<]+)</h1>'
)
chapters = {}
for m in ch_pattern.finditer(body):
    anchor_id = m.group(1)
    ch_num = int(m.group(3))
    title_only = m.group(4).strip()
    chapters[ch_num] = (anchor_id, title_only)
```

**順序很重要**：要在 `split_h1` 把 H1 拆成 `<span>` 三段**之前**抓——拆完就 match 不到原始格式。

#### (c) 生成章節速覽 HTML

```python
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
```

#### (d) Inject 章首/章末 nav（split_h1 改寫之後）

```python
top_nav = '<p class="ch-nav-top"><a href="#ch-overview">← 回章節速覽</a></p>'
body = re.sub(
    r'(<h1 id="[^"]+"><span class="chnum">第\s*\d+\s*章</span>'
    r'<span class="chsep">[^<]*</span>'
    r'<span class="chname">[^<]+</span></h1>)',
    r'\1' + top_nav,
    body
)

bottom_nav = '<p class="ch-nav-bot"><a href="#ch-overview">← 回章節速覽</a></p>'
body = body.replace(
    '<div class="page-break"></div>',
    bottom_nav + '\n<div class="page-break"></div>'
)
body += '\n' + bottom_nav   # 最後一章在 body 結尾再 append 一次
```

#### (e) 組裝 final HTML

```python
html = f"""<!doctype html><html lang="zh-Hant"><head>...</head><body>
<div class="cover">...</div>
{overview_html}                                            ← 章節速覽
<div class="toc"><h1>完整目錄</h1>{toc}</div>               ← 完整目錄
{body}                                                     ← 各章
</body></html>"""
```

### 16.4 CSS 配套

#### `@page` 規則 — 三種特殊頁

```css
@page cover       { @top-left { content: ""; } @top-right { content: ""; } @bottom-center { content: ""; } }
@page ch_overview { @top-left { content: ""; } @top-right { content: "章節速覽"; } }
@page toc         { @top-left { content: ""; } @top-right { content: "完整目錄"; } }

.cover       { page: cover; }
.ch-overview { page: ch_overview; }
.toc         { page: toc; }
```

#### 別忘了清章節速覽的 page-header string

```css
.cover h1, .toc h1, .ch-overview h1 {
  break-before: auto;
  string-set: chnum "", chapter "";
}
```

否則章節速覽的 H1「章節速覽」會把 `chnum`/`chapter` string 蓋掉、後面章節頁首抓不到字。

#### 章節速覽 樣式

```css
.ch-overview {
  break-after: page;
  border: 1px solid #ddd;
  padding: 18px 24px;
  border-radius: 6px;
  background: #fbfbfb;
}
.ch-overview h1 { border: none; font-size: 22pt; text-align: center; margin: 0 0 0.6em 0; }
.ch-overview h2 { margin: 0.9em 0 0.3em; border: none; font-size: 13pt; color: #1554a4; }
.ch-overview ul { list-style: none; padding-left: 0.4em; }
.ch-overview li { margin: 2px 0; font-size: 10pt; break-inside: avoid; }
.ch-overview a  { color: #222; }
```

#### 章首/章末 nav

```css
.ch-nav-top, .ch-nav-bot { text-align: right; font-size: 9pt; }
.ch-nav-top { margin: 0.3em 0 0.8em; }
.ch-nav-bot { margin: 1.6em 0 0.4em; }
.ch-nav-top a, .ch-nav-bot a {
  color: #666; text-decoration: none;
  border: 1px solid #ccc; padding: 2px 8px;
  border-radius: 3px; background: #fafafa;
}
```

### 16.5 多 Part plan.md 結構

大書通常分多 Part。plan.md 的 §2 outline 用 §2.A / §2.B / §2.C ... 分 Part 列章節：

```
## 2. 章節大綱
### 2.A Part I — <Part 1 主題>
| # | 章名 | 主軸 | drawio | 公式 |
| 0 | ... |
...
### 2.B Part II — <Part 2 主題>
...
```

每 Part 開頭加「讀者轉變」段落，明確：
- Part 之前該讀完的前置章節
- 為什麼要學這個 Part（用例 / 動機）
- 與其他 Part 的關聯

範本：[SystemC plan.md](/Volumes/4T_OFFICE/_Claude/SystemC/plan.md)（7 Part / 53 章）。

### 16.6 規模差異與調整

兩層 TOC + 章間 nav 是**所有規模通用**的預設規格（§8）。不同規模僅作微調：

```
規模 < 15 章 / 單 Part   → PARTS 只有一個 entry：[("Part I — <主題>", 0, N-1)]
                            章節速覽很短（半頁）但仍保留統一視覺
規模 15-50 章 / 多 Part  → 標準 SystemC/scripts/md2html.py 預設配置
規模 50+ 章              → 加 toc_depth='1-2' 把完整目錄縮短（去 H3）
                          或 .ch-overview ul column-count: 2 章節速覽改雙欄
```

舊單 TOC 模式（TPU_v6e/scripts/md2html.py）仍可用——但新專案建議直接走兩層 TOC，不必之後再切換。複製範本是 `/Volumes/4T_OFFICE/_Claude/SystemC/scripts/md2html.py`。

### 16.7 大書模式踩坑

| 問題 | 解 |
|---|---|
| 章節速覽 page header 顯示前面章節的字 | `.ch-overview h1` 要加 `string-set: chnum "", chapter ""` |
| 章節速覽佔太多頁 | 把 `column-count: 2` 加進 `.ch-overview ul` 改雙欄 |
| 章首 nav 接在 H1 後但渲染順序怪 | 用 regex 確認 `<p class="ch-nav-top">` 排在 `</h1>` 後面、不要包進 H1 內 |
| 最後一章末沒 nav | body 字串 `+= '\n' + bottom_nav` 解 |
| 完整目錄太長（20+ 頁） | toc_depth 改 '1-2'（去掉 H3） |
| 抓不到 chapter id | `ch_pattern` 必須在 `split_h1` 前用，否則 H1 已被改成 `<span>` 三段 |
| Chrome 漏 render 章節速覽錨點 | `--virtual-time-budget=30000` 給夠（複雜頁 Chrome 需多時間） |
