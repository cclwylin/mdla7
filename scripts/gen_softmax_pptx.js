// gen_softmax_pptx.js — Softmax MB Chain → L1Mesh Stress Test presentation
"use strict";
const pptxgen = require("/usr/local/lib/node_modules/pptxgenjs");

const OUT = "/Volumes/4T_OFFICE/_Claude/MDLA7_Claude/softmax.pptx";

// ── Palette (Midnight Executive) ──────────────────────────────────────────────
const C = {
  navy:    "1E2761",
  mid:     "2D3E8A",
  ice:     "CADCFC",
  iceD:    "A8C4F8",
  white:   "FFFFFF",
  offwhite:"F7F9FF",
  accent:  "4FC3F7",   // cyan highlight
  accentG: "00E5A0",   // green for OK
  warn:    "F9A825",   // amber for bottleneck
  gray:    "64748B",
  lightG:  "E8EEF7",
  code:    "1A1F36",
};

const FONT_H = "Calibri";
const FONT_B = "Calibri";

const pres = new pptxgen();
pres.layout  = "LAYOUT_16x9";
pres.title   = "Softmax MB Chain → L1Mesh Stress Test";
pres.author  = "MDLA7";
pres.subject = "Hardware Architecture";

// ── helpers ───────────────────────────────────────────────────────────────────

function titleBar(slide, text, sub) {
  // dark header band
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: 10, h: 1.05,
    fill: { color: C.navy }, line: { color: C.navy },
  });
  slide.addText(text, {
    x: 0.35, y: 0.08, w: 9, h: 0.58,
    fontSize: 22, bold: true, color: C.white,
    fontFace: FONT_H, margin: 0,
  });
  if (sub) {
    slide.addText(sub, {
      x: 0.35, y: 0.65, w: 9, h: 0.35,
      fontSize: 10, color: C.ice, fontFace: FONT_B, margin: 0,
    });
  }
}

function footerBar(slide, note) {
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 5.33, w: 10, h: 0.30,
    fill: { color: C.navy }, line: { color: C.navy },
  });
  slide.addText(note || "MDLA7 — Confidential", {
    x: 0.3, y: 5.34, w: 9.4, h: 0.25,
    fontSize: 7.5, color: C.ice, fontFace: FONT_B,
    align: "right", margin: 0,
  });
}

function card(slide, x, y, w, h, opts) {
  slide.addShape(pres.shapes.RECTANGLE, {
    x, y, w, h,
    fill: { color: opts.fill || C.white },
    line: { color: opts.border || C.iceD, width: 1 },
    shadow: { type: "outer", color: "000000", blur: 5, offset: 2, angle: 135, opacity: 0.10 },
  });
}

function dot(slide, x, y, color) {
  slide.addShape(pres.shapes.OVAL, {
    x, y, w: 0.18, h: 0.18,
    fill: { color }, line: { color },
  });
}

// ── Slide 1: Title ────────────────────────────────────────────────────────────
{
  const s = pres.addSlide();
  s.background = { color: C.navy };

  // diagonal accent shape
  s.addShape(pres.shapes.RECTANGLE, {
    x: 6.5, y: 0, w: 4, h: 5.63,
    fill: { color: C.mid, transparency: 40 }, line: { color: C.mid, transparency: 40 },
  });
  s.addShape(pres.shapes.RECTANGLE, {
    x: 7.8, y: 0, w: 2.2, h: 5.63,
    fill: { color: "253070", transparency: 20 }, line: { color: "253070", transparency: 20 },
  });

  s.addText("Softmax MB Chain", {
    x: 0.5, y: 1.1, w: 7, h: 0.85,
    fontSize: 38, bold: true, color: C.white,
    fontFace: FONT_H, margin: 0,
  });
  s.addText("L1Mesh Stress Test", {
    x: 0.5, y: 1.92, w: 7, h: 0.72,
    fontSize: 32, bold: false, color: C.accent,
    fontFace: FONT_H, margin: 0,
  });

  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 2.78, w: 2.2, h: 0.04,
    fill: { color: C.accentG }, line: { color: C.accentG },
  });

  s.addText([
    { text: "目標：", options: { bold: true, color: C.ice } },
    { text: "BMM + EWE + POOL 全程留在 L1Mesh，壓測 bank conflict / NoC / 仲裁能力", options: { color: C.white } },
  ], {
    x: 0.5, y: 2.95, w: 8.5, h: 0.6,
    fontSize: 12, fontFace: FONT_B,
  });

  s.addText([
    { text: "模型  ", options: { bold: true, color: C.iceD } },
    { text: "bmm_softmax_bmm_2.5ms_1g_int8.tflite   ", options: { color: C.white } },
    { text: "B=1  H=32  S=2048  D=128", options: { bold: true, color: C.accent } },
  ], {
    x: 0.5, y: 3.65, w: 8.5, h: 0.5,
    fontSize: 11, fontFace: FONT_B,
  });

  s.addText("MDLA7  ·  Architecture Team", {
    x: 0.5, y: 5.0, w: 5, h: 0.3,
    fontSize: 9, color: C.gray, fontFace: FONT_B, margin: 0,
  });
}

// ── Slide 2: 問題 — 為什麼串不起來 ────────────────────────────────────────────
{
  const s = pres.addSlide();
  s.background = { color: C.offwhite };
  titleBar(s, "為什麼現在串不起來？", "三個根本原因");

  const reasons = [
    {
      num: "01", color: C.warn,
      title: "Score tensor 比 L1 大 44×",
      body: "scores [1,32,2048,2048] INT8 = 134 MB\nL1Mesh SRAM = 3 MB\n→ 必然 spill DRAM",
    },
    {
      num: "02", color: "#E53935",
      title: "Softmax 需要 global reduce",
      body: "max(x_row) 和 Σexp(x_j) 都需\n看完整列 2048 元素才能算\n→ pipeline 中間斷掉",
    },
    {
      num: "03", color: C.mid,
      title: "Engine 異構，hand-off = DRAM",
      body: "CONV → Requant → L1_Manager\n→ EWE → L1_Manager → BMM2\n跨 engine 無 register forwarding",
    },
  ];

  reasons.forEach((r, i) => {
    const x = 0.3 + i * 3.22;
    card(s, x, 1.22, 3.05, 3.5, { fill: C.white, border: C.iceD });
    s.addShape(pres.shapes.RECTANGLE, {
      x, y: 1.22, w: 3.05, h: 0.42,
      fill: { color: r.color }, line: { color: r.color },
    });
    s.addText(r.num, {
      x: x + 0.12, y: 1.24, w: 0.55, h: 0.38,
      fontSize: 18, bold: true, color: C.white, fontFace: FONT_H, margin: 0,
    });
    s.addText(r.title, {
      x: x + 0.12, y: 1.78, w: 2.8, h: 0.55,
      fontSize: 12, bold: true, color: C.navy, fontFace: FONT_H, margin: 0,
    });
    s.addText(r.body, {
      x: x + 0.12, y: 2.38, w: 2.82, h: 2.1,
      fontSize: 10.5, color: "2D3748", fontFace: FONT_B,
      lineSpacingMultiple: 1.3, margin: 0,
    });
  });

  // arrow / result
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.3, y: 4.9, w: 9.4, h: 0.35,
    fill: { color: C.navy }, line: { color: C.navy },
  });
  s.addText("結果：BMM → [134MB DRAM] → Softmax → [134MB DRAM] → BMM2   共 2 次 DRAM round-trip", {
    x: 0.45, y: 4.91, w: 9.1, h: 0.3,
    fontSize: 9.5, bold: true, color: C.accent, fontFace: FONT_B, margin: 0,
  });

  footerBar(s, "Softmax MB Chain — L1Mesh Stress Test");
}

// ── Slide 3: 解法 — Online Softmax Tiling ─────────────────────────────────────
{
  const s = pres.addSlide();
  s.background = { color: C.offwhite };
  titleBar(s, "解法：Online Softmax Tiling（Flash Attention 風格）",
           "把 global reduce 變成 running scalar，全程留在 L1");

  // left: algorithm phases
  card(s, 0.25, 1.15, 5.55, 3.95, { fill: C.code, border: C.code });

  const codeLines = [
    { text: "for each ", opts: { color: C.accent } },
    { text: "Q_tile", opts: { color: C.accentG, bold: true } },
    { text: " [1, H, T_q, D]:", opts: { color: C.white } },
    { text: "\n    m=-∞  s=0  O=0", opts: { color: C.iceD } },
    { text: "\n    for each ", opts: { color: C.accent } },
    { text: "K_tile / V_tile", opts: { color: C.accentG, bold: true } },
    { text: ":", opts: { color: C.white } },
    { text: "\n\n        ── Phase A: BMM1 ──", opts: { color: C.warn, bold: true } },
    { text: "\n        score_tile = CONV(Q, K^T)       ", opts: { color: C.white } },
    { text: "← L1", opts: { color: C.accentG } },
    { text: "\n\n        ── Phase B: EWE+POOL ──", opts: { color: C.warn, bold: true } },
    { text: "\n        m_new = max(m, POOL_max(score))", opts: { color: C.white } },
    { text: "\n        rescale = EWE_exp(m − m_new)", opts: { color: C.white } },
    { text: "\n        e_tile = EWE_exp(score − m_new)", opts: { color: C.white } },
    { text: "\n        s = s×rescale + POOL_sum(e)", opts: { color: C.white } },
    { text: "\n\n        ── Phase C: BMM2 ──", opts: { color: C.warn, bold: true } },
    { text: "\n        O = O×rescale + CONV(e, V)", opts: { color: C.white } },
    { text: "  ← L1", opts: { color: C.accentG } },
    { text: "\n\n    output_tile = EWE(O / s) → ", opts: { color: C.iceD } },
    { text: "flush DRAM ✓", opts: { color: C.accent, bold: true } },
  ];

  s.addText(codeLines.map(l => ({ text: l.text, options: l.opts })), {
    x: 0.38, y: 1.22, w: 5.35, h: 3.82,
    fontSize: 8.5, fontFace: "Courier New", margin: 0,
    lineSpacingMultiple: 1.25,
  });

  // right: L1 budget table
  card(s, 6.0, 1.15, 3.72, 3.95, { fill: C.white, border: C.iceD });

  s.addText("L1 佔用分析", {
    x: 6.12, y: 1.22, w: 3.48, h: 0.35,
    fontSize: 11, bold: true, color: C.navy, fontFace: FONT_H, margin: 0,
  });
  s.addText("T_q = T_k = 64  ·  B=1 H=32 D=128  INT8", {
    x: 6.12, y: 1.55, w: 3.48, h: 0.25,
    fontSize: 8, color: C.gray, fontFace: FONT_B, margin: 0,
  });

  const rows = [
    [{ text: "Buffer", options: { bold: true, color: C.white } },
     { text: "Size", options: { bold: true, color: C.white } }],
    ["Q_tile",          "262 KB"],
    ["K_tile",          "262 KB"],
    ["V_tile",          "262 KB"],
    ["score / e_tile",  "131 KB"],
    ["O (accumulator)", "262 KB"],
    ["m, s (stats)",    "4 KB"],
    [{ text: "合計", options: { bold: true } },
     { text: "~1.2 MB ✓", options: { bold: true, color: "#1B7B4E" } }],
  ];

  s.addTable(rows, {
    x: 6.12, y: 1.82, w: 3.5, h: 2.8,
    fontSize: 9.5, fontFace: FONT_B,
    border: { pt: 0.5, color: C.iceD },
    colW: [2.1, 1.4],
    rowH: 0.33,
    fill: { color: C.white },
  });

  // table header fill via separate shape
  s.addShape(pres.shapes.RECTANGLE, {
    x: 6.12, y: 1.82, w: 3.5, h: 0.33,
    fill: { color: C.navy }, line: { color: C.navy },
  });
  s.addText("Buffer", {
    x: 6.15, y: 1.83, w: 2.08, h: 0.30,
    fontSize: 9, bold: true, color: C.white, fontFace: FONT_B, margin: 0,
  });
  s.addText("Size", {
    x: 8.23, y: 1.83, w: 1.38, h: 0.30,
    fontSize: 9, bold: true, color: C.white, fontFace: FONT_B, margin: 0,
  });

  // green bar: L1 = 3MB
  s.addShape(pres.shapes.RECTANGLE, {
    x: 6.12, y: 4.72, w: 3.5, h: 0.28,
    fill: { color: "1B7B4E" }, line: { color: "1B7B4E" },
  });
  s.addText("L1Mesh 3 MB 上限  →  1.2 MB 使用，有餘裕可調 tile size", {
    x: 6.15, y: 4.73, w: 3.44, h: 0.24,
    fontSize: 8, bold: true, color: C.white, fontFace: FONT_B, margin: 0,
  });

  footerBar(s, "Softmax MB Chain — L1Mesh Stress Test");
}

// ── Slide 4: L1Mesh 壓力分析 ──────────────────────────────────────────────────
{
  const s = pres.addSlide();
  s.background = { color: C.offwhite };
  titleBar(s, "L1Mesh 壓力分析", "Tile loop 的三個 Phase — 同時打到 L1Mesh 的各個面向");

  // Phase boxes
  const phases = [
    {
      label: "Phase A / C", sub: "BMM（CONV 主導）",
      color: C.mid, x: 0.22,
      rows: [
        ["Engine", "方向", "路數", "路徑"],
        ["CONV", "ACT_R (Q/e)", "32", "專線 bypass"],
        ["CONV", "WGT_R (K/V)", "32", "專線 bypass"],
        ["Requant", "W (score)", "8",  "L1_Manager"],
        ["UDMA", "W (prefetch)", "16", "L1_Manager"],
      ],
      note: "W 需求 24 → 上限 16W\nRequant + UDMA 競爭",
      noteColor: C.warn,
    },
    {
      label: "Phase B", sub: "EWE + POOL（softmax stats）",
      color: "#0277BD", x: 5.1,
      rows: [
        ["Engine", "方向", "路數", "路徑"],
        ["POOL", "R (score/e)", "16", "L1_Manager"],
        ["POOL", "W (m, s)",    "8",  "L1_Manager"],
        ["EWE",  "R (O, score)","16", "L1_Manager"],
        ["EWE",  "W (e, O)",    "8",  "L1_Manager"],
      ],
      note: "R 需求 32 → 上限 16R\nEWE + POOL round-robin",
      noteColor: "#0277BD",
    },
  ];

  phases.forEach(p => {
    card(s, p.x, 1.15, 4.72, 4.02, { fill: C.white, border: C.iceD });
    s.addShape(pres.shapes.RECTANGLE, {
      x: p.x, y: 1.15, w: 4.72, h: 0.44,
      fill: { color: p.color }, line: { color: p.color },
    });
    s.addText(p.label, {
      x: p.x + 0.1, y: 1.17, w: 2.5, h: 0.22,
      fontSize: 11, bold: true, color: C.white, fontFace: FONT_H, margin: 0,
    });
    s.addText(p.sub, {
      x: p.x + 0.1, y: 1.37, w: 4.5, h: 0.2,
      fontSize: 8.5, color: C.ice, fontFace: FONT_B, margin: 0,
    });

    // table rows (skip header row, draw manually)
    const tData = p.rows.slice(1).map(r => r.map(c => c));
    s.addTable(tData, {
      x: p.x + 0.1, y: 1.68, w: 4.52, h: 1.95,
      fontSize: 9, fontFace: FONT_B,
      border: { pt: 0.4, color: C.iceD },
      colW: [1.05, 1.3, 0.7, 1.47],
      rowH: 0.38,
      fill: { color: C.white },
    });
    // header row
    s.addShape(pres.shapes.RECTANGLE, {
      x: p.x + 0.1, y: 1.68, w: 4.52, h: 0.32,
      fill: { color: C.lightG }, line: { color: C.iceD },
    });
    ["Engine", "方向", "路數", "路徑"].forEach((h, ci) => {
      const cxs = [0, 1.05, 2.35, 3.05];
      s.addText(h, {
        x: p.x + 0.15 + cxs[ci], y: 1.69, w: 1.0, h: 0.28,
        fontSize: 8.5, bold: true, color: C.navy, fontFace: FONT_B, margin: 0,
      });
    });

    // note
    s.addShape(pres.shapes.RECTANGLE, {
      x: p.x + 0.1, y: 3.68, w: 4.52, h: 0.56,
      fill: { color: p.noteColor, transparency: 88 },
      line: { color: p.noteColor, width: 1 },
    });
    s.addText(p.note, {
      x: p.x + 0.2, y: 3.70, w: 4.3, h: 0.5,
      fontSize: 9, bold: true, color: p.noteColor, fontFace: FONT_B, margin: 0,
    });
  });

  // engines total summary bar
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.22, y: 4.85, w: 9.56, h: 0.35,
    fill: { color: C.navy }, line: { color: C.navy },
  });
  s.addText(
    "Peak (BMM): 3 engines  ·  64 dedicated CONV R + 16W L1_Manager    |    " +
    "Phase B: 2 engines  ·  16R + 16W L1_Manager",
    {
      x: 0.35, y: 4.86, w: 9.3, h: 0.30,
      fontSize: 9.5, bold: true, color: C.accent, fontFace: FONT_B, margin: 0,
    }
  );

  footerBar(s, "Softmax MB Chain — L1Mesh Stress Test");
}

// ── Slide 5: 壓測面向 & Test Plan ─────────────────────────────────────────────
{
  const s = pres.addSlide();
  s.background = { color: C.offwhite };
  titleBar(s, "壓測面向 & 測試計畫", "三個 timing mode 分別觀測不同 L1Mesh 行為");

  // left: stress aspects
  const aspects = [
    {
      icon: "⚡", color: C.warn,
      title: "Bank Conflict",
      when: "BMM phase：CONV 64R + Requant/UDMA W 爭同一 bank",
      mode: "--l1-timing=conflict",
    },
    {
      icon: "🔀", color: "#7B1FA2",
      title: "NoC 路由競爭",
      when: "Phase A↔B 切換：CONV 專線與 L1_Manager lane 同時 in-flight",
      mode: "--l1-timing=mesh",
    },
    {
      icon: "⚖", color: "#0277BD",
      title: "L1_Manager 仲裁",
      when: "Phase B: EWE+POOL 搶 16R  /  Phase A: Requant+UDMA 搶 16W",
      mode: "--l1-timing=mesh",
    },
  ];

  aspects.forEach((a, i) => {
    const y = 1.18 + i * 1.22;
    card(s, 0.22, y, 5.0, 1.08, { fill: C.white, border: C.iceD });
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.22, y, w: 0.22, h: 1.08,
      fill: { color: a.color }, line: { color: a.color },
    });
    s.addText(a.title, {
      x: 0.55, y: y + 0.06, w: 3.5, h: 0.3,
      fontSize: 11, bold: true, color: C.navy, fontFace: FONT_H, margin: 0,
    });
    s.addText(a.when, {
      x: 0.55, y: y + 0.36, w: 4.55, h: 0.38,
      fontSize: 9, color: "2D3748", fontFace: FONT_B, margin: 0,
    });
    s.addShape(pres.shapes.RECTANGLE, {
      x: 3.5, y: y + 0.06, w: 1.6, h: 0.28,
      fill: { color: a.color, transparency: 85 },
      line: { color: a.color, width: 1 },
    });
    s.addText(a.mode, {
      x: 3.55, y: y + 0.07, w: 1.55, h: 0.24,
      fontSize: 7.5, bold: true, color: a.color, fontFace: "Courier New", margin: 0,
    });
  });

  // right: test commands
  card(s, 5.45, 1.18, 4.3, 3.65, { fill: C.code, border: C.code });
  s.addText("測試指令", {
    x: 5.6, y: 1.24, w: 4.0, h: 0.3,
    fontSize: 10, bold: true, color: C.accent, fontFace: FONT_H, margin: 0,
  });

  const cmds = [
    { label: "Fast  (BW baseline)", color: C.accentG,
      cmd: "./batch/run_systemc.py\n  --filter bmm --fast-only --rerun-all" },
    { label: "CX  (bank conflict)", color: C.warn,
      cmd: "./batch/run_systemc.py\n  --filter bmm --cx --fast-only --rerun-all" },
    { label: "Verilog  (closed-loop)", color: C.accent,
      cmd: "./batch/run_verilog.py\n  --filter bmm --rerun-all --timeout 180" },
  ];

  cmds.forEach((c, i) => {
    const cy = 1.62 + i * 0.95;
    s.addText(c.label, {
      x: 5.6, y: cy, w: 4.0, h: 0.22,
      fontSize: 8.5, bold: true, color: c.color, fontFace: FONT_B, margin: 0,
    });
    s.addText(c.cmd, {
      x: 5.6, y: cy + 0.24, w: 4.05, h: 0.48,
      fontSize: 8, color: C.white, fontFace: "Courier New", margin: 0,
      lineSpacingMultiple: 1.2,
    });
  });

  // bottom: param table
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.22, y: 4.55, w: 9.56, h: 0.28,
    fill: { color: C.navy }, line: { color: C.navy },
  });
  s.addText("結果 → 設計參數對應", {
    x: 0.35, y: 4.56, w: 9.2, h: 0.24,
    fontSize: 9, bold: true, color: C.ice, fontFace: FONT_H, margin: 0,
  });

  const paramRows = [
    ["BMM phase bank conflict 嚴重", "L1_Manager W lane 數 / Requant+UDMA 分時策略"],
    ["Phase B EWE stall 多", "EWE R lane 數（16 → 32）"],
    ["UDMA 預取與 Requant 衝突", "arbitration policy（round-robin → priority）"],
    ["NoC router 擁塞", "Router FIFO depth（目前 2 flits）"],
    ["Tile overhead 大", "T_q / T_k 往上調（64 → 128）"],
  ];
  s.addTable(paramRows, {
    x: 0.22, y: 4.85, w: 9.56, h: 0.75,
    fontSize: 8, fontFace: FONT_B,
    border: { pt: 0.3, color: C.iceD },
    colW: [4.0, 5.56],
    rowH: 0.15,
    fill: { color: C.white },
  });

  footerBar(s, "Softmax MB Chain — L1Mesh Stress Test");
}

// ── Slide 6: 實作 Roadmap ──────────────────────────────────────────────────────
{
  const s = pres.addSlide();
  s.background = { color: C.offwhite };
  titleBar(s, "實作 Roadmap", "三個 Step → 生 TFLite → Compile → 跑測試");

  const steps = [
    {
      num: "1", color: C.mid,
      title: "生成 TFLite",
      file: "gen_bmm25_int8_tflite.py  --tiled",
      items: [
        "確認 EWE RTL 有無 exp activation（或需 TNPS LUT）",
        "加 --tiled flag，展開 tile loop graph (T_q=T_k=64)",
        "輸出 bmm_softmax_bmm_2.5ms_1g_mb_int8.tflite",
      ],
    },
    {
      num: "2", color: "#0277BD",
      title: "compile_model.py lowering",
      file: "systemc/scripts/compile_model.py",
      items: [
        "辨識 BATCH_MATMUL → SOFTMAX → BATCH_MATMUL pattern",
        "Lower 成 tile loop (outer=Q, inner=K/V)",
        "loop body 插入 rescale step (running max 更新時)",
        "m, s, O 配置在 L1，不走 DRAM",
      ],
    },
    {
      num: "3", color: "#1B7B4E",
      title: "跑測試 & 調參",
      file: "batch/run_systemc.py + run_verilog.py",
      items: [
        "fast vs cx vs mesh cycle 差距",
        "識別主要 bottleneck",
        "提出 RTL 參數調整建議",
      ],
    },
  ];

  steps.forEach((st, i) => {
    const x = 0.22 + i * 3.28;
    card(s, x, 1.15, 3.1, 4.05, { fill: C.white, border: C.iceD });
    s.addShape(pres.shapes.RECTANGLE, {
      x, y: 1.15, w: 3.1, h: 0.52,
      fill: { color: st.color }, line: { color: st.color },
    });
    s.addText(`STEP ${st.num}`, {
      x: x + 0.12, y: 1.17, w: 0.75, h: 0.24,
      fontSize: 8.5, bold: true, color: C.ice, fontFace: FONT_B, margin: 0,
    });
    s.addText(st.title, {
      x: x + 0.12, y: 1.38, w: 2.85, h: 0.26,
      fontSize: 12, bold: true, color: C.white, fontFace: FONT_H, margin: 0,
    });
    s.addText(st.file, {
      x: x + 0.12, y: 1.75, w: 2.85, h: 0.28,
      fontSize: 7.5, color: st.color, fontFace: "Courier New", margin: 0,
    });
    s.addText(
      st.items.map((t, j) => ({
        text: `${["①","②","③","④"][j]}  ${t}`,
        options: { breakLine: j < st.items.length - 1, color: "2D3748" },
      })),
      {
        x: x + 0.12, y: 2.1, w: 2.86, h: 2.85,
        fontSize: 9.5, fontFace: FONT_B, lineSpacingMultiple: 1.5, margin: 0,
      }
    );
  });

  // connect arrows
  [3.32, 6.6].forEach(ax => {
    s.addShape(pres.shapes.LINE, {
      x: ax, y: 3.18, w: 0.25, h: 0,
      line: { color: C.navy, width: 1.5 },
    });
  });

  footerBar(s, "Softmax MB Chain — L1Mesh Stress Test");
}

// ── write ─────────────────────────────────────────────────────────────────────
pres.writeFile({ fileName: OUT }).then(() => {
  console.log("Written:", OUT);
}).catch(e => { console.error(e); process.exit(1); });
