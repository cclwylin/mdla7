# Profile HTML Format

This document describes the current MDLA7 profile HTML formats generated under
`batch/`.

There are two HTML report shapes:

| File | Generator | Purpose |
| --- | --- | --- |
| `batch/output/<model>.html` | `batch/run_model.py::_write_html_report` | Per-model layer profile, engine timeline, compile log, and bandwidth/cycle summary. |
| `batch/profile_mdla6_pattern.html` | `batch/gen_model_profile.py` via `batch/run_mdla6_pattern.py` | MDLA6-pattern index with CX baseline columns. |
| `batch/profile_hotspot.html` | `batch/gen_model_profile.py` via `batch/run_hotspot.py` | Hotspot-slice index without CX columns; sorted for L1 timing debug. |
| `batch/profile_ethz_v5.html` | `batch/gen_model_profile.py` via `batch/run_ethz_v5.py` | ETHZ_v5 corpus index without CX columns. |
| `batch/profile_ethz_v6.html` | `batch/gen_model_profile.py` via `batch/run_ethz_v6.py` | ETHZ_v6 corpus index without CX columns. |
| `batch/profile_mlperf.html` | `batch/gen_model_profile.py` via `batch/run_mlperf.py` | MLPerf Tiny corpus index without CX columns. |

All HTML files are self-contained snapshots with inline CSS/JS. They can be
opened directly with `file://`; the profile indexes can additionally refresh
from `output/` when served from the `batch/` directory over HTTP.

## Per-Model Profile HTML

Path pattern:

```text
batch/output/<model_stem>.html
```

Main data sources:

| Source | Usage |
| --- | --- |
| `batch/output/<model_stem>.profile.json` | Source of truth for simulated cycles, engine busy time, engine tasks, layer stats, verification status, and memory traffic. |
| Console compile log from `compile_model.py` | Parsed for compile-log rows, dtype display, and skipped compile layers. |

### Page Sections

The page is currently ordered as:

1. Header summary
2. Per-engine busy table
3. Interactive Gantt timeline
4. Per-layer profile table
5. Compile log table

### Header Summary

The header shows:

| Field | Meaning |
| --- | --- |
| `Model` | Model path relative to the repo when possible. |
| `Layers` | Total layer count plus PASS/FAIL counts. |
| `Sim time` | `total_cycles / 1.9e6` ms, with raw cycle count also shown. |
| `Util` | Average and peak engine utilization. |
| `DRAM r/w` | Total DRAM read/write bytes, displayed in MB. |
| `SRAM r/w` | Total SRAM read/write bytes, displayed in MB. |

Cycle-to-time convention:

```text
ms = cycles / 1.9e6
```

### Per-Engine Busy

Columns:

| Column | Meaning |
| --- | --- |
| `engine` | Engine name, for example `udma_r`, `udma_w`, `conv`, `requant`, `ewe`, `pool`. |
| `busy cycles` | Total busy cycles reported for that engine. |
| `utilization` | `busy_cycles / total_cycles * 100`. |

### Gantt Timeline

The Gantt is driven by embedded JSON:

```html
<script id="gantt-data" type="application/json">...</script>
```

Current JSON shape:

```json
{
  "engines": {
    "conv": {
      "busy": 123,
      "tasks": [
        [0, 100, "L0 conv", 1024]
      ]
    }
  },
  "layers": [
    {"id": 0, "op": "CONV_2D", "end": 100}
  ],
  "conv_wait": [
    [10, 20, "L0 conv input/tile", 4096]
  ],
  "total": 100
}
```

Task tuple format:

| Index | Meaning |
| --- | --- |
| `0` | Start cycle. |
| `1` | End cycle. |
| `2` | Optional label shown in tooltip. |
| `3` | Optional byte count shown in tooltip. |

Current engine colors:

| Engine | Color |
| --- | --- |
| `udma_r` | `#4287f5` |
| `udma_w` | `#7c4dff` |
| `conv` | `#e84545` |
| `requant` | `#f5a142` |
| `ewe` | `#3ec56e` |
| `pool` | `#a945e8` |

Important convention:

- `conv_wait` is synthetic. It is derived from UDMA read work that blocks a
  non-streamed `CONV`, `DEPTHWISE_CONV`, or `FC` layer.
- `conv_wait` is not displayed as a separate lane or table column. It is
  overlaid into the `conv` lane so blocked CONV time remains visible in the
  same visual band.
- UDMA read labels may include categories such as `params`, `input/tile`,
  `weight/tile`, or `read`.

Current interactions:

| Control | Behavior |
| --- | --- |
| `+` / `-` | Zoom in or out. |
| `Reset` | Reset timeline viewport. |
| `scope` input | Accepts op text, `L<id>`, or a range such as `3-6`. |
| Prev/next buttons | Jump between scope matches. |
| Drag on lanes | Zoom to selected range. |
| Alt-drag or right-drag | Pan. |
| Wheel on lanes | Zoom. |
| Wheel on axis | Scroll/pan. |
| Keyboard left/right | Pan. |
| Keyboard `+`, `-`, `0` | Zoom in, zoom out, reset. |

### Per-Layer Profile Table

Columns:

| Column | Meaning |
| --- | --- |
| `id` | Layer id. |
| `op` | Operation name. |
| `iH`, `iW`, `iC` | Input tensor shape. |
| `oH`, `oW`, `oC` | Output tensor shape. |
| `kH`, `kW` | Kernel size. |
| `sH`, `sW` | Stride. |
| `group` | Group count. |
| `tiles (HxOC)` | Tiling shape in output height by output channels. |
| `cyc/layer` | Actual simulated cycles for this layer. |
| `cyc/cum` | Actual cumulative cycles through this layer. |
| `ideal cyc/layer` | Compute-only ideal cycle estimate. |
| `ideal cyc/cum` | Running sum of ideal cycles. |
| `ideal util` | `ideal cyc/layer / cyc/layer * 100`. |
| `conv occupancy` | CONV engine busy fraction inside this layer window. Includes visible wait/stall time. |
| `DRAM r` | Layer DRAM read traffic, shown in KB. |
| `DRAM w` | Layer DRAM write traffic, shown in KB. Non-zero values are red/bold. |
| `SRAM r` | Layer SRAM read traffic, shown in KB. |
| `SRAM w` | Layer SRAM write traffic, shown in KB. |
| `verify` | PASS/FAIL status. |

Memory sizes in this table are displayed as:

```text
KB = bytes / 1024
```

### Ideal Cycle Calculation

The ideal cycle columns are compute-only estimates. They do not include DRAM,
SRAM, command overhead, dependency stalls, UDMA waits, or tiling overhead.

Current peak compute assumptions:

| Data type / path | MAC per cycle |
| --- | ---: |
| INT8x8 default | 16,384 |
| INT16 / FP / BF | 4,096 |
| INT4 / INT8x4 | 32,768 |

Current formulas:

| Op class | Ideal cycles |
| --- | --- |
| Conv / FC | `ceil(oH * oW * oC * kH * kW * ceil(iC / group) / mac_per_cycle)` |
| Depthwise conv | `ceil(oH * oW * oC * kH * kW / mac_per_cycle)` |
| AvgPool / MaxPool | `ceil(output_elements * kH * kW / 16)` |
| Add / Mul / Sub / H_SWISH / GELU | `ceil(output_elements / 16)` |
| Softmax | `3 * ceil(output_elements / 16)` |
| Other ops | `0` |

`ideal cyc/cum` is the running sum of `ideal cyc/layer`.

### Compile Log Table

The compile log table is parsed from compile output and is useful for comparing
compiler-visible layer metadata with simulator-visible layer metadata.

Header:

```text
Compile log (<ready layer count> layers, <skipped layer count> skipped)
```

Columns:

| Column | Meaning |
| --- | --- |
| `id` | Compile log layer id. |
| `op` | Compile log op name. |
| `iH`, `iW`, `iC` | Input shape. |
| `kH`, `kW` | Kernel size. |
| `sH`, `sW` | Stride. |
| `group` | Group count. |
| `oH`, `oW`, `oC` | Output shape. |
| `elements` | Output element count. |
| `dtype` | Compile-time dtype display. |
| `status` | `ready` or skipped reason. |

### Table Sorting And Filtering

Both the per-layer table and compile-log table support:

- Header-click sorting.
- Three-state sort: ascending, descending, original order.
- Numeric sort that strips commas, `%`, and `KB`.
- Row filtering by substring match across the full row.
- Match count display as `shown / total match`.

### Styling Contract

Current base style:

| Selector / Item | Style |
| --- | --- |
| `body` | 14px system font, margin 24px, text color `#222`. |
| `table` | Border-collapse table. |
| `th`, `td` | `1px solid #ddd`, padding `4px 6px`. |
| `th` | Background `#f4f4f4`. |
| `.kv` | Block summary line with 4px margin. |
| `.filter` | Margin `4px 0 8px`, padding `4px 6px`, min-width 280px. |
| `.sort-ind` | Small muted sort indicator. |

## Profile Index HTML

Paths:

```text
batch/profile_mdla6_pattern.html
batch/profile_hotspot.html
batch/profile_ethz_v5.html
batch/profile_ethz_v6.html
batch/profile_mlperf.html
```

Purpose:

- Provide batch-level landing pages for selected `batch/output/*.html` reports.
- Work as a static snapshot when opened through `file://`.
- Refresh from `output/` relative to the profile page when served from `batch/`
  by an HTTP server with directory listing.

### Columns

| Column | Meaning |
| --- | --- |
| `pattern` | Pattern/model name. Prefer CSV pattern name when available. |
| `link` | Relative link `output/<model_stem>.html`, which resolves to `batch/output/<model_stem>.html` when opened from `batch/`. |
| `cx` | MDLA6 baseline CX value. Present only in `profile_mdla6_pattern.html`. |
| `our_ms` | MDLA7 measured time in milliseconds. This is not cycles. |
| `conflict_ms` | MDLA7 time with per-bank SRAM port conflict timing. |
| `mesh_ms` | MDLA7 time with per-bank SRAM port conflict plus mesh router/link timing. |
| `conflict/fast` | `conflict_ms / our_ms`. |
| `mesh/fast` | `mesh_ms / our_ms`. |

`mesh` includes the SRAM bank/port conflict model from `conflict`; it then adds
edge ingress, router/link arbitration, and SRAM macro-port arbitration. Use
`mesh/conflict` when you want to isolate the extra NoC overhead.

Only `profile_mdla6_pattern.html` shows `cx` and `myms/cx`. Hotspot, ETHZ, and
MLPerf indexes intentionally hide CX columns because those corpora do not have
an MDLA6 baseline in the runner CSV.

### Embedded Row Shape

The page embeds a snapshot:

```js
const EMBEDDED_ROWS = [
  {
    pattern: "vsr_quant",
    stem: "vsr_quant",
    link: "output/vsr_quant.html",
    cx: 0.77,
    our_ms: 1.720
  }
];
```

### Data Sources

The generator scans:

```text
batch/output/*.html
```

Excluded names:

```text
profile_mdla6_pattern.html
profile_hotspot.html
._*
*.fast.html
*.conflict.html
*.mesh.html
```

Metric sources:

| Source | Usage |
| --- | --- |
| `batch/mdla6_ethz_v6_sorted.csv` | Baseline pattern list and CX values for MDLA6-pattern reports. |
| `batch/output/mdla6_pattern_regression.csv` | Latest regression result rows and preferred `our_ms` values. |
| `batch/output/hotspot_regression.csv` | Latest Hotspot slice result rows. |
| `batch/output/ethz_v5_regression.csv` | Latest ETHZ_v5 corpus result rows. |
| `batch/output/ethz_v6_regression.csv` | Latest ETHZ_v6 corpus result rows. |
| `batch/output/mlperf_regression.csv` | Latest MLPerf Tiny corpus result rows. |
| `batch/output/<stem>.profile.json` | Fallback `our_ms = total_cycles / 1.9e6`. |
| `batch/output/<stem>.html` | Fallback parse of `Sim time` ms or cycles. |

Pattern normalization:

- Strip `.cut`.
- Replace `__` with `_`.

`our_ms` priority order:

1. `mdla7_ms` from `mdla6_pattern_regression.csv`, when parseable.
2. `total_cycles / 1.9e6` from `<stem>.profile.json`.
3. Parsed `Sim time` from `<stem>.html`.

MDLA6 rows default-sort by `myms/cx`. Non-CX indexes default-sort by
`mesh/fast`.

### Refresh Behavior

Toolbar controls:

| Control | Behavior |
| --- | --- |
| `Refresh Output` | Fetch `output/` directory listing relative to the opened profile page and rebuild rows. |
| `filter pattern` | Substring filter on pattern/stem/link fields. |
| Status text | Shows whether data came from embedded snapshot or refresh. |

When HTTP refresh is unavailable, for example under `file://`, the page keeps
using `EMBEDDED_ROWS`.

### Index Styling

Current index style:

| Item | Style |
| --- | --- |
| Page | Light theme, max width 1180px, centered main, 24px padding. |
| CSS variables | `--bg:#f7f8fa`, `--panel:#fff`, `--line:#d8dde6`, `--text:#17202a`, `--muted:#657080`, `--head:#eef2f7`, `--link:#0b5cad`. |
| Table header | Sticky header. |
| Numeric cells | Right aligned with tabular numerals. |
| `our_ms` display | `toFixed(3)` when numeric. |

## Current Notes

- `profile.json` is the source of truth for actual simulator behavior.
- The compile-log table is diagnostic metadata from compiler output, not the
  source of truth for simulated cycle timing.
- `DRAM w` shown in red/bold means the layer writes non-zero data to DRAM.
- `conv occupancy` is an actual timeline occupancy metric and can include
  CONV-side waiting. It is different from `ideal util`.
- `ideal cyc/layer` is intentionally optimistic. Use it to compare compute
  demand, not to explain total layer latency by itself.
- `our_ms` in profile index pages is milliseconds. Raw cycles should only be
  shown inside the per-model HTML summary.
