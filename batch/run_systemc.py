#!/usr/bin/env python3
"""Unified SystemC regression runner.

Examples:
    ./batch/run_systemc.py --filter ethz
    ./batch/run_systemc.py --filter ethz_v6 --model-filter mobilenet --limit 3
    ./batch/run_systemc.py --filter hotspot --cx
    ./batch/run_systemc.py --filter bmm
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
OUT_ROOT = HERE / "output"
OUT_DIR = OUT_ROOT
OUTPUT_DIR = OUT_DIR
SYSTEMC_DIR = REPO_ROOT / "systemc"
COMPILE_PY = SYSTEMC_DIR / "scripts" / "compile_model.py"
PLOT_PY = SYSTEMC_DIR / "scripts" / "plot_profile.py"
MODEL_PROFILE_PY = HERE / "gen_model_profile.py"
MODEL_RUNNER = SYSTEMC_DIR / "build" / "mdla7_model_runner"

ETHZ_V6_MDLA6_CX: tuple[tuple[str, float], ...] = (
    ("resnet_quant", 0.02),
    ("resnet_float", 0.02),
    ("mobilenet_v3_quant", 0.25),
    ("xlsr_quant", 0.33),
    ("mv3_depth_quant", 0.6),
    ("yolo_v8_quant", 0.6),
    ("mobilenet_v3_float", 0.61),
    ("inception_v3_quant", 0.62),
    ("midas_v3_quant", 0.66),
    ("llama2_quant.cut", 0.71),
    ("mobilenet_v3_b4_quant", 0.77),
    ("vsr_quant", 0.77),
    ("efficientnet_b4_quant", 0.96),
    ("esrgan_quant", 0.97),
    ("swin_quant", 1.0),
    ("xlsr_float", 1.14),
    ("mobilevit_v2_quant", 1.17),
    ("mobilebert_quant.cut", 1.2),
    ("gpt2_quant.cut", 1.44),
    ("midas_v3_float", 1.54),
    ("inception_v3_float", 1.56),
    ("mv3_depth_float", 1.59),
    ("sd_diffusion_quant", 1.61),
    ("swin_float", 1.68),
    ("unet_quant", 1.84),
    ("deeplab_v3_plus_quant", 1.86),
    ("sam_quant.cut", 2.03),
    ("mobilenet_v3_b4_float", 2.13),
    ("yolo_v8_float", 2.16),
    ("mobilevit_v2_float", 2.16),
    ("efficientnet_b4_float", 2.16),
    ("esrgan_float", 2.22),
    ("vsr_float", 2.37),
    ("imdn_quant", 2.46),
    ("esrgan__int16", 3.05),
    ("sd_encoder_quant", 3.22),
    ("unet_int16", 3.39),
    ("srgan_quant", 3.4),
    ("sam_float.cut", 3.49),
    ("unet_float", 4.81),
    ("vit_b16_quant", 4.83),
    ("deeplab_v3_plus_float", 4.87),
    ("dped_int16", 5.42),
    ("microisp_quant", 5.42),
    ("imdn_float", 5.58),
    ("sd_decoder_quant", 6.53),
    ("microisp_int16", 7.11),
    ("microisp_float", 8.39),
    ("pynet_v2_quant", 9.2),
    ("srgan_float", 9.31),
    ("pynet_v2_float", 14.38),
    ("dped_quant", 23.27),
    ("dped_float", 33.4),
)

VENV_DIR = Path(os.environ.get("MDLA7_VENV") or
                Path.home() / ".venvs/mdla7").expanduser()
VENV_PY = VENV_DIR / "bin" / "python"

if VENV_PY.exists() and Path(sys.prefix).resolve() != VENV_DIR.resolve():
    os.execv(str(VENV_PY), [str(VENV_PY), __file__, *sys.argv[1:]])


def _set_output_dir(mode: str) -> None:
    global OUT_DIR, OUTPUT_DIR
    safe = (mode or "fast").replace("/", "-")
    OUT_DIR = OUT_ROOT / safe
    OUTPUT_DIR = OUT_DIR
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _artefact_paths(model: Path) -> dict[str, Path]:
    """Per-model output paths under output/<stem>.* — mdla7_model_runner derives
    .profile.json / .profile.csv from the program.bin stem so we just need
    to pass it a stable per-model name.

    Note: model names like 'mobilenet_v1_0.25_128_quant.tflite' contain dots
    in the stem, so we use f-strings rather than Path.with_suffix() (which
    treats '.25_128_quant' as a suffix and strips it)."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = model.stem
    return {
        "prog":  OUTPUT_DIR / f"{stem}.bin",
        "prof":  OUTPUT_DIR / f"{stem}.profile.json",
        "csv":   OUTPUT_DIR / f"{stem}.profile.csv",
        "gantt": OUTPUT_DIR / f"{stem}.profile.png",
        "html":  OUTPUT_DIR / f"{stem}.html",
    }

_COMPILE_FULL_RE = re.compile(
    r'^\s*layer\s+(\d+)\s+(\S+)\s+'
    r'in=(\d+)x(\d+)x(\d+)\s+'
    r'k=(\d+)x(\d+)\s+s=(\d+)x(\d+)\s+g=(\d+)\s+'
    r'out=(\d+)x(\d+)x(\d+)\s+'
    r'\((\d+)\s+(\S+)\)\s+(.*)$'
)
_COMPILE_SKIP_RE = re.compile(
    r'^\s*layer\s+(\d+)\s+(\S+)\s+'
    r'in=(\d+)x(\d+)x(\d+)\s+'
    r'skipped\s*\((.*)\)\s*$'
)

_COMPILE_FROM_RE = re.compile(r'\s+from=(\S+)\s*$')

def _parse_compile_log(log_lines: list[str]) -> list[dict]:
    """Extract the per-layer compile_model.py lines (canonical "ready" line +
    early-skip line) into structured records. Returns one entry per compile-
    stage layer, including layers the simulator never sees because compile
    skipped them (e.g., FP ADD).

    Materialized fallbacks carry a ` from=<TFLITE_OP>` trailing tag so the
    original op name survives the collapse to op_kind="matrlz"."""
    rows: list[dict] = []
    for ln in log_lines:
        m = _COMPILE_FULL_RE.match(ln)
        if m:
            status = m.group(16).strip()
            tflite_op = ""
            fm = _COMPILE_FROM_RE.search(status)
            if fm:
                tflite_op = fm.group(1)
                status = _COMPILE_FROM_RE.sub("", status).strip()
            rows.append({
                "id":     int(m.group(1)),
                "op":     m.group(2),
                "tflite_op": tflite_op,
                "in":     (int(m.group(3)), int(m.group(4)), int(m.group(5))),
                "k":      (int(m.group(6)), int(m.group(7))),
                "s":      (int(m.group(8)), int(m.group(9))),
                "group":  int(m.group(10)),
                "out":    (int(m.group(11)), int(m.group(12)), int(m.group(13))),
                "nelem":  int(m.group(14)),
                "dtype":  m.group(15),
                "status": status,
            })
            continue
        m = _COMPILE_SKIP_RE.match(ln)
        if m:
            rows.append({
                "id":     int(m.group(1)),
                "op":     m.group(2),
                "tflite_op": "",
                "in":     (int(m.group(3)), int(m.group(4)), int(m.group(5))),
                "k":      None, "s": None, "group": None, "out": None,
                "nelem":  None, "dtype": "",
                "status": f"skipped ({m.group(6)})",
            })
    return rows


def _compile_skipped_rows(compile_stdout: str) -> list[dict]:
    return [
        row for row in _parse_compile_log((compile_stdout or "").splitlines())
        if not str(row.get("status", "")).startswith("ready")
    ]


def _status_with_compile_skips(status: str, compile_stdout: str) -> str:
    skipped = _compile_skipped_rows(compile_stdout)
    if not skipped:
        return status
    tag = f"compile-skipped:{len(skipped)}"
    if status == "ok":
        return tag
    if tag in status:
        return status
    return f"{status}; {tag}"


def _write_html_report(model: Path, paths: dict[str, Path],
                       log_lines: list[str],
                       mode_label: str = "") -> Path:
    """Bundle console output + profile summary + interactive Gantt into one HTML.

    Self-contained — no CDN deps. The Gantt is a vanilla-JS SVG widget driven
    by per-engine task data embedded inline as JSON.
    """
    import html, json

    with open(paths["prof"]) as f:
        profile = json.load(f)
    summary = profile.get("summary", {}) or {}
    engines = profile.get("engines", {}) or {}
    layers  = profile.get("layers", []) or []
    task_meta = profile.get("task_meta", {}) or {}
    total_cyc = int(summary.get("total_cycles", 0) or 0)

    compile_rows_data = _parse_compile_log(log_lines)
    ready_compile_rows = [
        c for c in compile_rows_data
        if c.get("status", "").startswith("ready")
    ]

    def _kb(b: int) -> str:
        return f"{b/1024:.1f}"

    def _dtype_elem_bytes(dtype: str, *, weight: bool = False) -> int:
        d = (dtype or "").upper()
        if weight and "INT16X8" in d:
            return 1
        if "INT16" in d or "FP" in d or "BF" in d:
            return 2
        return 1

    def _conv_pure_weight_bytes(L: dict, compile_row: dict | None) -> int:
        ih, iw, ic = (L.get("in") or [0, 0, 0])
        oh, ow, oc = (L.get("out") or [0, 0, 0])
        kh, kw = (L.get("k") or [0, 0])
        group = int(L.get("group", 1) or 1)
        dtype = str((compile_row or {}).get("dtype", "") or "")
        if compile_row and compile_row.get("k"):
            kh, kw = compile_row["k"]
        return int(oc) * int(kh) * int(kw) * (int(ic) // max(group, 1)) * _dtype_elem_bytes(dtype, weight=True)

    def _conv_params_bytes(L: dict, compile_row: dict | None) -> int:
        _, _, oc = (L.get("out") or [0, 0, 0])
        dtype = str((compile_row or {}).get("dtype", "") or "")
        d = dtype.upper()
        # Mirrors mdla7_model_runner.cpp's normal params blob. INT correlation blobs are
        # model-specific and not present in compile-log rows, so leave those in
        # the generic DRAM-r total instead of inventing precision we do not have.
        if "FP" in d or "BF" in d:
            return 8 + 4 * int(oc)
        return 12 + 9 * int(oc)

    def _annotate_udma_read_tasks(tasks: list) -> list:
        """Return HTML-only UDMA-R tasks with layer/kind/byte tooltip metadata."""
        annotated = [list(t[:2]) for t in tasks]
        if not annotated or not layers:
            return annotated

        prev_end = 0
        claimed: set[int] = set()
        layer_starts: list[int] = []
        p = 0
        for L in layers:
            layer_starts.append(p)
            p = int(L.get("cycles_cum", 0) or 0)

        # FC weight prefetch can start during the preceding transpose window,
        # so start-window annotation would otherwise label the long UDMA_R as a
        # transpose read.  Mark the overlapping read as the consumer FC weight
        # prefetch in the HTML-only Gantt payload.
        for idx, L in enumerate(layers):
            op = str(L.get("op", "")).strip().lower()
            if op != "fc" or idx < 2:
                continue
            prev_op = str(layers[idx - 1].get("op", "")).strip().lower()
            prev2_op = str(layers[idx - 2].get("op", "")).strip().lower()
            if prev_op != "reshape" or prev2_op != "trnps":
                continue
            fc_start = layer_starts[idx]
            c = ready_compile_rows[idx] if idx < len(ready_compile_rows) else None
            wgt_b = _conv_pure_weight_bytes(L, c)
            candidates = [
                ti for ti, t in enumerate(annotated)
                if ti not in claimed and len(t) >= 2
                and int(t[0]) < fc_start <= int(t[1])
            ]
            if not candidates:
                continue
            ti = max(candidates, key=lambda k: int(annotated[k][1]) - int(annotated[k][0]))
            t = annotated[ti]
            layer_id = int(L.get("id", idx) or idx)
            label = f"L{layer_id} fc weight prefetch"
            if wgt_b:
                label += f" ({_kb(wgt_b)} KB)"
            t.append(label)
            if wgt_b:
                t.append(wgt_b)
            claimed.add(ti)

        for idx, L in enumerate(layers):
            end = int(L.get("cycles_cum", 0) or 0)
            op = str(L.get("op", "")).strip().lower()
            layer_tasks = [
                ti for ti, t in enumerate(annotated)
                if ti not in claimed and len(t) >= 2 and prev_end <= int(t[0]) < end
            ]
            if not layer_tasks:
                prev_end = end
                continue

            layer_id = int(L.get("id", idx) or idx)
            c = ready_compile_rows[idx] if idx < len(ready_compile_rows) else None
            labels: list[tuple[str, int]] = []
            if op in ("conv", "dwconv", "fc"):
                params_b = _conv_params_bytes(L, c)
                wgt_b = _conv_pure_weight_bytes(L, c)
                labels.append(("params", params_b))
                # If there are exactly two reads in the layer window, this is
                # the fused CONV-class case: input is already resident in L1 and
                # the second read is the weight slice. If there are more, the
                # simulator emitted input/weight/tile reads; label the first
                # weight-sized slot we can identify and leave the rest generic.
                if len(layer_tasks) == 2:
                    labels.append(("weight", wgt_b))
                elif len(layer_tasks) >= 3:
                    labels.append(("input/tile", 0))
                    labels.append(("weight/tile", 0))
            elif op in ("add", "mul", "sub", "h_swsh", "gelu", "softmax"):
                labels.append(("input/params", int(L.get("dram_r", 0) or 0)))
            elif op in ("avgpool", "maxpool"):
                labels.append(("input", int(L.get("dram_r", 0) or 0)))

            # Single-tile EWE->EWE prefetch intentionally starts the consumer's
            # input-B read before the producer layer boundary. The raw task only
            # has timing, so start-window labeling would otherwise call it a
            # generic producer read even though it visually overlaps producer EWE.
            if (len(layer_tasks) == 2 and op in ("add", "mul", "sub") and
                    idx + 1 < len(layers)):
                next_L = layers[idx + 1]
                next_op = str(next_L.get("op", "")).strip().lower()
                if next_op in ("add", "mul", "sub"):
                    ti = layer_tasks.pop()
                    next_id = int(next_L.get("id", idx + 1) or (idx + 1))
                    nbytes = int(next_L.get("dram_r", 0) or 0)
                    t = annotated[ti]
                    label = f"L{next_id} {next_op} input/params"
                    if nbytes:
                        label += f" ({_kb(nbytes)} KB)"
                    t.append(label)
                    if nbytes:
                        t.append(nbytes)
                    claimed.add(ti)

            for pos, ti in enumerate(layer_tasks):
                kind, nbytes = labels[pos] if pos < len(labels) else ("read", 0)
                t = annotated[ti]
                label = f"L{layer_id} {op} {kind}"
                if nbytes:
                    label += f" ({_kb(nbytes)} KB)"
                t.append(label)
                if nbytes:
                    t.append(nbytes)
            prev_end = end
        return annotated

    def _annotate_udma_write_tasks(tasks: list) -> list:
        """Return HTML-only UDMA-W tasks with layer/write byte metadata.

        The simulator serializes write-lane timing as bare [start,end] pairs.
        Layer accounting in profile.json is already authoritative, so attach
        each write task to the layer window it starts in and split that layer's
        dram_w bytes across its tasks.  This keeps Gantt hover/stat summaries
        consistent with the layer table and top-level DRAM W total.
        """
        annotated = [list(t[:2]) for t in tasks]
        if not annotated or not layers:
            return annotated

        write_layers = [
            (idx, L, int(L.get("dram_w", 0) or 0))
            for idx, L in enumerate(layers)
            if int(L.get("dram_w", 0) or 0) > 0
        ]
        total_w_all = sum(w for _, _, w in write_layers)
        if not write_layers or total_w_all <= 0:
            return annotated

        cursor = 0
        for wi, (idx, L, total_w) in enumerate(write_layers):
            remaining_tasks = len(annotated) - cursor
            if remaining_tasks <= 0:
                break
            if wi + 1 == len(write_layers):
                take = remaining_tasks
            else:
                take = round(len(annotated) * (total_w / total_w_all))
                take = max(1, min(take, remaining_tasks - (len(write_layers) - wi - 1)))
            if take <= 0:
                continue
            layer_tasks = list(range(cursor, cursor + take))
            cursor += take
            layer_id = int(L.get("id", idx) or idx)
            op = str(L.get("op", "")).strip().lower()
            base = total_w // len(layer_tasks) if layer_tasks else 0
            rem = total_w % len(layer_tasks) if layer_tasks else 0
            for pos, ti in enumerate(layer_tasks):
                nbytes = base + (1 if pos < rem else 0)
                t = annotated[ti]
                label = f"L{layer_id} {op} output/store"
                if nbytes:
                    label += f" ({_kb(nbytes)} KB)"
                t.append(label)
                t.append(nbytes)
        return annotated

    def _conv_wait_tasks_from_udma(udma_tasks: list) -> list:
        """Synthetic Gantt lane: CONV-class layer front-end waits on UDMA reads."""
        waits = []
        if not udma_tasks or not layers:
            return waits
        prev_end = 0
        for idx, L in enumerate(layers):
            end = int(L.get("cycles_cum", 0) or 0)
            op = str(L.get("op", "")).strip().lower()
            if L.get("streamed"):
                prev_end = end
                continue
            if op not in ("conv", "dwconv", "fc"):
                prev_end = end
                continue
            reads = [
                t for t in udma_tasks
                if len(t) >= 2 and prev_end <= int(t[0]) < end
            ]
            if not reads:
                prev_end = end
                continue
            wait_end = max(int(t[1]) for t in reads)
            if wait_end > prev_end:
                layer_id = int(L.get("id", idx) or idx)
                nbytes = sum(int(t[3]) for t in reads if len(t) > 3 and str(t[3]).isdigit())
                label = f"L{layer_id} {op} waits for udma_r"
                if nbytes:
                    label += f" ({_kb(nbytes)} KB)"
                waits.append([prev_end, wait_end, label, nbytes])
            prev_end = end
        return waits

    def _stage_from_meta(engine: str, meta: dict) -> str:
        flags = int(meta.get("stream_flags", 0) or 0)
        if engine == "udma_r":
            if flags & 0x2:
                return "load_b"
            return "load"
        if engine == "udma_w":
            return "store"
        if engine in ("conv", "requant"):
            return engine
        if engine in ("ewe", "pool", "tnps"):
            return "ewe/pool/tnps"
        return engine

    def _decorate_task_meta(engine: str, tasks: list, rtl_phases: list | None = None) -> list:
        metas = list(task_meta.get(engine) or [])
        phases = list(rtl_phases or [])
        if not metas and not phases:
            return tasks
        out = []
        for idx, raw in enumerate(tasks):
            t = list(raw)
            if len(t) < 2:
                out.append(t)
                continue
            meta = dict(metas[idx]) if idx < len(metas) and isinstance(metas[idx], dict) else {}
            if idx < len(phases) and phases[idx]:
                meta["rtl_phases"] = phases[idx]
            layer_raw = meta.get("layer", -1)
            layer_idx = int(layer_raw if layer_raw is not None else -1)
            mb = int(meta.get("mb", 0) or 0)
            flags = int(meta.get("flags", 0) or 0)
            sflags = int(meta.get("stream_flags", 0) or 0)
            if flags & 0x10 and sflags and 0 <= layer_idx < len(layers):
                L = layers[layer_idx]
                lid = int(L.get("id", layer_idx) or layer_idx)
                flow = int(L.get("flow", L.get("id", layer_idx))
                           if L.get("flow") is not None else L.get("id", layer_idx))
                op = str(L.get("op", "")).strip()
                stage = _stage_from_meta(engine, meta)
                label = f"L{lid} F{flow} mb{mb} {op} {stage}"
                if sflags & 0x10:
                    label += " final"
                if flags & 0x20:
                    label += " tail"
                if len(t) >= 3 and t[2]:
                    label += f" · {t[2]}"
                if len(t) < 3:
                    t.append(label)
                else:
                    t[2] = label
                if len(t) < 4:
                    t.append(0)
            elif len(t) < 4:
                while len(t) < 4:
                    t.append("" if len(t) == 2 else 0)
            t.append(meta)
            out.append(t)
        return out

    # v8.2: build an interactive SVG Gantt instead of embedding the matplotlib PNG.
    # Engines render as horizontal lanes; tasks as colored rects; horizontal
    # mouse wheel zooms; click+drag pans; hover shows tooltip.
    eng_payload = {}
    conv_wait_tasks = []
    for name, e in engines.items():
        tasks = list(e.get("tasks") or [])
        rtl_phases = list(e.get("rtl_phases") or [])
        if name == "udma_r":
            tasks = _annotate_udma_read_tasks(tasks)
            tasks = _decorate_task_meta(name, tasks, rtl_phases)
            conv_wait_tasks = _conv_wait_tasks_from_udma(tasks)
        elif name == "udma_w":
            tasks = _annotate_udma_write_tasks(tasks)
            tasks = _decorate_task_meta(name, tasks, rtl_phases)
        else:
            tasks = _decorate_task_meta(name, tasks, rtl_phases)
        eng_payload[name] = {
            "busy": int(e.get("busy_cycles", 0) or 0),
            "tasks": tasks,
        }

    def _rtl_phase_summary_rows() -> str:
        rows = []
        for engine in ("cmd", "udma_r", "udma_w", "conv", "requant", "ewe",
                       "pool", "tnps", "l1mgr", "l1mesh"):
            phase_totals: dict[str, int] = {}
            task_count = 0
            for task in eng_payload.get(engine, {}).get("tasks", []):
                meta = task[-1] if task and isinstance(task[-1], dict) else {}
                phases = meta.get("rtl_phases") if isinstance(meta, dict) else None
                if not isinstance(phases, list) or not phases:
                    continue
                task_count += 1
                for phase in phases:
                    if isinstance(phase, dict):
                        name = str(phase.get("name", ""))
                        cycles = int(phase.get("cycles", 0) or 0)
                    elif isinstance(phase, list) and len(phase) >= 2:
                        name = str(phase[0])
                        cycles = int(phase[1] or 0)
                    else:
                        continue
                    phase_totals[name] = phase_totals.get(name, 0) + cycles
            total = sum(phase_totals.values())
            if not task_count or total <= 0:
                continue
            ordered = [
                "issue", "decode", "act_read", "wgt_read", "read", "param_read",
                "dram_read", "l1_read", "l1_write", "dram_write", "codec",
                "dispatch", "l1_route", "mesh", "sram", "chain_read", "mac",
                "compute", "pack", "chain", "write", "fill", "done",
            ]
            parts = []
            for name in ordered:
                if name not in phase_totals:
                    continue
                cycles = phase_totals[name]
                parts.append(f"{html.escape(name)} {cycles:,} ({cycles * 100.0 / total:.1f}%)")
            rows.append(
                "<tr>"
                f"<td>{html.escape(engine)}</td>"
                f"<td style='text-align:right'>{task_count:,}</td>"
                f"<td style='text-align:right'>{total:,}</td>"
                f"<td>{' · '.join(parts)}</td>"
                "</tr>"
            )
        if not rows:
            return ""
        return (
            "<h2>RTL phase summary</h2>\n"
            "<table><thead><tr><th>engine</th><th>tasks</th><th>phase cycles</th><th>breakdown</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>\n"
        )

    rtl_phase_summary_table = _rtl_phase_summary_rows()

    def _layer_task_spans() -> list[tuple[int | None, int | None]]:
        spans: list[list[int | None]] = [[None, None] for _ in layers]

        def add_span(layer_id: int, start: int, end: int) -> None:
            if not (0 <= layer_id < len(spans)) or end < start:
                return
            cur = spans[layer_id]
            cur[0] = start if cur[0] is None else min(int(cur[0]), start)
            cur[1] = end if cur[1] is None else max(int(cur[1]), end)

        # Labeled UDMA tasks cover most visible layer front/tail work.
        for eng_name, e in eng_payload.items():
            if eng_name == "udma_w":
                continue
            for t in e.get("tasks", []):
                if len(t) < 3:
                    continue
                m = re.match(r"L(\d+)\b", str(t[2]))
                if m:
                    add_span(int(m.group(1)), int(t[0]), int(t[1]))

        # CONV/REQUANT task arrays are serialized without labels. Map them back
        # to conv-class layers by the same tile-count convention used for MAC
        # utilization so overlapped layer bands don't appear on TNPS time.
        for eng_name in ("conv", "requant"):
            tasks = list(eng_payload.get(eng_name, {}).get("tasks") or [])
            task_idx = 0
            for idx, L in enumerate(layers):
                op = str(L.get("op", "")).strip().lower()
                if op not in ("conv", "dwconv", "fc"):
                    continue
                th, toc = (L.get("tiles") or [1, 1])
                n_tasks = max(1, int(th or 1) * int(toc or 1))
                for _ in range(n_tasks):
                    if task_idx >= len(tasks):
                        break
                    t = tasks[task_idx]
                    if len(t) >= 2:
                        add_span(idx, int(t[0]), int(t[1]))
                    task_idx += 1

        return [(s, e) for s, e in spans]

    layer_spans = _layer_task_spans()
    layer_marks = [
        {"id": int(L.get("id", i)),
         "flow": int(L.get("flow", L.get("id", i)) if L.get("flow") is not None else L.get("id", i)),
         "op": str(L.get("op", "")).strip(),
         "start": (min(int(layer_spans[i][0]), int(L.get("cycles_cum", 0) or 0))
                   if layer_spans[i][0] is not None
                   else max(0, int(L.get("cycles_cum", 0) or 0) - int(L.get("cycles_layer", 0) or 0))),
         "end": int(L.get("cycles_cum", 0) or 0)}
        for i, L in enumerate(layers)
    ]

    def _tile_count_for_layer(idx: int) -> int:
        if not (0 <= idx < len(layers)):
            return 1
        th, toc = (layers[idx].get("tiles") or [1, 1])
        return max(1, int(th or 1) * int(toc or 1))

    def _microblock_payload() -> dict:
        """HTML-only view that re-buckets engine tasks by microblock stage."""
        lane_order = ["load", "conv", "requant", "ewe/pool/tnps", "store"]
        lanes = {name: {"tasks": []} for name in lane_order}

        def layer_info(idx: int) -> tuple[int, int, str]:
            if not (0 <= idx < len(layers)):
                return idx, idx, ""
            L = layers[idx]
            lid = int(L.get("id", idx) or idx)
            flow = int(L.get("flow", L.get("id", idx)) if L.get("flow") is not None else L.get("id", idx))
            op = str(L.get("op", "")).strip()
            return lid, flow, op

        def add(stage: str, task: list, layer_idx: int, mb: int,
                engine: str, detail: str = "", nbytes: int = 0) -> None:
            if stage not in lanes or len(task) < 2:
                return
            s, e = int(task[0]), int(task[1])
            if e < s:
                return
            lid, flow, op = layer_info(layer_idx)
            label = f"L{lid} F{flow} mb{mb} {op} {engine}"
            if detail:
                label += f" · {detail}"
            lanes[stage]["tasks"].append([s, e, label, int(nbytes or 0), engine])

        if task_meta:
            for engine, payload in eng_payload.items():
                for task in payload.get("tasks", []):
                    if len(task) < 5 or not isinstance(task[-1], dict):
                        continue
                    meta = task[-1]
                    mflags = int(meta.get("flags", 0) or 0)
                    if not (mflags & 0x10):
                        continue
                    if not int(meta.get("stream_flags", 0) or 0):
                        continue
                    stage = _stage_from_meta(engine, meta)
                    if stage == "load_b":
                        stage = "load"
                    if stage not in lanes:
                        continue
                    layer_raw = meta.get("layer", -1)
                    layer_idx = int(layer_raw if layer_raw is not None else -1)
                    mb = int(meta.get("mb", 0) or 0)
                    nbytes = int(task[3]) if len(task) > 3 and isinstance(task[3], int) else 0
                    detail = "tail" if int(meta.get("flags", 0) or 0) & 0x20 else ""
                    add(stage, task, layer_idx, mb, engine, detail, nbytes)
            for lane in lanes.values():
                lane["tasks"].sort(key=lambda t: (int(t[0]), int(t[1])))
            return {"lane_order": lane_order, "lanes": lanes, "source": "task_meta"}

        def assign_dense_engine(engine: str, stage: str, ops: set[str]) -> None:
            tasks = list(eng_payload.get(engine, {}).get("tasks") or [])
            task_idx = 0
            for layer_idx, L in enumerate(layers):
                op = str(L.get("op", "")).strip().lower()
                if op not in ops:
                    continue
                n = _tile_count_for_layer(layer_idx)
                for mb in range(n):
                    if task_idx >= len(tasks):
                        return
                    add(stage, tasks[task_idx], layer_idx, mb, engine)
                    task_idx += 1

        conv_ops = {"conv", "dwconv", "fc"}
        ewe_ops = {"add", "mul", "sub", "h_swsh", "relu", "gelu", "softmax"}
        pool_ops = {"avgpool", "maxpool", "mean"}
        tnps_ops = {"d2spac", "trnps", "reshape", "concat"}
        assign_dense_engine("conv", "conv", conv_ops)
        assign_dense_engine("requant", "requant", conv_ops)
        assign_dense_engine("ewe", "ewe/pool/tnps", ewe_ops)
        assign_dense_engine("pool", "ewe/pool/tnps", pool_ops)
        assign_dense_engine("tnps", "ewe/pool/tnps", tnps_ops)

        udma_counts: dict[tuple[str, int], int] = {}
        for engine, stage in (("udma_r", "load"), ("udma_w", "store")):
            for task in eng_payload.get(engine, {}).get("tasks", []):
                if len(task) < 3:
                    continue
                m = re.match(r"L(\d+)\b", str(task[2]))
                if not m:
                    continue
                layer_idx = int(m.group(1))
                n = _tile_count_for_layer(layer_idx)
                key = (engine, layer_idx)
                mb = udma_counts.get(key, 0) % n
                udma_counts[key] = udma_counts.get(key, 0) + 1
                nbytes = int(task[3]) if len(task) > 3 and str(task[3]).isdigit() else 0
                add(stage, task, layer_idx, mb, engine, str(task[2]), nbytes)

        for lane in lanes.values():
            lane["tasks"].sort(key=lambda t: (int(t[0]), int(t[1])))
        return {"lane_order": lane_order, "lanes": lanes, "source": "inferred"}

    gantt_data_json = json.dumps({
        "engines": eng_payload,
        "microblocks": _microblock_payload(),
        "layers":  layer_marks,
        "conv_wait": conv_wait_tasks,
        "total":   total_cyc,
    })

    def _dtype_bit_widths(dtype: str) -> tuple[int, int]:
        d = (dtype or "").upper()
        # Mirrors ConvEngine::conv_cycles(): bit-mult invariant with
        # 1,048,576 bit-mults/cycle.
        if "FP8" in d:
            return (8, 8)
        if "INT8X4" in d or "INT4" in d:
            return (8, 4)
        if "INT16X4" in d:
            return (16, 4)
        if "INT16X8" in d:
            return (16, 8)
        if "INT16" in d or "FP" in d or "BF" in d:
            return (16, 16)
        return (8, 8)

    def _conv_model_cycles(macs: int, dtype: str, tile_count: int) -> int:
        a_bits, b_bits = _dtype_bit_widths(dtype)
        return _ceil_div(int(macs) * a_bits * b_bits, 1_048_576) + 64 * max(1, tile_count)

    def _conv_task_cycles_by_layer() -> list[int]:
        """Map serialized CONV engine tasks back to conv-class layers.

        The layer wall window can be shorter than a CONV task when functional
        chain delivery and stream overlap let a successor start before the
        producer's modeled compute tail has retired.  For MAC utilization,
        divide by the CONV task time itself, not by that shortened layer window.
        """
        tasks = list((engines.get("conv", {}) or {}).get("tasks") or [])
        by_layer = [0] * len(layers)
        task_idx = 0
        for idx, L in enumerate(layers):
            op = str(L.get("op", "")).strip().lower()
            if op not in ("conv", "dwconv", "fc"):
                continue
            th, toc = (L.get("tiles") or [1, 1])
            n_tasks = max(1, int(th or 1) * int(toc or 1))
            total = 0
            for _ in range(n_tasks):
                if task_idx >= len(tasks):
                    break
                t = tasks[task_idx]
                if len(t) >= 2:
                    total += max(0, int(t[1]) - int(t[0]))
                task_idx += 1
            by_layer[idx] = total
        return by_layer

    conv_task_cycles = _conv_task_cycles_by_layer()

    def _dtype_elem_lanes(dtype: str) -> int:
        d = (dtype or "").upper()
        if "FP" in d or "BF" in d:
            return 32
        if "INT16" in d:
            return 32
        if "INT4" in d or "INT8X4" in d:
            return 64
        return 64

    def _dtype_elem_bytes(dtype: str) -> int:
        d = (dtype or "").upper()
        if "INT16" in d or "FP16" in d or "BFP16" in d or "BF16" in d:
            return 2
        return 1   # INT8 / FP8 / unknown -> assume single byte per element

    def _ceil_div(a: int, b: int) -> int:
        return (a + b - 1) // b if b > 0 else 0

    # TNPS architectural target: 8 lanes x 16B R + 8 lanes x 16B W through
    # L1Manager (memory.md). 128 B/cycle is the L1<->L1 sustained throughput
    # ceiling; setup overhead is small (a single descriptor wakes the engine).
    TNPS_BYTES_PER_CYCLE = 128
    TNPS_SETUP_CYC = 16

    # Layer ops that go through TNPS-class data movement (matches the
    # _OP_TO_ENGINE "tnps" partition plus folding-eligible variants).
    TNPS_OPS = frozenset((
        "reshape", "trnps", "d2spac", "s2spac", "concat",
        "squeez", "expand", "slice", "sslice", "split",
        "pad", "pack", "unpack", "tile",
    ))

    def _ideal_layer_cycles(L: dict, compile_row: dict | None) -> int:
        op = str(L.get("op", "")).strip().lower()
        ih, iw, ic = (L.get("in")  or [0, 0, 0])
        oh, ow, oc = (L.get("out") or [0, 0, 0])
        kh, kw = (L.get("k") or [0, 0])
        group = int(L.get("group", 1) or 1)
        dtype = ""
        if compile_row:
            dtype = str(compile_row.get("dtype", "") or "")
            if compile_row.get("k"):
                kh, kw = compile_row["k"]
        out_elems = int(oh) * int(ow) * int(oc)
        th, toc = (L.get("tiles") or [1, 1])
        tile_count = max(1, int(th or 1) * int(toc or 1))
        if op in ("conv", "dwconv", "fc"):
            if op == "dwconv":
                macs = out_elems * int(kh) * int(kw)
            else:
                macs = out_elems * int(kh) * int(kw) * _ceil_div(int(ic), max(group, 1))
            return _conv_model_cycles(macs, dtype, tile_count)
        if op in ("avgpool", "maxpool"):
            return _ceil_div(out_elems * int(kh) * int(kw), _dtype_elem_lanes(dtype))
        if op in ("add", "mul", "sub", "h_swsh", "gelu",
                  "logist", "rsqrt", "tanh"):
            # INT8 RSQRT/TANH/LOGISTIC use a single-pass LUT lookup; same
            # throughput model as binary EWE since the lookup retires one
            # output byte per lane per cycle.
            return _ceil_div(out_elems, _dtype_elem_lanes(dtype))
        if op == "softmax":
            return 3 * _ceil_div(out_elems, _dtype_elem_lanes(dtype))
        if op in TNPS_OPS or op == "matrlz":
            # Pure data-movement floor: bytes / TNPS B-per-cyc + per-tile setup.
            # Even if the runtime folded the layer (view passthrough or layout
            # tail), this is what TNPS WOULD spend if it actually ran the copy.
            # Materialized fallbacks issue an explicit TNPS self-copy of
            # ref_size bytes (see runner OK_MATERIALIZE case), so they share
            # the same throughput floor as the native TNPS-class ops.
            ref_bytes = 0
            if compile_row and compile_row.get("nelem"):
                ref_bytes = int(compile_row["nelem"]) * _dtype_elem_bytes(dtype)
            else:
                ref_bytes = out_elems * _dtype_elem_bytes(dtype)
            return _ceil_div(ref_bytes, TNPS_BYTES_PER_CYCLE) + TNPS_SETUP_CYC * tile_count
        return 0

    ideal_rows: list[tuple[int, int]] = []
    ideal_cum = 0
    for idx, L in enumerate(layers):
        c = ready_compile_rows[idx] if idx < len(ready_compile_rows) else None
        ideal = _ideal_layer_cycles(L, c)
        ideal_cum += ideal
        ideal_rows.append((ideal, ideal_cum))

    # Materialized fallback layers carry their original TFLite op as a
    # ` from=<OPNAME>` tag in the compile log; surface it inline beside the
    # matrlz label so the per-layer profile shows e.g. matrlz(BATCH_MATMUL).
    tflite_op_by_layer: dict[int, str] = {
        int(c["id"]): str(c.get("tflite_op", "") or "").strip()
        for c in ready_compile_rows
        if c.get("tflite_op")
    }

    # Map layer op to the HW engine that runs it. Mirrors the
    # assign_dense_engine() partition above; conv-class layers go through
    # both CONV and REQUANT, so list both. TNPS group covers every layout-
    # movement op the runtime routes through the TNPS engine codepath.
    _OP_TO_ENGINE = {
        "conv": "conv+requant", "dwconv": "conv+requant", "fc": "conv+requant",
        "add": "ewe", "mul": "ewe", "sub": "ewe",
        "h_swsh": "ewe", "relu": "ewe", "gelu": "ewe", "softmax": "ewe",
        "logist": "ewe", "rsqrt": "ewe", "tanh": "ewe",
        "avgpool": "pool", "maxpool": "pool", "mean": "pool",
        "shape": "—",   # constant fill, no engine — wgt UDMA load only
    }
    for _tnps_op in TNPS_OPS:
        _OP_TO_ENGINE[_tnps_op] = "tnps"

    def _tnps_fold_tag(L: dict) -> str:
        """Classify how the runtime scheduled a TNPS-class layer using the
        per-layer counters in profile.json:
            view   - no descriptor emitted (L1 view passthrough); zero traffic
            folded - layout tail emitted but absorbed into the prior chain;
                     SRAM/DRAM accounted but the wall window is ~0
            copy   - real engine work, non-zero wall cycles"""
        cyc    = int(L.get("cycles_layer", 0) or 0)
        sram_r = int(L.get("sram_r", 0) or 0)
        sram_w = int(L.get("sram_w", 0) or 0)
        dram_r = int(L.get("dram_r", 0) or 0)
        dram_w = int(L.get("dram_w", 0) or 0)
        if cyc == 0 and sram_r == 0 and sram_w == 0 and dram_r == 0 and dram_w == 0:
            return "view"
        if cyc == 0 and (sram_r > 0 or sram_w > 0 or dram_w > 0):
            return "folded"
        return "copy"

    def _layer_row(L: dict) -> str:
        layer_idx = int(L.get("id", 0) or 0)
        flow_raw = L.get("flow")
        flow_idx = int(flow_raw if flow_raw is not None else layer_idx)
        ih, iw, ic = (L.get("in")  or [0, 0, 0])
        oh, ow, oc = (L.get("out") or [0, 0, 0])
        kh, kw     = (L.get("k")   or [0, 0])
        sh, sw     = (L.get("s")   or [0, 0])
        th, toc    = (L.get("tiles") or [1, 1])
        ideal_layer, ideal_cum_val = (
            ideal_rows[layer_idx] if 0 <= layer_idx < len(ideal_rows) else (0, 0)
        )
        cycles_layer = int(L.get('cycles_layer', 0) or 0)
        op_norm = str(L.get("op", "")).strip().lower()
        engine_name = _OP_TO_ENGINE.get(op_norm, "—")
        # view-class TNPS (cy=0, no bus): engine column shows "—" since no
        # hardware dispatch happens.
        if op_norm in TNPS_OPS and _tnps_fold_tag(L) == "view":
            engine_name = "—"
        conv_task = conv_task_cycles[layer_idx] if 0 <= layer_idx < len(conv_task_cycles) else 0
        mac_util = (
            100.0 * ideal_layer / conv_task
            if conv_task > 0 and ideal_layer > 0 and op_norm in ("conv", "dwconv", "fc")
            else 0.0
        )
        # v8.5: old "conv util" is actually CONV-engine occupancy within the
        # layer window. It includes dependency stalls hidden inside the CONV
        # task, so keep it explicit and show ideal/actual util separately.
        util       = float(L.get("conv_util_pct", L.get("util_pct", 0.0)) or 0.0)
        passed = L.get("pass", False)
        pass_cell = ('<td style="color:#0a7d23">PASS</td>' if passed
                     else '<td style="color:#b00020">FAIL</td>')
        op_label = str(L.get('op', '')).strip()
        if op_label == "matrlz":
            tflite_op = tflite_op_by_layer.get(layer_idx, "")
            label = f"matrlz({tflite_op})" if tflite_op else "matrlz"
            op_cell_layer = (f"<td style='color:#b00020;font-weight:600'>"
                             f"{html.escape(label)}</td>")
        elif op_norm in TNPS_OPS:
            tag = _tnps_fold_tag(L)
            label = f"{op_label}({tag})" if tag != "view" else op_label
            style = " style='color:#888'" if tag == "view" else (
                    " style='color:#a06000'" if tag == "folded" else "")
            op_cell_layer = f"<td{style}>{html.escape(label)}</td>"
        elif op_label == "fc(bmm)":
            op_cell_layer = "<td style='color:#1a7ab0;font-weight:600'>fc(bmm)</td>"
        else:
            op_cell_layer = f"<td>{html.escape(op_label)}</td>"
        return ("<tr>" +
            f"<td>{layer_idx}</td>" +
            f"<td>F{flow_idx}</td>" +
            op_cell_layer +
            f"<td>{html.escape(engine_name)}</td>" +
            f"<td style='text-align:right'>{cycles_layer:,}</td>" +
            f"<td style='text-align:right'>{int(L.get('cycles_cum',0)):,}</td>" +
            f"<td style='text-align:right'>{ideal_layer:,}</td>" +
            f"<td style='text-align:right'>{ideal_cum_val:,}</td>" +
            f"<td style='text-align:right'>{mac_util:.2f}%</td>" +
            f"<td style='text-align:right'>{util:.1f}%</td>" +
            f"<td style='text-align:right'>{_kb(L.get('dram_r',0))}</td>" +
            (f"<td style='text-align:right;color:#b00020;font-weight:600'>{_kb(L.get('dram_w',0))}</td>"
             if int(L.get('dram_w', 0) or 0) > 0
             else f"<td style='text-align:right'>{_kb(L.get('dram_w',0))}</td>") +
            f"<td style='text-align:right'>{_kb(L.get('sram_r',0))}</td>" +
            f"<td style='text-align:right'>{_kb(L.get('sram_w',0))}</td>" +
            f"<td>{ih}</td><td>{iw}</td><td>{ic}</td>" +
            f"<td>{oh}</td><td>{ow}</td><td>{oc}</td>" +
            f"<td>{kh}</td><td>{kw}</td><td>{sh}</td><td>{sw}</td>" +
            f"<td>{L.get('group','')}</td>" +
            f"<td>{th}×{toc}</td>" +
            pass_cell + "</tr>")
    layer_rows = "".join(_layer_row(L) for L in layers)

    def _eng_row(name: str, e: dict) -> str:
        busy = int(e.get("busy_cycles", 0) or 0)
        pct = (100.0 * busy / total_cyc) if total_cyc else 0.0
        return (f"<tr><td>{html.escape(name)}</td>"
                f"<td style='text-align:right'>{busy:,}</td>"
                f"<td style='text-align:right'>{pct:.1f}%</td></tr>")
    eng_rows = "".join(_eng_row(n, e) for n, e in engines.items())

    # v8.16: surface compile_model's per-layer "ready / skipped" lines as a
    # structured table. v8.37: some former skips now appear as ready `matrlz`
    # fallback layers, so keep the raw op name/status visible.
    def _compile_row(c: dict) -> str:
        ih, iw, ic = c["in"]
        if c["out"] is None:
            shape_cells = (
                f"<td>{ih}</td><td>{iw}</td><td>{ic}</td>" +
                "<td colspan='2' style='color:#888'>—</td>" +
                "<td colspan='2' style='color:#888'>—</td>" +
                "<td style='color:#888'>—</td>" +
                "<td colspan='3' style='color:#888'>—</td>"
            )
        else:
            kh, kw   = c["k"]
            sh, sw   = c["s"]
            oh, ow, oc = c["out"]
            shape_cells = (
                f"<td>{ih}</td><td>{iw}</td><td>{ic}</td>" +
                f"<td>{kh}</td><td>{kw}</td>" +
                f"<td>{sh}</td><td>{sw}</td>" +
                f"<td>{c['group']}</td>" +
                f"<td>{oh}</td><td>{ow}</td><td>{oc}</td>"
            )
        nelem  = (f"{c['nelem']:,}" if c["nelem"] is not None
                  else "<span style='color:#888'>—</span>")
        dtype  = c["dtype"] or ""
        status = c["status"]
        if status == "ready":
            status_cell = "<td style='color:#0a7d23'>ready</td>"
        else:
            status_cell = (f"<td style='color:#a06000'>"
                           f"{html.escape(status)}</td>")
        op_str = c['op'].strip()
        tflite_op = (c.get("tflite_op", "") or "").strip()
        if op_str == "matrlz":
            label = f"matrlz({tflite_op})" if tflite_op else "matrlz"
            op_cell = (f"<td style='color:#b00020;font-weight:600'>"
                       f"{html.escape(label)}</td>")
        else:
            op_cell = f"<td>{html.escape(op_str)}</td>"
        return ("<tr>" +
                f"<td>{c['id']}</td>" +
                op_cell +
                shape_cells +
                f"<td style='text-align:right'>{nelem}</td>" +
                f"<td>{html.escape(dtype)}</td>" +
                status_cell + "</tr>")
    compile_rows = "".join(_compile_row(c) for c in compile_rows_data)
    n_compile = len(compile_rows_data)
    n_skipped_compile = sum(1 for c in compile_rows_data
                            if not c["status"].startswith("ready"))
    if n_skipped_compile:
        graph_coverage_row = (
            f"  <span class=\"kv warn\"><b>Graph coverage:</b> INCOMPLETE — "
            f"{n_skipped_compile}/{n_compile} compile rows skipped; simulator PASS covers compiled layers only</span>"
        )
    else:
        graph_coverage_row = (
            f"  <span class=\"kv\"><b>Graph coverage:</b> complete — "
            f"0/{n_compile} compile rows skipped</span>"
        )
    verification_row = (
        "  <span class=\"kv\"><b>Verification:</b> per compiled-layer DRAM "
        "reference compare, including the final compiled layer; no separate "
        "original-TFLite final-output checker</span>"
    )

    tag = f" ({mode_label})" if mode_label else ""
    title = f"MDLA7 profile — {model.name}{tag}"
    n_pass = int(summary.get("pass", 0) or 0)
    n_fail = int(summary.get("fail", 0) or 0)
    n_total = int(summary.get("layers", 0) or 0)
    l1mesh = summary.get("l1mesh") if isinstance(summary.get("l1mesh"), dict) else None
    l1mesh_rows = ""
    if l1mesh:
        def _lane_table_rows(kind: str) -> str:
            rows = []
            for lane in l1mesh.get(kind, []) or []:
                rows.append(
                    "<tr>"
                    f"<td>{int(lane.get('id', 0) or 0)}</td>"
                    f"<td style='text-align:right'>{int(lane.get('accesses', 0) or 0):,}</td>"
                    f"<td style='text-align:right'>{int(lane.get('bytes', 0) or 0) / 1024:.1f}</td>"
                    f"<td style='text-align:right'>{int(lane.get('avg_latency_cycles', 0) or 0):,}</td>"
                    f"<td style='text-align:right'>{int(lane.get('avg_wait_cycles', 0) or 0):,}</td>"
                    f"<td style='text-align:right'>{int(lane.get('avg_service_cycles', 0) or 0):,}</td>"
                    f"<td style='text-align:right'>{int(lane.get('max_latency_cycles', 0) or 0):,}</td>"
                    f"<td style='text-align:right'>{int(lane.get('max_wait_cycles', 0) or 0):,}</td>"
                    f"<td style='text-align:right'>{int(lane.get('max_service_cycles', 0) or 0):,}</td>"
                    "</tr>"
                )
            return "".join(rows)
        l1mesh_lane_tables = f"""
<h2>L1Mesh Payload lane latency</h2>
<div style="display:grid; grid-template-columns:1fr 1fr; gap:16px;">
  <div>
    <h3 style="font-size:13px; margin:0 0 6px;">Payload R lanes</h3>
    <table class="sortable"><thead><tr><th>lane</th><th>accesses</th><th>KB</th><th>avg cyc</th><th>avg wait</th><th>avg service</th><th>max cyc</th><th>max wait</th><th>max service</th></tr></thead>
    <tbody>{_lane_table_rows('read_lanes')}</tbody></table>
  </div>
  <div>
    <h3 style="font-size:13px; margin:0 0 6px;">Payload W lanes</h3>
    <table class="sortable"><thead><tr><th>lane</th><th>accesses</th><th>KB</th><th>avg cyc</th><th>avg wait</th><th>avg service</th><th>max cyc</th><th>max wait</th><th>max service</th></tr></thead>
    <tbody>{_lane_table_rows('write_lanes')}</tbody></table>
  </div>
</div>
"""
        l1mesh_rows = f"""
  <span class="kv"><b>L1Mesh NoC:</b> {int(l1mesh.get('accesses',0) or 0):,} accesses / {int(l1mesh.get('stripes',0) or 0):,} stripes / imposed {int(l1mesh.get('imposed_wait_cycles',0) or 0):,} cycles</span>
  <span class="kv"><b>NoC wait edge/router/link/local/sram:</b> {int(l1mesh.get('edge_wait_cycles',0) or 0):,} / {int(l1mesh.get('router_wait_cycles',0) or 0):,} / {int(l1mesh.get('link_wait_cycles',0) or 0):,} / {int(l1mesh.get('local_wait_cycles',0) or 0):,} / {int(l1mesh.get('sram_wait_cycles',0) or 0):,} cycles</span>
  <span class="kv"><b>NoC service edge/router/link/local/sram:</b> {int(l1mesh.get('edge_service_cycles',0) or 0):,} / {int(l1mesh.get('router_service_cycles',0) or 0):,} / {int(l1mesh.get('link_service_cycles',0) or 0):,} / {int(l1mesh.get('local_service_cycles',0) or 0):,} / {int(l1mesh.get('sram_service_cycles',0) or 0):,} cycles</span>"""
    else:
        l1mesh_lane_tables = ""
    html_doc = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>
body {{ font-family: -apple-system,Segoe UI,Helvetica,Arial,sans-serif;
        max-width: 1200px; margin: 24px auto; padding: 0 16px; color:#222; }}
h1 {{ font-size: 20px; margin-bottom: 4px; }}
h2 {{ font-size: 15px; border-bottom: 1px solid #ddd; padding-bottom: 4px;
      margin-top: 28px; }}
table {{ border-collapse: collapse; font-size: 12px; width: 100%; }}
th,td {{ border: 1px solid #e4e4e4; padding: 4px 8px; text-align: left;
         font-variant-numeric: tabular-nums; }}
th {{ background:#f4f4f4; }}
table.sortable th {{ cursor: pointer; user-select: none; position: relative; padding-right: 16px; }}
table.sortable th:hover {{ background:#e8e8e8; }}
table.sortable th[data-sort="asc"]::after  {{ content: " ▲"; position:absolute; right:4px; color:#888; }}
table.sortable th[data-sort="desc"]::after {{ content: " ▼"; position:absolute; right:4px; color:#888; }}
input.filter {{ width: 220px; padding: 3px 6px; margin: 4px 0 8px 0;
                font: 12px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
                border: 1px solid #ccc; border-radius: 3px; }}
.filter-info {{ display: inline-block; margin-left: 8px; color: #888; font-size: 11px; }}
.kv {{ display: block; margin: 2px 0; }}
.kv b {{ color:#556; font-weight:600; }}
.warn {{ color:#a06000; font-weight:600; }}
</style></head>
<body>
<h1>{html.escape(title)}</h1>
<div>
  <span class="kv"><b>Model:</b> {html.escape(str(model.relative_to(REPO_ROOT)))}</span>
  <span class="kv"><b>Compiled layers:</b> {n_total} (PASS {n_pass} / FAIL {n_fail})</span>
{graph_coverage_row}
{verification_row}
  <span class="kv"><b>Sim time:</b> {total_cyc/1.9e6:.3f} ms @ 1.9 GHz ({total_cyc:,} cycles)</span>
  <span class="kv"><b>Util:</b> avg {float(summary.get('util_avg_pct',0.0) or 0.0):.1f}% / peak {float(summary.get('util_peak_pct',0.0) or 0.0):.1f}% ({html.escape(str(summary.get('util_peak_engine','')))})</span>
  <span class="kv"><b>DRAM r/w:</b> {summary.get('dram_read_bytes',0)/1e6:.2f} / {summary.get('dram_write_bytes',0)/1e6:.2f} MB</span>
  <span class="kv"><b>SRAM r/w:</b> {summary.get('sram_read_bytes',0)/1e6:.2f} / {summary.get('sram_write_bytes',0)/1e6:.2f} MB</span>
{l1mesh_rows}
</div>

<h2>Per-engine busy</h2>
<table><thead><tr><th>engine</th><th>busy cycles</th><th>utilization</th></tr></thead>
<tbody>{eng_rows}</tbody></table>

{l1mesh_lane_tables}

{rtl_phase_summary_table}

<h2>Gantt timeline (interactive)</h2>
<div id="gantt-controls" style="font-size:12px; margin:6px 0;">
  <button id="gantt-zoomin">+</button>
  <button id="gantt-zoomout">−</button>
  <button id="gantt-reset">Reset</button>
  &nbsp;|&nbsp;
  <label>scope:
    <input id="gantt-scope" placeholder="op name · L&lt;id&gt; · range 3-6"
           style="width:200px; padding:1px 4px; font: 11px ui-monospace,Menlo,Consolas,monospace;"
           autocomplete="off"/></label>
  <button id="gantt-prev"  title="previous match">◀</button>
  <button id="gantt-next"  title="next match">▶</button>
  <span id="gantt-matches" style="margin-left:6px; color:#666;"></span>
  <br/>
  <span id="gantt-cursor" style="color:#555;">cycle: —</span>
  <span id="gantt-window" style="margin-left:12px; color:#888;"></span>
  <span style="float:right; color:#888;">drag = zoom-to-range · alt+drag <i>or</i> right-drag = pan · wheel-on-lanes = zoom · wheel-on-axis = scroll · keys: ←/→ pan, +/− zoom, 0 reset</span>
</div>
<h3 style="font-size:13px; margin:10px 0 4px;">Original engine timeline</h3>
<div id="gantt-tip"
     style="position:absolute; pointer-events:none; background:#222; color:#eee;
            font: 11px/1.3 ui-monospace,Menlo,Consolas,monospace; padding:4px 7px;
            border-radius:4px; opacity:0; transition:opacity 0.08s;
            white-space:pre; z-index:10;"></div>
<svg id="gantt-svg" width="100%"
     style="border:1px solid #ddd; background:#fafafa; cursor:crosshair;
            display:block; user-select:none;">
  <g id="gantt-grid"></g>
  <g id="gantt-ops"></g>
  <g id="gantt-layers"></g>
  <g id="gantt-scope-hi"></g>
  <g id="gantt-bars"></g>
  <g id="gantt-anchor"></g>
  <g id="gantt-axis"></g>
  <rect id="gantt-selrect" x="0" y="0" width="0" height="0"
        fill="rgba(66,135,245,0.18)" stroke="#4287f5" stroke-width="1"
        stroke-dasharray="3,2" style="display:none; pointer-events:none;"/>
</svg>
<h3 style="font-size:13px; margin:14px 0 4px;">Microblock stage timeline</h3>
<svg id="micro-gantt-svg" width="100%"
     style="border:1px solid #ddd; background:#fafafa; cursor:default;
            display:block; user-select:none;">
  <g id="micro-gantt-grid"></g>
  <g id="micro-gantt-bars"></g>
  <g id="micro-gantt-axis"></g>
</svg>
<script id="gantt-data" type="application/json">{gantt_data_json}</script>
<script>
(function() {{
  const data = JSON.parse(document.getElementById('gantt-data').textContent);
  // Pre-compute layer windows. New profiles include task-derived starts so
  // overlapped/non-monotonic layer ends don't make a CONV/FC band appear over
  // TNPS time; older profiles fall back to previous-end windows.
  data.layers.forEach((L, i) => {{
    if (L.start === undefined || L.start === null) {{
      L.start = i > 0 ? data.layers[i-1].end : 0;
    }}
  }});

  const ENG_COLORS = {{
    cmd:     '#555555',
    udma:    '#4287f5',           // legacy (kept for older bins)
    udma_r:  '#4287f5',           // DRAM → L1 (load): blue
    udma_w:  '#7c4dff',           // L1  → DRAM (store): purple-blue
    conv:    '#e84545',
    requant: '#f5a142',
    ewe:     '#3ec56e',
    pool:    '#a945e8',
    tnps:    '#00a6a6',
    l1mgr:   '#607d8b',
    l1mesh:  '#795548',
  }};
  const PHASE_COLORS = {{
    issue:   '#202020',
    decode:  '#455a64',
    dispatch:'#6b6b6b',
    dram_read:'#1f6fd1',
    dram_write:'#6f42c1',
    l1_read: '#4f9df8',
    l1_write:'#38a169',
    l1_route:'#607d8b',
    mesh:    '#795548',
    sram:    '#8d6e63',
    codec:   '#b7791f',
    act_read:'#2f80ed',
    wgt_read:'#56ccf2',
    param_read:'#7b61ff',
    chain_read:'#00a6a6',
    mac:     '#eb5757',
    pack:    '#f2994a',
    chain:   '#00a6a6',
    fill:    '#bdbdbd',
    read:    '#2f80ed',
    compute: '#f2994a',
    write:   '#27ae60',
    done:    '#8a8a8a',
  }};
  const svg     = document.getElementById('gantt-svg');
  const gGrid   = document.getElementById('gantt-grid');
  const gOps    = document.getElementById('gantt-ops');
  const gLayers = document.getElementById('gantt-layers');
  const gScope  = document.getElementById('gantt-scope-hi');
  const gBars   = document.getElementById('gantt-bars');
  const gAnchor = document.getElementById('gantt-anchor');
  const gAxis   = document.getElementById('gantt-axis');
  const microSvg  = document.getElementById('micro-gantt-svg');
  const microGrid = document.getElementById('micro-gantt-grid');
  const microBars = document.getElementById('micro-gantt-bars');
  const microAxis = document.getElementById('micro-gantt-axis');
  const selRect = document.getElementById('gantt-selrect');
  const tip     = document.getElementById('gantt-tip');
  const cursorInfo = document.getElementById('gantt-cursor');
  const windowInfo = document.getElementById('gantt-window');
  const matchInfo  = document.getElementById('gantt-matches');
  const scopeInput = document.getElementById('gantt-scope');

  const eng_names = Object.keys(data.engines);
  const N_LANES = eng_names.length;
  const micro = data.microblocks || {{lane_order: [], lanes: {{}}}};
  const micro_names = micro.lane_order || Object.keys(micro.lanes || {{}});
  const N_MICRO_LANES = micro_names.length;
  // v8.3 / v8.9 geometry: TOP → OP_ROW (28 px) → engine lanes (56 px each)
  // → axis (32 px). The OP_ROW shows one rect per layer with the op id and
  // name; click an op rect to zoom to that layer.
  const LANE_H = 56, LANE_PAD = 8, LEFT = 72, RIGHT = 12, TOP = 8, AXIS_H = 32;
  const OP_ROW_H = 34;
  const total = Math.max(1, data.total | 0);
  const CYCLES_PER_MS = 1.9e6;

  // viewState: visible cycle window.
  let view0 = 0, view1 = total;
  // scope highlight: which layer (if any) is currently focused.
  let highlight = null;       // {{start, end, id, op}} or null
  let anchorCycle = null;      // left-click anchor for cycle/ms delta readout

  function fmt(n) {{ return n.toLocaleString(); }}
  function fmtSigned(n) {{ return (n >= 0 ? '+' : '-') + fmt(Math.abs(n)); }}
  function fmtSignedMs(n) {{
    const v = n / CYCLES_PER_MS;
    return (v >= 0 ? '+' : '-') + Math.abs(v).toFixed(6);
  }}
  function taskMeta(task) {{
    const m = task && task.length ? task[task.length - 1] : null;
    return m && typeof m === 'object' && !Array.isArray(m) ? m : null;
  }}
  function rtlPhaseText(task) {{
    const m = taskMeta(task);
    const phases = m && Array.isArray(m.rtl_phases) ? m.rtl_phases : null;
    if (!phases || !phases.length) return '';
    return phases.map(p => {{
      const name = Array.isArray(p) ? p[0] : p.name;
      const cyc = +(Array.isArray(p) ? p[1] : p.cycles) || 0;
      const rb = +(Array.isArray(p) ? 0 : p.read_bytes) || 0;
      const wb = +(Array.isArray(p) ? 0 : p.write_bytes) || 0;
      const elems = +(Array.isArray(p) ? 0 : p.elems) || 0;
      const lanes = +(Array.isArray(p) ? 0 : p.lanes) || 0;
      const stall = Array.isArray(p) ? '' : (p.stall || '');
      let text = `${{name}}:${{fmt(cyc)}}`;
      if (rb) text += `/r${{fmt(rb)}}B`;
      if (wb) text += `/w${{fmt(wb)}}B`;
      if (elems) text += `/e${{fmt(elems)}}`;
      if (lanes) text += `/l${{fmt(lanes)}}`;
      if (stall) text += `/${{stall}}`;
      return text;
    }}).join('  ');
  }}
  function rtlPhases(task) {{
    const m = taskMeta(task);
    const phases = m && Array.isArray(m.rtl_phases) ? m.rtl_phases : null;
    return phases && phases.length ? phases : null;
  }}
  function renderPhaseSegments(task, y, h, x, left, right) {{
    const phases = rtlPhases(task);
    if (!phases) return '';
    const s = +task[0], e = +task[1];
    const dur = Math.max(1, e - s);
    const totalPhase = phases.reduce((acc, p) => acc + Math.max(0, +p[1] || 0), 0);
    if (totalPhase <= 0) return '';
    let out = '';
    let acc = 0;
    const yy = y + Math.max(1, h - 7);
    const hh = Math.min(5, Math.max(2, h - 2));
    for (const p of phases) {{
      const name = String((Array.isArray(p) ? p[0] : p.name) || '');
      const cyc = Math.max(0, +(Array.isArray(p) ? p[1] : p.cycles) || 0);
      if (!cyc) continue;
      const ps = s + dur * acc / totalPhase;
      acc += cyc;
      const pe = s + dur * acc / totalPhase;
      const x0 = Math.max(x(ps), left);
      const x1 = Math.min(x(pe), right);
      const w = x1 - x0;
      if (w < 0.8) continue;
      const color = PHASE_COLORS[name] || '#555';
      out += `<rect class="rtl-phase" x="${{x0}}" y="${{yy}}" width="${{Math.max(1, w)}}" height="${{hh}}" `
          + `fill="${{color}}" opacity="0.9" pointer-events="none"/>`;
    }}
    return out;
  }}
  function clamp(v, lo, hi) {{ return Math.max(lo, Math.min(hi, v)); }}
  function esc(s) {{
    return String(s).replace(/[&<>"']/g, ch => ({{
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
    }}[ch]));
  }}
  function pxToCyc(px, W) {{
    const span = view1 - view0;
    const sx = (W - LEFT - RIGHT) / span;
    return view0 + (px - LEFT) / sx;
  }}

  function renderMicro() {{
    if (!microSvg) return;
    const W = microSvg.clientWidth || 1100;
    const LANE_H2 = 42, LANE_PAD2 = 7, LEFT2 = 92, RIGHT2 = 12, TOP2 = 8, AXIS_H2 = 28;
    const H = TOP2 + Math.max(1, N_MICRO_LANES) * LANE_H2 + AXIS_H2;
    microSvg.setAttribute('viewBox', `0 0 ${{W}} ${{H}}`);
    microSvg.setAttribute('height', H);
    const span = Math.max(1, view1 - view0);
    const sx = (W - LEFT2 - RIGHT2) / span;
    const x = c => LEFT2 + (c - view0) * sx;

    let grid = '';
    if (!N_MICRO_LANES) {{
      grid += `<text x="${{LEFT2}}" y="${{TOP2 + 18}}" font-size="12" fill="#777">no inferred microblock tasks</text>`;
    }}
    micro_names.forEach((name, i) => {{
      const y = TOP2 + i * LANE_H2;
      const fill = (i % 2) ? '#f0f2f5' : '#ffffff';
      grid += `<rect x="${{LEFT2}}" y="${{y}}" width="${{W - LEFT2 - RIGHT2}}" height="${{LANE_H2}}" fill="${{fill}}"/>`;
      grid += `<text x="${{LEFT2 - 8}}" y="${{y + LANE_H2/2 + 5}}" text-anchor="end" font-size="12" fill="#333" font-weight="500">${{name}}</text>`;
    }});
    microGrid.innerHTML = grid;

    let bars = '';
    micro_names.forEach((name, i) => {{
      const lane = (micro.lanes || {{}})[name] || {{tasks: []}};
      const yy = TOP2 + i * LANE_H2 + LANE_PAD2;
      const hh = LANE_H2 - 2 * LANE_PAD2;
      for (const task of lane.tasks || []) {{
        const s = +task[0], e = +task[1];
        if (e <= view0 || s >= view1) continue;
        const x0 = Math.max(x(s), LEFT2);
        const x1 = Math.min(x(e), W - RIGHT2);
        const w = Math.max(1, x1 - x0);
        const label = task.length > 2 ? String(task[2]) : '';
        const bytes = task.length > 3 ? String(task[3]) : '';
        const phases = rtlPhaseText(task);
        const eng = task.length > 4 ? String(task[4]) : name;
        const color = ENG_COLORS[eng] || '#777';
        bars += `<rect class="gantt-task micro-task" x="${{x0}}" y="${{yy}}" width="${{w}}" height="${{hh}}" `
              + `fill="${{color}}" data-eng="micro ${{name}} / ${{eng}}" data-s="${{s}}" data-e="${{e}}" `
              + `data-label="${{esc(label)}}" data-bytes="${{esc(bytes)}}" data-phases="${{esc(phases)}}"/>`;
        bars += renderPhaseSegments(task, yy, hh, x, LEFT2, W - RIGHT2);
        if (w >= Math.min(160, label.length * 6 + 8)) {{
          bars += `<text x="${{x0 + 4}}" y="${{yy + hh/2 + 4}}" font-size="10" fill="#111" pointer-events="none">${{esc(label)}}</text>`;
        }}
      }}
    }});
    microBars.innerHTML = bars;

    let axis = '';
    const yAx = TOP2 + Math.max(1, N_MICRO_LANES) * LANE_H2;
    axis += `<line x1="${{LEFT2}}" y1="${{yAx}}" x2="${{W - RIGHT2}}" y2="${{yAx}}" stroke="#666"/>`;
    const niceStep = (rng) => {{
      const raw = rng / 10;
      const e10 = Math.pow(10, Math.floor(Math.log10(raw)));
      const m = raw / e10;
      const s = m < 1.5 ? 1 : m < 3 ? 2 : m < 7 ? 5 : 10;
      return s * e10;
    }};
    const step = niceStep(span);
    const t0 = Math.ceil(view0 / step) * step;
    for (let t = t0; t <= view1; t += step) {{
      const xx = x(t);
      axis += `<line x1="${{xx}}" y1="${{yAx}}" x2="${{xx}}" y2="${{yAx + 5}}" stroke="#666"/>`;
      axis += `<text x="${{xx}}" y="${{yAx + 18}}" font-size="11" text-anchor="middle" fill="#444">${{fmt(t)}}</text>`;
    }}
    microAxis.innerHTML = axis;
  }}

  function render() {{
    const W = svg.clientWidth || 1100;
    const Y_OPS  = TOP;
    const Y_ENG  = TOP + OP_ROW_H;                    // engine lanes start here
    const H = Y_ENG + N_LANES * LANE_H + AXIS_H;
    svg.setAttribute('viewBox', `0 0 ${{W}} ${{H}}`);
    svg.setAttribute('height', H);                    // 1:1 screen-y ↔ svg-y
    const span = Math.max(1, view1 - view0);
    const sx = (W - LEFT - RIGHT) / span;
    const x  = c => LEFT + (c - view0) * sx;

    // Lane backgrounds + engine labels (engine grid sits below the OP row).
    let grid = '';
    for (let i = 0; i < N_LANES; ++i) {{
      const y = Y_ENG + i * LANE_H;
      const fill = (i % 2) ? '#f0f2f5' : '#ffffff';
      grid += `<rect x="${{LEFT}}" y="${{y}}" width="${{W - LEFT - RIGHT}}" height="${{LANE_H}}" fill="${{fill}}"/>`;
      grid += `<text x="${{LEFT - 8}}" y="${{y + LANE_H/2 + 5}}" text-anchor="end" font-size="13" fill="#333" font-weight="500">${{eng_names[i]}}</text>`;
    }}
    gGrid.innerHTML = grid;

    // ----- OP ID row (one rect per layer with id + op name) -----
    let ops = '';
    ops += `<rect x="${{LEFT}}" y="${{Y_OPS}}" width="${{W - LEFT - RIGHT}}" height="${{OP_ROW_H}}" fill="#fafbfd" stroke="#e0e3e8" stroke-width="1"/>`;
    ops += `<text x="${{LEFT - 8}}" y="${{Y_OPS + OP_ROW_H/2 + 4}}" text-anchor="end" font-size="11" fill="#333" font-weight="500">op id</text>`;
    for (const L of data.layers) {{
      if (L.end <= view0 || L.start >= view1) continue;
      const x0 = Math.max(x(L.start), LEFT);
      const x1 = Math.min(x(L.end),   W - RIGHT);
      const w  = Math.max(1, x1 - x0);
      // Alternating fill stripes to make adjacent layers visually distinct.
      const fill = (L.id % 2) ? '#dde6f5' : '#eaf0fb';
      ops += `<rect class="gantt-op" x="${{x0}}" y="${{Y_OPS + 2}}" width="${{w}}" height="${{OP_ROW_H - 4}}" `
           + `fill="${{fill}}" stroke="#9bb5db" stroke-width="0.5" `
           + `data-id="${{L.id}}" data-flow="${{L.flow}}" data-op="${{L.op}}" data-start="${{L.start}}" data-end="${{L.end}}"/>`;
      // Label: two lines when there is room: layer/flow on top, op below.
      const label_top = `L${{L.id}} F${{L.flow}}`;
      const label_flow = `F${{L.flow}}`;
      const min_w_full = Math.max(label_top.length, String(L.op).length) * 6.5;
      const min_w_top  = label_top.length * 6.5;
      const min_w_flow = label_flow.length * 6.5;
      if (w >= min_w_full) {{
        ops += `<text x="${{x0 + 4}}" y="${{Y_OPS + 14}}" font-size="10" fill="#1a3766" pointer-events="none">${{label_top}}</text>`;
        ops += `<text x="${{x0 + 4}}" y="${{Y_OPS + 27}}" font-size="10" fill="#1a3766" pointer-events="none">${{L.op}}</text>`;
      }} else if (w >= min_w_top) {{
        ops += `<text x="${{x0 + 3}}" y="${{Y_OPS + OP_ROW_H/2 + 4}}" font-size="11" fill="#1a3766" pointer-events="none">${{label_top}}</text>`;
      }} else if (w >= min_w_flow) {{
        ops += `<text x="${{x0 + 3}}" y="${{Y_OPS + OP_ROW_H/2 + 4}}" font-size="11" fill="#1a3766" pointer-events="none">${{label_flow}}</text>`;
      }}
    }}
    gOps.innerHTML = ops;

    // Scope highlight band (if a search match is active and visible).
    let hi = '';
    if (highlight && highlight.end > view0 && highlight.start < view1) {{
      const x0 = Math.max(x(highlight.start), LEFT);
      const x1 = Math.min(x(highlight.end),   W - RIGHT);
      const w  = Math.max(1, x1 - x0);
      hi += `<rect x="${{x0}}" y="${{Y_ENG}}" width="${{w}}" height="${{N_LANES * LANE_H}}" `
          + `fill="rgba(255,221,0,0.18)" stroke="#e6b800" stroke-width="1"/>`;
    }}
    gScope.innerHTML = hi;

    // Layer boundaries (dashed verticals across engine lanes).
    let lay = '';
    for (const L of data.layers) {{
      if (L.end <= view0 || L.end >= view1) continue;
      const xx = x(L.end);
      lay += `<line x1="${{xx}}" y1="${{Y_ENG}}" x2="${{xx}}" y2="${{Y_ENG + N_LANES*LANE_H}}" stroke="#bbb" stroke-width="1" stroke-dasharray="3,3"/>`;
    }}
    gLayers.innerHTML = lay;

    // Tasks. Skip rects entirely outside the view; clip width to viewport.
    let bars = '';
    eng_names.forEach((name, i) => {{
      const lane = data.engines[name];
      const color = ENG_COLORS[name] || '#888';
      const yy = Y_ENG + i * LANE_H + LANE_PAD;
      const hh = LANE_H - 2 * LANE_PAD;
      for (const task of lane.tasks) {{
        const s = +task[0], e = +task[1];
        if (e <= view0 || s >= view1) continue;
        const x0 = Math.max(x(s), LEFT);
        const x1 = Math.min(x(e), W - RIGHT);
        const w  = Math.max(1, x1 - x0);
        const label = task.length > 2 ? String(task[2]) : '';
        const bytes = task.length > 3 ? String(task[3]) : '';
        const phases = rtlPhaseText(task);
        bars += `<rect class="gantt-task" x="${{x0}}" y="${{yy}}" width="${{w}}" height="${{hh}}" `
              + `fill="${{color}}" data-eng="${{name}}" data-s="${{s}}" data-e="${{e}}" `
              + `data-label="${{esc(label)}}" data-bytes="${{esc(bytes)}}" data-phases="${{esc(phases)}}"/>`;
        bars += renderPhaseSegments(task, yy, hh, x, LEFT, W - RIGHT);
      }}
    }});
    const convLaneIdx = eng_names.indexOf('conv');
    if (convLaneIdx >= 0 && data.conv_wait) {{
      const yy = Y_ENG + convLaneIdx * LANE_H + LANE_PAD;
      const hh = LANE_H - 2 * LANE_PAD;
      for (const task of data.conv_wait) {{
        const s = +task[0], e = +task[1];
        if (e <= view0 || s >= view1) continue;
        const x0 = Math.max(x(s), LEFT);
        const x1 = Math.min(x(e), W - RIGHT);
        const w  = Math.max(1, x1 - x0);
        const label = task.length > 2 ? String(task[2]) : '';
        const bytes = task.length > 3 ? String(task[3]) : '';
        bars += `<rect class="gantt-task gantt-wait" x="${{x0}}" y="${{yy}}" width="${{w}}" height="${{hh}}" `
              + `fill="#8a8a8a" opacity="0.85" data-eng="conv wait" data-s="${{s}}" data-e="${{e}}" `
              + `data-label="${{esc(label)}}" data-bytes="${{esc(bytes)}}"/>`;
      }}
    }}
    gBars.innerHTML = bars;

    let anchor = '';
    if (anchorCycle !== null && anchorCycle >= view0 && anchorCycle <= view1) {{
      const ax = x(anchorCycle);
      const y0 = Y_OPS;
      const y1 = Y_ENG + N_LANES * LANE_H;
      anchor += `<line x1="${{ax}}" y1="${{y0}}" x2="${{ax}}" y2="${{y1}}" `
             + `stroke="#111" stroke-width="1.5" stroke-dasharray="4,3" pointer-events="none"/>`;
      anchor += `<text x="${{Math.min(ax + 5, W - RIGHT - 90)}}" y="${{Y_OPS + 11}}" `
             + `font-size="10" fill="#111" pointer-events="none">anchor ${{fmt(anchorCycle)}}</text>`;
    }}
    gAnchor.innerHTML = anchor;

    // X-axis ticks.
    let axis = '';
    const yAx = Y_ENG + N_LANES * LANE_H;
    axis += `<line x1="${{LEFT}}" y1="${{yAx}}" x2="${{W - RIGHT}}" y2="${{yAx}}" stroke="#666"/>`;
    const niceStep = (rng) => {{
      const raw = rng / 10;
      const e10 = Math.pow(10, Math.floor(Math.log10(raw)));
      const m = raw / e10;
      const s = m < 1.5 ? 1 : m < 3 ? 2 : m < 7 ? 5 : 10;
      return s * e10;
    }};
    const step = niceStep(span);
    const t0 = Math.ceil(view0 / step) * step;
    for (let t = t0; t <= view1; t += step) {{
      const xx = x(t);
      axis += `<line x1="${{xx}}" y1="${{yAx}}" x2="${{xx}}" y2="${{yAx + 5}}" stroke="#666"/>`;
      axis += `<text x="${{xx}}" y="${{yAx + 18}}" font-size="11" text-anchor="middle" fill="#444">${{fmt(t)}}</text>`;
    }}
    gAxis.innerHTML = axis;

    windowInfo.textContent = `view: [${{fmt(view0)}}, ${{fmt(view1)}}] (${{fmt(span)}} cyc)`;
    renderMicro();
  }}

  // ----- Wheel -----
  // Above the x-axis: zoom around cursor (or pan with shift+wheel).
  // ON or BELOW the axis line: wheel pans horizontally (the axis row acts as
  // a scroll strip — same idea as a video timeline scrubber). Direction
  // matches deltaY: scrolling down → view moves right.
  const Y_AXIS_LINE = TOP + OP_ROW_H + N_LANES * LANE_H;
  svg.addEventListener('wheel', (ev) => {{
    const r = svg.getBoundingClientRect();
    const py = ev.clientY - r.top;
    const px = ev.clientX - r.left;
    const span = view1 - view0;

    if (py >= Y_AXIS_LINE || ev.shiftKey) {{
      // pan (scroll); deltaY's sign drives view direction.
      ev.preventDefault();
      const dx = ev.deltaY * span * 0.001;
      view0 = clamp(view0 + dx, 0, total - span);
      view1 = view0 + span;
    }} else {{
      // zoom around cursor.
      ev.preventDefault();
      if (px < LEFT) return;
      const W = svg.clientWidth;
      const cx = pxToCyc(px, W);
      const factor = Math.exp(ev.deltaY * 0.0015);
      let new_span = clamp(span * factor, 4, total);
      let new0 = clamp(cx - (cx - view0) * (new_span / span), 0, total - new_span);
      view0 = new0; view1 = new0 + new_span;
    }}
    render();
  }}, {{passive: false}});

  if (microSvg) microSvg.addEventListener('wheel', (ev) => {{
    ev.preventDefault();
    const r = microSvg.getBoundingClientRect();
    const W = microSvg.clientWidth || 1100;
    const px = ev.clientX - r.left;
    const LEFT2 = 92, RIGHT2 = 12;
    const span = view1 - view0;
    if (ev.shiftKey) {{
      const dx = ev.deltaY * span * 0.001;
      view0 = clamp(view0 + dx, 0, total - span);
      view1 = view0 + span;
    }} else {{
      const sx = (W - LEFT2 - RIGHT2) / span;
      const cx = view0 + (px - LEFT2) / sx;
      const factor = Math.exp(ev.deltaY * 0.0015);
      const new_span = clamp(span * factor, 4, total);
      const new0 = clamp(cx - (cx - view0) * (new_span / span), 0, total - new_span);
      view0 = new0; view1 = new0 + new_span;
    }}
    render();
  }}, {{passive: false}});

  // ----- Mouse-drag: zoom-to-range (default), or pan with alt/right-button -----
  let mode = 'idle';   // 'select' | 'pan'
  let startX = 0, startView0 = 0;

  function updateCursor(px, W) {{
    const cyc = Math.round(pxToCyc(px, W));
    let text = 'cycle: ' + fmt(cyc);
    if (anchorCycle !== null) {{
      const d = cyc - anchorCycle;
      text += ' | anchor: ' + fmt(anchorCycle)
           + ' | Δ ' + fmtSigned(d) + ' cyc / ' + fmtSignedMs(d) + ' ms';
    }}
    cursorInfo.textContent = text;
  }}

  function setAnchor(px, W) {{
    anchorCycle = Math.round(clamp(pxToCyc(px, W), 0, total));
    updateCursor(px, W);
    render();
  }}

  function showSel(x0, x1) {{
    const yy = TOP, hh = OP_ROW_H + N_LANES * LANE_H;
    const left = Math.min(x0, x1), w = Math.abs(x1 - x0);
    selRect.setAttribute('x', left);
    selRect.setAttribute('y', yy);
    selRect.setAttribute('width', w);
    selRect.setAttribute('height', hh);
    selRect.style.display = 'block';
  }}

  // Suppress the right-click context menu so right-button drag can pan freely.
  svg.addEventListener('contextmenu', (ev) => ev.preventDefault());

  svg.addEventListener('mousedown', (ev) => {{
    // Accept left-button (default = select; +alt = pan) AND right-button (= pan).
    if (ev.button !== 0 && ev.button !== 2) return;
    const r = svg.getBoundingClientRect();
    const px = ev.clientX - r.left;
    if (px < LEFT) return;
    startX = px;
    const isPan = (ev.button === 0 && ev.altKey) || ev.button === 2;
    if (isPan) {{
      mode = 'pan';
      startView0 = view0;
      svg.style.cursor = 'grabbing';
    }} else {{
      mode = 'select';
      svg.style.cursor = 'crosshair';
      showSel(px, px);
    }}
    ev.preventDefault();
  }});

  window.addEventListener('mousemove', (ev) => {{
    const r = svg.getBoundingClientRect();
    const W = svg.clientWidth;
    const px = clamp(ev.clientX - r.left, LEFT, W - RIGHT);

    if (mode === 'pan') {{
      const span = view1 - view0;
      const sx = (W - LEFT - RIGHT) / span;
      const dx = (ev.clientX - r.left - startX) / sx;
      const new0 = clamp(startView0 - dx, 0, total - span);
      view0 = new0; view1 = new0 + span;
      render();
    }} else if (mode === 'select') {{
      showSel(startX, px);
    }}

    if (px >= LEFT && px <= W - RIGHT) {{
      updateCursor(px, W);
    }}
  }});

  window.addEventListener('mouseup', (ev) => {{
    if (mode === 'select') {{
      const r = svg.getBoundingClientRect();
      const W = svg.clientWidth;
      const endX = clamp(ev.clientX - r.left, LEFT, W - RIGHT);
      const x0 = Math.min(startX, endX);
      const x1 = Math.max(startX, endX);
      // Min-drag threshold: 4 px → treat as a click, no zoom.
      if (x1 - x0 >= 4) {{
        const c0 = clamp(pxToCyc(x0, W), 0, total);
        const c1 = clamp(pxToCyc(x1, W), 0, total);
        if (c1 - c0 >= 1) {{ view0 = c0; view1 = c1; render(); }}
      }} else {{
        setAnchor(endX, W);
      }}
      selRect.style.display = 'none';
    }}
    mode = 'idle';
    svg.style.cursor = 'crosshair';
  }});

  // Tooltip on task / op-rect hover (uses bubbling so it works while drag is active).
  function handleTooltip(ev) {{
    if (mode !== 'idle') {{ tip.style.opacity = '0'; return; }}
    const t = ev.target;
    if (t && t.classList && t.classList.contains('gantt-task')) {{
      const s = +t.getAttribute('data-s');
      const e = +t.getAttribute('data-e');
      const label = t.getAttribute('data-label') || '';
      const bytes = +(t.getAttribute('data-bytes') || 0);
      const phases = t.getAttribute('data-phases') || '';
      let text = `${{t.getAttribute('data-eng')}}`;
      if (label) text += `\\n${{label}}`;
      text += `\\n${{fmt(s)}} → ${{fmt(e)}} cyc\\nΔ ${{fmt(e - s)}} cyc`;
      if (bytes) text += `\\n${{(bytes / 1024).toFixed(1)}} KB`;
      if (phases) text += `\\nrtl phases: ${{phases}}`;
      tip.textContent = text;
      tip.style.left = (ev.pageX + 12) + 'px';
      tip.style.top  = (ev.pageY + 12) + 'px';
      tip.style.opacity = '1';
    }} else if (t && t.classList && t.classList.contains('gantt-op')) {{
      const id  = t.getAttribute('data-id');
      const flow = t.getAttribute('data-flow');
      const op  = t.getAttribute('data-op');
      const s   = +t.getAttribute('data-start');
      const e   = +t.getAttribute('data-end');
      tip.textContent = `L${{id}} F${{flow}} ${{op}}\\n${{fmt(s)}} → ${{fmt(e)}} cyc\\nΔ ${{fmt(e - s)}} cyc\\n(click to zoom)`;
      tip.style.left = (ev.pageX + 12) + 'px';
      tip.style.top  = (ev.pageY + 12) + 'px';
      tip.style.opacity = '1';
    }} else {{
      tip.style.opacity = '0';
    }}
  }}
  svg.addEventListener('mousemove', handleTooltip);
  if (microSvg) microSvg.addEventListener('mousemove', handleTooltip);
  svg.addEventListener('mouseleave', () => {{ tip.style.opacity = '0'; }});
  if (microSvg) microSvg.addEventListener('mouseleave', () => {{ tip.style.opacity = '0'; }});

  // Click an op-rect → zoom to that layer's window.
  svg.addEventListener('click', (ev) => {{
    const t = ev.target;
    if (!(t && t.classList && t.classList.contains('gantt-op'))) return;
    const s = +t.getAttribute('data-start');
    const e = +t.getAttribute('data-end');
    const dur = Math.max(1, e - s);
    const pad = Math.max(50, dur * 0.1);
    view0 = Math.max(0, s - pad);
    view1 = Math.min(total, e + pad);
    if (view1 - view0 < 4) view1 = view0 + 4;
    highlight = {{start: s, end: e, id: t.getAttribute('data-id'), op: t.getAttribute('data-op')}};
    render();
  }});

  // ----- Pan / zoom helpers (used by buttons + keyboard) -----
  function panBy(frac) {{
    // frac > 0 → view moves right; frac is in units of the current span.
    const span = view1 - view0;
    const dx = span * frac;
    view0 = clamp(view0 + dx, 0, total - span);
    view1 = view0 + span;
  }}
  function zoomBy(factor) {{
    // factor < 1 → zoom in (smaller span);  > 1 → zoom out.
    const c = (view0 + view1) / 2;
    const new_span = clamp((view1 - view0) * factor, 4, total);
    view0 = clamp(c - new_span / 2, 0, total - new_span);
    view1 = view0 + new_span;
  }}

  // ----- Buttons -----
  const reset = () => {{ view0 = 0; view1 = total; highlight = null; render(); }};
  document.getElementById('gantt-reset').onclick   = reset;
  document.getElementById('gantt-zoomin').onclick  = () => {{ zoomBy(0.5); render(); }};
  document.getElementById('gantt-zoomout').onclick = () => {{ zoomBy(2.0); render(); }};
  svg.addEventListener('dblclick', reset);

  // ----- Keyboard navigation -----
  // ←/→ pan, +/- zoom around center, 0 = reset, Home/End = jump to start/end.
  // Disabled while focus is in the scope input (so typing isn't hijacked).
  window.addEventListener('keydown', (ev) => {{
    if (ev.target === scopeInput) return;
    if (ev.metaKey || ev.ctrlKey) return;          // leave OS / browser shortcuts alone
    const big = ev.shiftKey ? 0.5 : 0.1;           // shift+arrow = jump 50% of view
    let handled = true;
    switch (ev.key) {{
      case 'ArrowLeft':  panBy(-big);    break;
      case 'ArrowRight': panBy( big);    break;
      case '+': case '=':                        // '=' is the unshifted key on US layout
        zoomBy(ev.shiftKey ? 0.5 : 0.7); break;
      case '-': case '_':
        zoomBy(ev.shiftKey ? 2.0 : 1.4); break;
      case '0':          reset(); return;        // reset() already calls render()
      case 'Home': {{
        const span = view1 - view0;
        view0 = 0; view1 = Math.min(total, span);
        break;
      }}
      case 'End': {{
        const span = view1 - view0;
        view1 = total; view0 = Math.max(0, total - span);
        break;
      }}
      default: handled = false;
    }}
    if (handled) {{ ev.preventDefault(); render(); }}
  }});

  // ----- Scope-by-text-input search -----
  let matches = [];
  let matchIdx = -1;

  // Each match is either {{kind:'layer', L}} or {{kind:'range', lo, hi, count, start, end}}.
  function findMatches(q) {{
    q = q.trim().toLowerCase().replace(/\\u2013|\\u2014/g, '-');   // en-/em-dash → hyphen
    if (!q) return [];

    // Range: "3-6" / "L3-L6" / "L3-6" / "3-L6".
    const rng = q.match(/^l?(\\d+)\\s*-\\s*l?(\\d+)$/);
    if (rng) {{
      const a = parseInt(rng[1], 10), b = parseInt(rng[2], 10);
      const lo = Math.min(a, b), hi = Math.max(a, b);
      const La = data.layers.find(L => L.id === lo);
      const Lb = data.layers.find(L => L.id === hi);
      if (La && Lb) {{
        const count = data.layers.filter(L => L.id >= lo && L.id <= hi).length;
        return [{{kind: 'range', lo, hi, count,
                  start: La.start, end: Lb.end}}];
      }}
      return [];
    }}

    // Single id: "5" or "L5".
    const idM = q.match(/^l?(\\d+)$/);
    if (idM) {{
      const target = parseInt(idM[1], 10);
      const L = data.layers.find(L => L.id === target);
      return L ? [{{kind: 'layer', L}}] : [];
    }}

    // Substring on op name (returns one match per hit).
    return data.layers
      .filter(L => L.op.toLowerCase().includes(q))
      .map(L => ({{kind: 'layer', L}}));
  }}

  function showMatch() {{
    if (!matches.length) {{ highlight = null; render(); return; }}
    const m = matches[matchIdx];
    let s, e;
    if (m.kind === 'range') {{
      s = m.start; e = m.end;
      highlight = {{start: s, end: e, id: `${{m.lo}}–${{m.hi}}`, op: `${{m.count}} layers`}};
    }} else {{
      const L = m.L;
      s = L.start; e = L.end;
      highlight = L;
    }}
    const dur = Math.max(1, e - s);
    const pad = Math.max(50, dur * 0.1);
    view0 = Math.max(0,     s - pad);
    view1 = Math.min(total, e + pad);
    if (view1 - view0 < 4) {{ view1 = view0 + 4; }}
    render();
  }}

  function updateMatchUI() {{
    if (!matches.length) {{
      matchInfo.textContent = scopeInput.value.trim() ? 'no match' : '';
      matchInfo.style.color = scopeInput.value.trim() ? '#b00020' : '#666';
      return;
    }}
    const m = matches[matchIdx];
    let label;
    if (m.kind === 'range') {{
      label = `range L${{m.lo}}–L${{m.hi}} (${{m.count}} layers) [${{fmt(m.start)}}, ${{fmt(m.end)}}]`;
    }} else {{
      const L = m.L;
      label = `${{matchIdx + 1}}/${{matches.length}}: L${{L.id}} ${{L.op}} [${{fmt(L.start)}}, ${{fmt(L.end)}}]`;
    }}
    matchInfo.textContent = label;
    matchInfo.style.color = '#0a7d23';
  }}

  scopeInput.addEventListener('input', () => {{
    matches = findMatches(scopeInput.value);
    matchIdx = matches.length ? 0 : -1;
    updateMatchUI();
    if (matches.length) showMatch();
    else {{ highlight = null; render(); }}
  }});
  scopeInput.addEventListener('keydown', (ev) => {{
    if (!matches.length) return;
    if (ev.key === 'Enter' || ev.key === 'ArrowDown' || ev.key === 'ArrowRight') {{
      matchIdx = (matchIdx + 1) % matches.length;
      ev.preventDefault();
      showMatch(); updateMatchUI();
    }} else if (ev.key === 'ArrowUp' || ev.key === 'ArrowLeft') {{
      matchIdx = (matchIdx - 1 + matches.length) % matches.length;
      ev.preventDefault();
      showMatch(); updateMatchUI();
    }}
  }});
  document.getElementById('gantt-prev').onclick = () => {{
    if (!matches.length) return;
    matchIdx = (matchIdx - 1 + matches.length) % matches.length;
    showMatch(); updateMatchUI();
  }};
  document.getElementById('gantt-next').onclick = () => {{
    if (!matches.length) return;
    matchIdx = (matchIdx + 1) % matches.length;
    showMatch(); updateMatchUI();
  }};

  window.addEventListener('resize', render);
  render();
}})();
</script>

<h2>Per-layer profile (sizes in KB)</h2>
<input class="filter" data-target="profile-tbl" type="search" placeholder="filter rows… (substring match)" autocomplete="off"/>
<span class="filter-info" data-info="profile-tbl"></span>
<table id="profile-tbl" class="sortable"><thead><tr>
<th>id</th><th>flow</th><th>op</th><th>engine</th>
<th>cyc/layer</th><th>cyc/cum</th>
<th>ideal<br>cyc/layer</th><th>ideal<br>cyc/cum</th>
<th>conv<br>MAC util</th><th>conv<br>occupancy</th>
<th>DRAM r</th><th>DRAM w</th><th>SRAM r</th><th>SRAM w</th>
<th>iH</th><th>iW</th><th>iC</th><th>oH</th><th>oW</th><th>oC</th>
<th>kH</th><th>kW</th><th>sH</th><th>sW</th><th>group</th>
<th>tiles<br>(H×OC)</th>
<th>verify</th>
</tr></thead>
<tbody>{layer_rows}</tbody></table>

<h2>Compile log ({n_compile} layers, {n_skipped_compile} skipped)</h2>
<input class="filter" data-target="compile-tbl" type="search" placeholder="filter rows… (substring match)" autocomplete="off"/>
<span class="filter-info" data-info="compile-tbl"></span>
<table id="compile-tbl" class="sortable"><thead><tr>
<th>id</th><th>op</th>
<th>iH</th><th>iW</th><th>iC</th>
<th>kH</th><th>kW</th><th>sH</th><th>sW</th><th>group</th>
<th>oH</th><th>oW</th><th>oC</th>
<th>elements</th><th>dtype</th><th>status</th>
</tr></thead>
<tbody>{compile_rows}</tbody></table>

<script>
(() => {{
  // Click a header to sort the table by that column. Numeric columns sort
  // numerically (commas / "%" / KB suffixes stripped); otherwise lexicographic.
  // Three-state: asc -> desc -> reset to original document order.
  const numClean = s => s.replace(/[,%]/g, '').replace(/[^\\d.\\-eE+]/g, '');
  function sortTable(tbl, idx, dir) {{
    const tbody = tbl.tBodies[0];
    const orig = tbl._origRows ||= Array.from(tbody.rows);
    const rows = (dir === 0) ? orig.slice() : orig.slice();
    if (dir !== 0) {{
      const allNum = rows.every(r => {{
        const t = (r.cells[idx]?.textContent || '').trim();
        if (t === '' || t === '—') return true;
        return !isNaN(parseFloat(numClean(t)));
      }});
      rows.sort((a, b) => {{
        const at = (a.cells[idx]?.textContent || '').trim();
        const bt = (b.cells[idx]?.textContent || '').trim();
        const aEmpty = (at === '' || at === '—');
        const bEmpty = (bt === '' || bt === '—');
        if (aEmpty && !bEmpty) return  1;
        if (bEmpty && !aEmpty) return -1;
        if (aEmpty &&  bEmpty) return  0;
        let cmp;
        if (allNum) {{
          cmp = (parseFloat(numClean(at)) || 0) - (parseFloat(numClean(bt)) || 0);
        }} else {{
          cmp = at.localeCompare(bt, undefined, {{numeric: true}});
        }}
        return dir > 0 ? cmp : -cmp;
      }});
    }}
    tbody.replaceChildren(...rows);
  }}
  document.querySelectorAll('table.sortable').forEach(tbl => {{
    const ths = tbl.tHead.querySelectorAll('th');
    let cur = {{ idx: -1, dir: 0 }};
    ths.forEach((th, idx) => {{
      th.addEventListener('click', () => {{
        let nextDir;
        if (cur.idx !== idx) nextDir = 1;
        else if (cur.dir === 1) nextDir = -1;
        else nextDir = 0;
        cur = {{ idx, dir: nextDir }};
        ths.forEach(t => t.removeAttribute('data-sort'));
        if (nextDir === 1)  th.setAttribute('data-sort', 'asc');
        if (nextDir === -1) th.setAttribute('data-sort', 'desc');
        sortTable(tbl, idx, nextDir);
      }});
    }});
  }});
  // Substring filter, runs against the row's full text.
  document.querySelectorAll('input.filter').forEach(inp => {{
    const tbl  = document.getElementById(inp.dataset.target);
    const info = document.querySelector(`.filter-info[data-info="${{inp.dataset.target}}"]`);
    inp.addEventListener('input', () => {{
      const q = inp.value.trim().toLowerCase();
      const rows = Array.from(tbl.tBodies[0].rows);
      let visible = 0;
      rows.forEach(r => {{
        const show = !q || r.textContent.toLowerCase().includes(q);
        r.style.display = show ? '' : 'none';
        if (show) ++visible;
      }});
      info.textContent = q ? `(${{visible}} / ${{rows.length}} match)` : '';
    }});
  }});
}})();
</script>
</body></html>
"""
    paths["html"].write_text(html_doc)
    return paths["html"]

# ---- SystemC compile/sim helpers ----

# v8.25: mdla7_model_runner.cpp prints "sim time: <cycles> cycles @ 1.9 GHz (= <ms> ms)"
# (pre-v8.25 was "sim time: <ns> ns"). Match both for forward/back compat.
SIM_TIME_RE = re.compile(r"sim time:\s*([\d,]+)\s*(?:cycles|ns)")

# SystemC license banner on stderr (pretty-printed at startup); suppress it
# when picking a meaningful error line out of stderr.
_BANNER_HINTS = ("ALL RIGHTS RESERVED", "Accellera", "Copyright (c)",
                 "ISO/IEC", "SystemC ", "Licensed under")

def _refresh_model_profile_index() -> None:
    try:
        subprocess.run([sys.executable, str(MODEL_PROFILE_PY),
                        "--html-out", "profile/profile_mdla6_pattern.html",
                        "--title", "MDLA7 MDLA6 Pattern Profiles",
                        "--only-metrics-rows"],
                       cwd=str(HERE), capture_output=True, text=True)
    except Exception:
        pass


def _meaningful_stderr_line(stderr: str) -> str:
    """Pick the most informative line from stderr — explicit `layer N: ...`
    errors win; otherwise the last non-banner non-empty line."""
    lines = [ln.strip() for ln in (stderr or "").splitlines() if ln.strip()]
    for ln in reversed(lines):
        if re.match(r"layer\s+\d+:", ln):
            return ln
    for ln in reversed(lines):
        if any(h in ln for h in _BANNER_HINTS):
            continue
        return ln
    return ""


def _normalise_pattern(pat: str) -> str:
    """Strip `.cut` suffix and fix known typos in the MDLA6 sheet."""
    # MDLA6 used truncated graphs for some transformer-class models;
    # MDLA7 runs the full .tflite (compile_model skips unsupported ops).
    if pat.endswith(".cut"):
        pat = pat[: -len(".cut")]
    # Typo: double underscore in esrgan__int16.
    pat = pat.replace("__", "_")
    return pat


def _normalise_engine(engine_model: str) -> str:
    if engine_model not in ("fast", "rtl", "cx"):
        raise ValueError(f"unknown engine mode: {engine_model}")
    return engine_model


def _normalise_l1(l1_timing: str) -> str:
    if l1_timing not in ("fast", "rtl", "cx"):
        raise ValueError(f"unknown L1 mode: {l1_timing}")
    return l1_timing


def _mode_suffix(l1_timing: str = "fast", engine_model: str = "fast") -> str:
    l1_timing = _normalise_l1(l1_timing)
    engine_model = _normalise_engine(engine_model)
    if l1_timing == "fast" and engine_model == "fast":
        return ""
    if l1_timing == engine_model:
        return l1_timing
    suffixes: list[str] = []
    if l1_timing != "fast":
        suffixes.append("cx-l1" if l1_timing == "cx" else f"L1-{l1_timing}")
    if engine_model != "fast":
        suffixes.append(engine_model)
    return ".".join(suffixes)


def _profile_mode_suffix(l1_timing: str = "fast", engine_model: str = "fast") -> str:
    return _mode_suffix(l1_timing, engine_model)


def _mode_display(l1_timing: str = "fast", engine_model: str = "fast") -> str:
    l1_timing = _normalise_l1(l1_timing)
    engine_model = _normalise_engine(engine_model)
    if l1_timing == engine_model:
        return l1_timing
    if l1_timing == "fast":
        return engine_model
    if engine_model == "fast":
        return f"{l1_timing}-l1"
    return f"{l1_timing}/{engine_model}"


def _arg_was_set(*names: str) -> bool:
    for arg in sys.argv[1:]:
        for name in names:
            if arg == name or arg.startswith(f"{name}="):
                return True
    return False


def _report_exists_for(pattern: str, model_dir: Path, engine_model: str = "fast",
                       l1_timing: str = "fast") -> bool:
    l1_timing = _normalise_l1(l1_timing)
    engine_model = _normalise_engine(engine_model)
    canonical = _normalise_pattern(pattern)
    model_path = model_dir / f"{canonical}.tflite"
    suffix = _mode_suffix(l1_timing, engine_model)
    if suffix:
        return _mode_paths(model_path, suffix)["html"].exists()
    return _artefact_paths(model_path)["html"].exists()


def _mode_paths(model_path: Path, mode: str) -> dict[str, Path]:
    paths = _artefact_paths(model_path)
    if mode == "fast":
        return paths
    stem = model_path.stem
    file_stem = stem if OUT_DIR.name == mode else f"{stem}.{mode}"
    return {
        "prog":  OUT_DIR / f"{file_stem}.bin",
        "prof":  OUT_DIR / f"{file_stem}.profile.json",
        "csv":   OUT_DIR / f"{file_stem}.profile.csv",
        "gantt": OUT_DIR / f"{file_stem}.profile.png",
        "html":  OUT_DIR / f"{file_stem}.html",
    }


def _selected_log_lines(compile_stdout: str, sim_stdout: str) -> list[str]:
    lines: list[str] = []
    for ln in (compile_stdout or "").splitlines():
        if ln.startswith(("compile_model:", "  layer", "  →")):
            lines.append(ln)
    for ln in (sim_stdout or "").splitlines():
        if ln.startswith(("mdla7_model_runner:", "test_model:", "  layer", "  summary",
                          "  sim time", "  DRAM ", "  SRAM ", "  L1Mesh ",
                          "  per-engine", "  utilization", "    ",
                          "  profile", "  csv")):
            lines.append(ln)
    return lines


def _write_mode_html(model_path: Path, paths: dict[str, Path],
                     compile_stdout: str, sim_stdout: str,
                     mode_label: str = "") -> Path:
    if not paths["prof"].exists():
        raise RuntimeError(f"profile missing: {paths['prof'].name}")

    gr = subprocess.run(
        [sys.executable, str(PLOT_PY), str(paths["prof"]), "-o", str(paths["gantt"])],
        capture_output=True, text=True, timeout=120,
    )
    if gr.returncode != 0:
        msg = _meaningful_stderr_line(gr.stderr) or f"exit {gr.returncode}"
        raise RuntimeError(f"gantt-fail: {msg[:80]}")

    return _write_html_report(model_path, paths,
                              _selected_log_lines(compile_stdout, sim_stdout),
                              mode_label)


def _write_combined_html(model_path: Path,
                         fast_html: Path,
                         conflict_html: Path,
                         mesh_html: Path,
                         fast_ms: float | None,
                         conflict_ms: float | None,
                         mesh_ms: float | None,
                         fast_status: str,
                         conflict_status: str,
                         mesh_status: str) -> Path:
    paths = _artefact_paths(model_path)
    conflict_ratio = ""
    if fast_ms and conflict_ms is not None:
        conflict_ratio = f"{conflict_ms / fast_ms:.3f}x"
    mesh_ratio = ""
    if fast_ms and mesh_ms is not None:
        mesh_ratio = f"{mesh_ms / fast_ms:.3f}x"
    mesh_conflict_ratio = ""
    if conflict_ms and mesh_ms is not None:
        mesh_conflict_ratio = f"{mesh_ms / conflict_ms:.3f}x"
    fast_doc = fast_html.read_text(errors="ignore") if fast_html.exists() else ""
    conflict_doc = conflict_html.read_text(errors="ignore") if conflict_html.exists() else ""
    mesh_doc = mesh_html.read_text(errors="ignore") if mesh_html.exists() else ""
    def ms(v: float | None) -> str:
        return f"{v:.3f} ms" if v is not None else ""
    doc = f"""<!doctype html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MDLA7 profile — {html.escape(model_path.name)} — fast/conflict/mesh</title>
<style>
body {{ margin:0; font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
       color:#222; background:#f6f7f9; }}
header {{ position:sticky; top:0; z-index:10; background:#fff; border-bottom:1px solid #d8dde6;
          padding:12px 18px; box-shadow:0 1px 4px rgba(0,0,0,.04); }}
h1 {{ margin:0 0 8px; font-size:18px; }}
.summary {{ display:flex; flex-wrap:wrap; gap:10px 18px; color:#495464; font-size:12px; }}
.summary b {{ color:#222; }}
.tabs {{ margin-top:10px; display:flex; gap:8px; }}
button {{ border:1px solid #c8d0dc; background:#fff; border-radius:5px; padding:5px 10px; cursor:pointer; }}
button.active {{ background:#1f5fa8; color:#fff; border-color:#1f5fa8; }}
.pane {{ display:none; padding:0; }}
.pane.active {{ display:block; }}
iframe {{ width:100%; height:calc(100vh - 120px); border:0; background:#fff; display:block; }}
</style></head>
<body>
<header>
  <h1>{html.escape(model_path.name)} — fast/conflict/mesh profile</h1>
  <div class="summary">
    <span><b>fast:</b> {html.escape(ms(fast_ms))} {html.escape(fast_status)}</span>
    <span><b>conflict:</b> {html.escape(ms(conflict_ms))} {html.escape(conflict_status)}</span>
    <span><b>mesh:</b> {html.escape(ms(mesh_ms))} {html.escape(mesh_status)}</span>
    <span><b>conflict/fast:</b> {html.escape(conflict_ratio)}</span>
    <span><b>mesh/fast:</b> {html.escape(mesh_ratio)}</span>
    <span><b>mesh/conflict:</b> {html.escape(mesh_conflict_ratio)}</span>
  </div>
  <div class="tabs">
    <button class="tab active" data-target="fast">fast</button>
    <button class="tab" data-target="conflict">conflict</button>
    <button class="tab" data-target="mesh">mesh</button>
  </div>
</header>
<section id="fast" class="pane active">
  <iframe title="fast profile" srcdoc="{html.escape(fast_doc, quote=True)}"></iframe>
</section>
<section id="conflict" class="pane">
  <iframe title="conflict profile" srcdoc="{html.escape(conflict_doc, quote=True)}"></iframe>
</section>
<section id="mesh" class="pane">
  <iframe title="mesh profile" srcdoc="{html.escape(mesh_doc, quote=True)}"></iframe>
</section>
<script>
document.querySelectorAll('.tab').forEach(btn => btn.addEventListener('click', () => {{
  document.querySelectorAll('.tab').forEach(b => b.classList.toggle('active', b === btn));
  document.querySelectorAll('.pane').forEach(p => p.classList.toggle('active', p.id === btn.dataset.target));
}}));
</script>
</body></html>
"""
    paths["html"].write_text(doc)
    return paths["html"]


def _simulate_one(bin_path: Path, l1_timing: str,
                  engine_model: str = "fast") -> tuple[float | None, str, str]:
    """Run mdla7_model_runner once. Returns (ms, status, stdout)."""
    engine_model = _normalise_engine(engine_model)
    try:
        sr = subprocess.run(
            [str(MODEL_RUNNER), str(bin_path), "--quiet",
             f"--L1={l1_timing}", f"--engine={engine_model}"],
            capture_output=True, text=True, timeout=900,
        )
    except subprocess.TimeoutExpired:
        return None, "sim-timeout", ""
    m = SIM_TIME_RE.search(sr.stdout or "")
    if sr.returncode != 0 and not m:
        last = _meaningful_stderr_line(sr.stderr) or f"exit {sr.returncode}"
        return None, f"sim-fail: {last[:80]}", sr.stdout or ""
    if not m:
        return None, "sim-time-missing", sr.stdout or ""
    cycles = int(m.group(1).replace(",", ""))
    ms = cycles / 1.9e6   # spec frequency = 1.9 GHz (post-v8.25)

    sm = re.search(r"summary:\s+(\d+)/(\d+)\s+layers PASS,\s+(\d+)\s+FAIL",
                   sr.stdout or "")
    if sm:
        _, _, n_fail = (int(sm.group(k)) for k in (1, 2, 3))
        status = "ok" if n_fail == 0 else f"{n_fail}-FAIL"
    else:
        status = "ok" if sr.returncode == 0 else f"exit {sr.returncode}"
    return ms, status, sr.stdout or ""


def run_one(pattern: str, model_dir: Path, progress=None,
            fast_only: bool = False,
            skip_html: bool = False,
            engine_model: str = "rtl",
            l1_timing: str = "rtl") -> tuple[str, float | None, float | None, float | None, str, str, str]:
    """Compile + simulate one model; optionally skip conflict/mesh."""
    l1_timing = _normalise_l1(l1_timing)
    engine_model = _normalise_engine(engine_model)
    if l1_timing != "fast":
        fast_only = True
    canonical  = _normalise_pattern(pattern)
    model_path = model_dir / f"{canonical}.tflite"
    if not model_path.exists():
        return pattern, None, None, None, f"missing-tflite: {model_path.name}", "", ""
    suffix = _mode_suffix(l1_timing, engine_model)
    paths = _artefact_paths(model_path) if not suffix else _mode_paths(model_path, suffix)
    bin_path = paths["prog"]
    conflict_suffix = "conflict" if engine_model == "fast" else f"{engine_model}.conflict"
    mesh_suffix = "mesh" if engine_model == "fast" else f"{engine_model}.mesh"
    conflict_paths = _mode_paths(model_path, conflict_suffix)
    mesh_paths = _mode_paths(model_path, mesh_suffix)

    # ---- compile ----
    if progress:
        progress("compile")
    try:
        cr = subprocess.run(
            [sys.executable, str(COMPILE_PY), str(model_path), str(bin_path)],
            capture_output=True, text=True, timeout=600,
        )
    except subprocess.TimeoutExpired:
        return pattern, None, None, None, "compile-timeout", "", ""
    if cr.returncode != 0:
        last = _meaningful_stderr_line(cr.stderr) or f"exit {cr.returncode}"
        if "Model provided has model identifier" in last:
            last = "corrupt .tflite"
        return pattern, None, None, None, f"compile-fail: {last[:80]}", "", ""

    # ---- simulate/report path ----
    if progress:
        progress(f"simulate {_mode_display(l1_timing, engine_model)}")
    ms, status, fast_stdout = _simulate_one(bin_path, l1_timing, engine_model=engine_model)

    fast_html = OUT_DIR / f"{canonical}.fast.html" if not suffix else paths["html"]
    if not skip_html:
        if progress:
            progress(f"html {_mode_display(l1_timing, engine_model)}")
        try:
            _write_mode_html(model_path, paths, cr.stdout or "", fast_stdout,
                             _mode_display(l1_timing, engine_model))
            if paths["html"].exists() and paths["html"] != fast_html:
                fast_html.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(paths["html"], fast_html)
        except Exception as e:
            if status == "ok":
                status = f"html-fail: {str(e)[:80]}"
            else:
                status = f"{status}; html-fail"
    status = _status_with_compile_skips(status, cr.stdout or "")

    if fast_only:
        return pattern, ms, None, None, status, "", ""

    # ---- simulate: conflict timing only ----
    if progress:
        progress("simulate conflict")
    try:
        shutil.copyfile(bin_path, conflict_paths["prog"])
    except OSError:
        pass
    conflict_ms, conflict_status, conflict_stdout = _simulate_one(
        conflict_paths["prog"], "conflict", engine_model=engine_model)

    conflict_html = conflict_paths["html"]
    if not skip_html:
        if progress:
            progress("html conflict")
        try:
            _write_mode_html(model_path, conflict_paths, cr.stdout or "", conflict_stdout,
                             "conflict")
        except Exception as e:
            if conflict_status == "ok":
                conflict_status = f"html-fail: {str(e)[:80]}"
            else:
                conflict_status = f"{conflict_status}; html-fail"
    conflict_status = _status_with_compile_skips(conflict_status, cr.stdout or "")

    # ---- simulate: mesh timing + report ----
    if progress:
        progress("simulate mesh")
    try:
        shutil.copyfile(bin_path, mesh_paths["prog"])
    except OSError:
        pass
    mesh_ms, mesh_status, mesh_stdout = _simulate_one(
        mesh_paths["prog"], "mesh", engine_model=engine_model)

    mesh_html = mesh_paths["html"]
    if not skip_html:
        if progress:
            progress("html mesh")
        try:
            _write_mode_html(model_path, mesh_paths, cr.stdout or "", mesh_stdout,
                             "mesh")
        except Exception as e:
            if mesh_status == "ok":
                mesh_status = f"html-fail: {str(e)[:80]}"
            else:
                mesh_status = f"{mesh_status}; html-fail"
    mesh_status = _status_with_compile_skips(mesh_status, cr.stdout or "")

    if not skip_html:
        if progress:
            progress("html combined")
        try:
            if fast_html.exists() and conflict_html.exists() and mesh_html.exists():
                _write_combined_html(model_path, fast_html, conflict_html, mesh_html,
                                     ms, conflict_ms, mesh_ms,
                                     status, conflict_status, mesh_status)
        except Exception as e:
            if status == "ok":
                status = f"html-fail: combined {str(e)[:70]}"
    return pattern, ms, conflict_ms, mesh_ms, status, conflict_status, mesh_status


# ---- Corpus regression runner ----

def _load_prior_csv(csv_path: Path) -> dict[str, dict]:
    if not csv_path.exists():
        return {}
    out: dict[str, dict] = {}
    try:
        with csv_path.open(newline="") as f:
            for row in csv.DictReader(f):
                if row.get("pattern"):
                    if "cx_ms" in row and "mdla6_cx_ms" not in row:
                        row["mdla6_cx_ms"] = row.get("cx_ms", "")
                    if "rtl_over_cx" in row and "rtl_over_mdla6_cx" not in row:
                        row["rtl_over_mdla6_cx"] = row.get("rtl_over_cx", "")
                    if "synth_over_cx" in row and "cx_over_mdla6_cx" not in row:
                        row["cx_over_mdla6_cx"] = row.get("synth_over_cx", "")
                    out[row["pattern"]] = row
    except Exception:
        return {}
    return out


def _load_prior_results(csv_path: Path, fast_only: bool = False) -> dict[str, dict]:
    rows = {p: r for p, r in _load_prior_csv(csv_path).items()
            if r.get("status") == "ok" and r.get("mdla7_ms")}
    return rows


def _load_cached_result_from_artefacts(pattern: str,
                                       model_dir: Path,
                                       *,
                                       engine_model: str = "fast",
                                       l1_timing: str = "fast",
                                       require_html: bool = True) -> dict[str, str] | None:
    canonical = _normalise_pattern(pattern)
    model_path = model_dir / f"{canonical}.tflite"
    suffix = _mode_suffix(l1_timing, engine_model)
    paths = _mode_paths(model_path, suffix) if suffix else _artefact_paths(model_path)
    if require_html and not paths["html"].exists():
        return None
    if not paths["prof"].exists():
        return None
    try:
        with paths["prof"].open() as f:
            profile = json.load(f)
        summary = profile.get("summary", {}) or {}
        cycles = int(summary.get("total_cycles", 0) or 0)
        n_fail = int(summary.get("fail", 0) or 0)
    except Exception:
        return None
    if cycles <= 0 or n_fail != 0:
        return None
    status = "ok"
    if paths["html"].exists():
        try:
            text = paths["html"].read_text(errors="ignore")
            skip_match = re.search(
                r"Graph coverage:</b>\s*INCOMPLETE\s+—\s*(\d+)/", text)
            if skip_match is None:
                skip_match = re.search(r"Compile log \(\d+ layers,\s*(\d+) skipped\)", text)
            if skip_match and int(skip_match.group(1)) > 0:
                status = f"compile-skipped:{int(skip_match.group(1))}"
        except Exception:
            pass
    return {
        "pattern": canonical,
        "mdla7_ms": f"{cycles / 1.9e6:.3f}",
        "status": status,
    }


def _raise_if_not_clean(rows: list[dict], *, context: str = "run_systemc") -> None:
    bad = [
        (str(row.get("pattern", "")), str(row.get("status", "")))
        for row in rows
        if row.get("status") != "ok"
    ]
    if not bad:
        return
    print(f"[{context}] FAIL: {len(bad)}/{len(rows)} rows are not clean ok", flush=True)
    for pattern, status in bad[:12]:
        print(f"  {pattern}: {status}", flush=True)
    if len(bad) > 12:
        print(f"  ... {len(bad) - 12} more", flush=True)
    raise SystemExit(1)


def _discover_models(model_dir: Path, name_filter: str, recursive: bool = False) -> list[str]:
    if not model_dir.exists():
        raise SystemExit(f"model dir not found: {model_dir}")
    needle = name_filter.lower()
    patterns = []
    globber = model_dir.rglob if recursive else model_dir.glob
    for path in sorted(globber("*.tflite")):
        if path.name.startswith("._"):
            continue
        pattern = path.relative_to(model_dir).with_suffix("").as_posix()
        if needle and needle not in pattern.lower():
            continue
        patterns.append(pattern)
    return patterns


def _builtin_pattern_order(name: str) -> dict[str, tuple[float, int]]:
    if name != "ethz_v6":
        return {}
    return {
        _normalise_pattern(pattern): (mdla6_cx, idx)
        for idx, (pattern, mdla6_cx) in enumerate(ETHZ_V6_MDLA6_CX)
    }


def _load_pattern_order(source: Path | str) -> dict[str, tuple[float, int]]:
    if isinstance(source, str):
        return _builtin_pattern_order(source)
    csv_path = source
    if not csv_path.exists():
        raise SystemExit(f"pattern order CSV not found: {csv_path}")
    out: dict[str, tuple[float, int]] = {}
    with csv_path.open(newline="") as f:
        for idx, row in enumerate(csv.DictReader(f)):
            pattern = row.get("Pattern") or row.get("pattern") or ""
            pattern = _normalise_pattern(pattern.strip())
            if not pattern or pattern in out:
                continue
            try:
                mdla6_cx = float(row.get("mdla6_cx") or row.get("CX") or
                                  row.get("cx") or "inf")
            except ValueError:
                mdla6_cx = float("inf")
            out[pattern] = (mdla6_cx, idx)
    return out


def _apply_pattern_order(patterns: list[str], order_source: Path | str | None) -> list[str]:
    if not order_source:
        return patterns
    order = _load_pattern_order(order_source)

    def key(item: tuple[int, str]) -> tuple[int, float, int, str]:
        original_idx, pattern = item
        mdla6_order = order.get(_normalise_pattern(pattern))
        if mdla6_order:
            mdla6_cx, csv_idx = mdla6_order
            return (0, mdla6_cx, csv_idx, pattern)
        return (1, float("inf"), original_idx, pattern)

    return [pattern for _, pattern in sorted(enumerate(patterns), key=key)]


def _refresh_profile_index(title: str,
                           html_out: str,
                           csv_path: Path,
                           *,
                           show_mdla6_cx: bool = False,
                           primary_label: str = "fast",
                           ratio_label: str = "f/mdla6_cx",
                           report_suffix: str = "") -> None:
    try:
        html_path = Path(html_out)
        if not html_path.is_absolute():
            html_path = HERE / html_path
        if html_path.parent == HERE:
            html_out = f"profile/{html_path.name}"
        subprocess.run(
            [sys.executable, str(MODEL_PROFILE_PY),
             "--html-out", html_out,
             "--title", title,
             "--metrics-csv", str(csv_path),
             "--only-metrics-rows",
             "--hide-mode-columns",
             "--primary-label", primary_label,
             "--ratio-label", ratio_label,
             *(["--report-suffix", report_suffix] if report_suffix else []),
             *([] if show_mdla6_cx else ["--hide-mdla6-cx"])],
            cwd=str(HERE), capture_output=True, text=True,
        )
    except Exception:
        pass


def _profile_title_for_mode(title: str, primary_label: str) -> str:
    if primary_label == "cx":
        return f"{title} （CX)"
    return title


def _ms_cell(value: str) -> str:
    return f"{float(value):>8.2f} ms" if value else f"{'—':>8s}    "


def _ratio_cell(value: str) -> str:
    return f"{float(value):>6.2f}" if value else f"{'':>6s}"


def _ms_value_cell(value: str | float | None) -> str:
    if value in (None, ""):
        return ""
    return f"{float(value):.3f}"


def _ratio_from_ms(num: float | str | None, den: float | str | None) -> str:
    if num is None or den in (None, 0):
        return ""
    if num == "" or den == "":
        return ""
    den_f = float(den)
    if den_f <= 0:
        return ""
    return f"{float(num) / den_f:.3f}"


def _format_ms(value: float | None) -> str:
    return f"{value:.3f}" if value is not None else ""


def _load_mdla6_cx_ms(source: Path | str | None) -> dict[str, float]:
    if not source:
        return {}
    if isinstance(source, str):
        return {
            pattern: mdla6_cx
            for pattern, (mdla6_cx, _) in _builtin_pattern_order(source).items()
        }
    csv_path = source
    if not csv_path.exists():
        return {}
    out: dict[str, float] = {}
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            pattern = row.get("Pattern") or row.get("pattern") or ""
            pattern = _normalise_pattern(pattern.strip())
            if not pattern:
                continue
            value = row.get("mdla6_cx") or row.get("CX") or row.get("cx") or ""
            try:
                out[pattern] = float(value)
            except ValueError:
                continue
    return out


def _fit_cell(value: str, width: int = 30) -> str:
    if len(value) <= width:
        return f"{value:<{width}s}"
    keep = width - 1
    left = keep // 2
    right = keep - left
    return f"{value[:left]}…{value[-right:]}"


def _microblock_metrics_for(model_path: Path, l1_timing: str = "fast",
                            engine_model: str = "fast") -> dict[str, str]:
    suffix = _mode_suffix(l1_timing, engine_model)
    paths = _mode_paths(model_path, suffix) if suffix else _artefact_paths(model_path)
    prof = paths["prof"]
    if not prof.exists():
        return {
            "fuse_hit": "no",
            "fuse_flows": "",
            "streamed_layers": "",
            "mb_hit": "no",
            "mb_count": "0",
            "mb_layers": "",
            "mb_stages": "",
        }
    try:
        data = json.loads(prof.read_text())
    except Exception:
        return {
            "fuse_hit": "profile-error",
            "fuse_flows": "",
            "streamed_layers": "",
            "mb_hit": "profile-error",
            "mb_count": "0",
            "mb_layers": "",
            "mb_stages": "",
        }

    layers_json = data.get("layers") or []
    flow_members: dict[int, list[int]] = {}
    streamed_layers: list[int] = []
    for idx, layer in enumerate(layers_json):
        if not isinstance(layer, dict):
            continue
        lid = int(layer.get("id", idx) or idx)
        flow = int(layer.get("flow", lid) if layer.get("flow") is not None else lid)
        flow_members.setdefault(flow, []).append(lid)
        if layer.get("streamed"):
            streamed_layers.append(lid)
    fused_flows = [
        f"F{flow}:" + "+".join(f"L{x}" for x in members)
        for flow, members in sorted(flow_members.items())
        if len(members) > 1
    ]

    task_meta = data.get("task_meta") or {}
    seen: set[tuple[int, int]] = set()
    layers: set[int] = set()
    stages: set[str] = set()
    for engine, metas in task_meta.items():
        if not isinstance(metas, list):
            continue
        for meta in metas:
            if not isinstance(meta, dict):
                continue
            flags = int(meta.get("flags") or 0)
            stream_flags = int(meta.get("stream_flags") or 0)
            if not (flags & 0x10) or not stream_flags:
                continue
            layer = int(meta.get("layer") or 0)
            mb = int(meta.get("mb") or 0)
            seen.add((layer, mb))
            layers.add(layer)
            if engine in ("udma_r", "udma_w"):
                if stream_flags & 0x8:
                    stages.add("store")
                else:
                    stages.add("load")
            elif engine in ("ewe", "pool", "tnps"):
                stages.add("consumer")
            else:
                stages.add(str(engine))
    return {
        "fuse_hit": "yes" if fused_flows or streamed_layers else "no",
        "fuse_flows": ";".join(fused_flows),
        "streamed_layers": ";".join(f"L{x}" for x in streamed_layers),
        "mb_hit": "yes" if seen else "no",
        "mb_count": str(len(seen)),
        "mb_layers": ";".join(f"L{x}" for x in sorted(layers)),
        "mb_stages": "+".join(s for s in ("load", "conv", "requant", "consumer", "store")
                              if s in stages),
    }


def _compare_paths(model_path: Path) -> dict[str, Path]:
    return _mode_paths(model_path, "rtl_compare")


def _compare_report_exists_for(pattern: str, model_dir: Path) -> bool:
    canonical = _normalise_pattern(pattern)
    return _compare_paths(model_dir / f"{canonical}.tflite")["html"].exists()


def _cx_rtl_compare_paths(model_path: Path) -> dict[str, Path]:
    return _mode_paths(model_path, "cx_rtl_compare")


def _cx_rtl_compare_report_exists_for(pattern: str, model_dir: Path) -> bool:
    canonical = _normalise_pattern(pattern)
    return _cx_rtl_compare_paths(model_dir / f"{canonical}.tflite")["html"].exists()


def _write_two_mode_compare_html(model_path: Path,
                                 compare_suffix: str,
                                 left_label: str,
                                 left_html: Path,
                                 right_label: str,
                                 right_html: Path,
                                 mdla6_cx_ms: float | None,
                                 left_ms: float | None,
                                 right_ms: float | None,
                                 left_status: str,
                                 right_status: str,
                                 ratios: list[tuple[str, float | None, float | None]]) -> Path:
    out = _mode_paths(model_path, compare_suffix)["html"]
    left_doc = left_html.read_text(errors="ignore") if left_html.exists() else ""
    right_doc = right_html.read_text(errors="ignore") if right_html.exists() else ""

    def ms(v: float | None) -> str:
        return f"{v:.3f} ms" if v is not None else ""

    ratio_spans = []
    for label, num, den in ratios:
        ratio = _ratio_from_ms(num, den)
        ratio_spans.append(
            f"<span><b>{html.escape(label)}:</b> "
            f"{html.escape(ratio + 'x' if ratio else '')}</span>"
        )

    doc = f"""<!doctype html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MDLA7 profile — {html.escape(model_path.name)} — {html.escape(left_label)}/{html.escape(right_label)}</title>
<style>
body {{ margin:0; font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
       color:#222; background:#f6f7f9; }}
header {{ position:sticky; top:0; z-index:10; background:#fff; border-bottom:1px solid #d8dde6;
          padding:12px 18px; box-shadow:0 1px 4px rgba(0,0,0,.04); }}
h1 {{ margin:0 0 8px; font-size:18px; }}
.summary {{ display:flex; flex-wrap:wrap; gap:10px 18px; color:#495464; font-size:12px; }}
.summary b {{ color:#222; }}
.tabs {{ margin-top:10px; display:flex; gap:8px; }}
button {{ border:1px solid #c8d0dc; background:#fff; border-radius:5px; padding:5px 10px; cursor:pointer; }}
button.active {{ background:#1f5fa8; color:#fff; border-color:#1f5fa8; }}
.pane {{ display:none; padding:0; }}
.pane.active {{ display:block; }}
iframe {{ width:100%; height:calc(100vh - 120px); border:0; background:#fff; display:block; }}
</style></head>
<body>
<header>
  <h1>{html.escape(model_path.name)} — {html.escape(left_label)}/{html.escape(right_label)} profile</h1>
  <div class="summary">
    <span><b>mdla6_cx:</b> {html.escape(ms(mdla6_cx_ms))}</span>
    <span><b>{html.escape(left_label)}:</b> {html.escape(ms(left_ms))} {html.escape(left_status)}</span>
    <span><b>{html.escape(right_label)}:</b> {html.escape(ms(right_ms))} {html.escape(right_status)}</span>
    {''.join(ratio_spans)}
  </div>
  <div class="tabs">
    <button class="tab active" data-target="{html.escape(left_label)}">{html.escape(left_label)}</button>
    <button class="tab" data-target="{html.escape(right_label)}">{html.escape(right_label)}</button>
  </div>
</header>
<section id="{html.escape(left_label)}" class="pane active">
  <iframe title="{html.escape(left_label)} profile" srcdoc="{html.escape(left_doc, quote=True)}"></iframe>
</section>
<section id="{html.escape(right_label)}" class="pane">
  <iframe title="{html.escape(right_label)} profile" srcdoc="{html.escape(right_doc, quote=True)}"></iframe>
</section>
<script>
document.querySelectorAll('.tab').forEach(btn => btn.addEventListener('click', () => {{
  document.querySelectorAll('.tab').forEach(b => b.classList.toggle('active', b === btn));
  document.querySelectorAll('.pane').forEach(p => p.classList.toggle('active', p.id === btn.dataset.target));
}}));
</script>
</body></html>
"""
    out.write_text(doc)
    return out


def _write_rtl_compare_html(model_path: Path,
                            fast_html: Path,
                            rtl_html: Path,
                            mdla6_cx_ms: float | None,
                            fast_ms: float | None,
                            rtl_ms: float | None,
                            fast_status: str,
                            rtl_status: str) -> Path:
    return _write_two_mode_compare_html(
        model_path, "rtl_compare",
        "fast", fast_html, "rtl", rtl_html,
        mdla6_cx_ms, fast_ms, rtl_ms, fast_status, rtl_status,
        [("rtl/fast", rtl_ms, fast_ms), ("rtl/mdla6_cx", rtl_ms, mdla6_cx_ms)])


def _write_cx_rtl_compare_html(model_path: Path,
                                  rtl_html: Path,
                                  synth_html: Path,
                                  mdla6_cx_ms: float | None,
                                  rtl_ms: float | None,
                                  cx_ms: float | None,
                                  rtl_status: str,
                                  cx_status: str) -> Path:
    return _write_two_mode_compare_html(
        model_path, "cx_rtl_compare",
        "rtl", rtl_html, "cx", synth_html,
        mdla6_cx_ms, rtl_ms, cx_ms, rtl_status, cx_status,
        [("cx/rtl", cx_ms, rtl_ms),
         ("rtl/mdla6_cx", rtl_ms, mdla6_cx_ms),
         ("cx/mdla6_cx", cx_ms, mdla6_cx_ms)])


def _write_rtl_compare_index(title: str, html_out: str,
                             rows: list[dict], csv_path: Path) -> None:
    out_path = HERE / html_out
    link_prefix = "../output/" if out_path.parent == HERE / "profile" else "output/"
    body = []
    for row in rows:
        pat = row.get("pattern", "")
        link = f"{link_prefix}{html.escape(pat)}.rtl_compare.html"
        status = row.get("status", "")
        body.append(
            "<tr>"
            f"<td><a href=\"{link}\">{html.escape(pat)}</a></td>"
            f"<td style='text-align:right'>{html.escape(_ms_value_cell(row.get('mdla6_cx_ms', '')))}</td>"
            f"<td style='text-align:right'>{html.escape(_ms_value_cell(row.get('fast_ms', '')))}</td>"
            f"<td style='text-align:right'>{html.escape(_ms_value_cell(row.get('rtl_ms', '')))}</td>"
            f"<td style='text-align:right'>{html.escape(row.get('f_over_cx', row.get('fast_over_mdla6_cx', '')))}</td>"
            f"<td style='text-align:right'>{html.escape(row.get('rtl_over_fast', ''))}</td>"
            f"<td style='text-align:right'>{html.escape(row.get('rtl_over_mdla6_cx', ''))}</td>"
            f"<td>{html.escape(status)}</td>"
            "</tr>"
        )
    doc = f"""<!doctype html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
body {{ font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
       max-width:1200px; margin:24px auto; padding:0 16px; color:#222; }}
h1 {{ font-size:20px; margin-bottom:4px; }}
.meta {{ color:#666; font-size:12px; margin-bottom:14px; }}
table {{ border-collapse:collapse; width:100%; font-size:12px; }}
th,td {{ border:1px solid #e4e4e4; padding:5px 8px; font-variant-numeric:tabular-nums; }}
th {{ background:#f4f4f4; text-align:left; }}
a {{ color:#1f5fa8; text-decoration:none; }}
a:hover {{ text-decoration:underline; }}
</style></head><body>
<h1>{html.escape(title)}</h1>
<div class="meta">csv: {html.escape(str(csv_path))}</div>
<table>
  <thead><tr><th>pattern</th><th>mdla6_cx ms</th><th>fast ms</th><th>rtl ms</th><th>f/mdla6_cx</th><th>rtl/fast</th><th>rtl/mdla6_cx</th><th>status</th></tr></thead>
  <tbody>{''.join(body)}</tbody>
</table>
</body></html>
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(doc)


def _write_cx_rtl_compare_index(title: str, html_out: str,
                                   rows: list[dict], csv_path: Path) -> None:
    out_path = HERE / html_out
    link_prefix = "../output/" if out_path.parent == HERE / "profile" else "output/"
    body = []
    for row in rows:
        pat = row.get("pattern", "")
        link = f"{link_prefix}{html.escape(pat)}.cx_rtl_compare.html"
        status = row.get("status", "")
        body.append(
            "<tr>"
            f"<td><a href=\"{link}\">{html.escape(pat)}</a></td>"
            f"<td style='text-align:right'>{html.escape(_ms_value_cell(row.get('mdla6_cx_ms', '')))}</td>"
            f"<td style='text-align:right'>{html.escape(_ms_value_cell(row.get('rtl_ms', '')))}</td>"
            f"<td style='text-align:right'>{html.escape(_ms_value_cell(row.get('cx_ms', '')))}</td>"
            f"<td style='text-align:right'>{html.escape(row.get('cx_over_rtl', ''))}</td>"
            f"<td style='text-align:right'>{html.escape(row.get('rtl_over_mdla6_cx', ''))}</td>"
            f"<td style='text-align:right'>{html.escape(row.get('cx_over_mdla6_cx', ''))}</td>"
            f"<td>{html.escape(status)}</td>"
            "</tr>"
        )
    doc = f"""<!doctype html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
body {{ font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
       max-width:1200px; margin:24px auto; padding:0 16px; color:#222; }}
h1 {{ font-size:20px; margin-bottom:4px; }}
.meta {{ color:#666; font-size:12px; margin-bottom:14px; }}
table {{ border-collapse:collapse; width:100%; font-size:12px; }}
th,td {{ border:1px solid #e4e4e4; padding:5px 8px; font-variant-numeric:tabular-nums; }}
th {{ background:#f4f4f4; text-align:left; }}
a {{ color:#1f5fa8; text-decoration:none; }}
a:hover {{ text-decoration:underline; }}
</style></head><body>
<h1>{html.escape(title)}</h1>
<div class="meta">csv: {html.escape(str(csv_path))}</div>
<table>
  <thead><tr><th>pattern</th><th>mdla6_cx ms</th><th>rtl ms</th><th>cx ms</th><th>cx/rtl</th><th>rtl/mdla6_cx</th><th>cx/mdla6_cx</th><th>status</th></tr></thead>
  <tbody>{''.join(body)}</tbody>
</table>
</body></html>
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(doc)


def _row_print(s: str) -> None:
    sys.stdout.write("\r\033[2K" + s + "\n")
    sys.stdout.flush()


def _row_update(s: str) -> None:
    sys.stdout.write("\r\033[2K" + s)
    sys.stdout.flush()


def run_corpus(*,
               corpus_name: str,
               default_model_dir: Path,
               default_csv_out: Path,
               profile_html: str,
               profile_title: str,
               pattern_order_csv: Path | str | None = None,
               recursive: bool = False,
               microblock_metrics: bool = False) -> None:
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--model-dir", default=str(default_model_dir),
                    help=f"directory containing {corpus_name} .tflite models")
    ap.add_argument("--csv-out", "--csv", dest="csv_out",
                    default=str(default_csv_out),
                    help="output regression CSV")
    ap.add_argument("--filter", default="",
                    help="substring filter on model name")
    ap.add_argument("--pattern-order-csv", default="",
                    help="optional CSV with Pattern,mdla6_cx columns used to order selected models")
    ap.add_argument("--limit", type=int, default=0,
                    help="only run the first N selected models (0 = no limit)")
    ap.add_argument("--offset", type=int, default=0,
                    help="skip the first N selected models before applying --limit")
    ap.add_argument("--rerun-all", action="store_true",
                    help="ignore prior --csv-out cache and re-run everything")
    ap.add_argument("--fast-only", action="store_true",
                    help=argparse.SUPPRESS)
    ap.add_argument("--cx", action="store_true",
                    help="run cx L1Mesh and cx engine mode (default is fast)")
    ap.add_argument("--keep-bin", action="store_true",
                    help="keep per-model .bin files in output/ after the sweep")
    ap.add_argument("--no-html", action="store_true",
                    help="skip per-model and index HTML generation; keep profile JSON/CSV")
    ap.add_argument("--list", action="store_true",
                    help="list selected models and exit")
    args = ap.parse_args()
    args.l1_timing = "cx" if args.cx else "fast"
    args.engine_model = "cx" if args.cx else "fast"
    args.fast_only = True
    args.compare_rtl_fast = False
    args.compare_cx_rtl = False

    output_mode = _mode_display(args.l1_timing, args.engine_model)
    _set_output_dir(output_mode)
    configured_csv = Path(args.csv_out)
    configured_is_default = configured_csv == Path(default_csv_out)
    default_csv = OUT_DIR / Path(default_csv_out).name
    if configured_is_default:
        args.csv_out = str(default_csv)
    suffix = "" if (args.compare_rtl_fast or args.compare_cx_rtl) else _mode_suffix(
        args.l1_timing, args.engine_model)
    if suffix and OUT_DIR.name != suffix and configured_is_default:
        args.csv_out = str(default_csv.with_name(
            f"{default_csv.stem}.{suffix}{default_csv.suffix}"))
    profile_suffix = "" if (args.compare_rtl_fast or args.compare_cx_rtl) else _profile_mode_suffix(
        args.l1_timing, args.engine_model)
    if not profile_suffix and not (args.compare_rtl_fast or args.compare_cx_rtl):
        profile_suffix = "fast"
    if profile_suffix:
        profile_path = Path(profile_html)
        profile_html = str(profile_path.with_name(
            f"{profile_path.stem}.{profile_suffix}{profile_path.suffix}"))
    model_dir = Path(args.model_dir)
    patterns = _discover_models(model_dir, args.filter, recursive=recursive)
    order_source: Path | str | None
    if args.pattern_order_csv:
        order_source = Path(args.pattern_order_csv)
    else:
        order_source = pattern_order_csv
    patterns = _apply_pattern_order(patterns, order_source)
    if args.offset:
        patterns = patterns[args.offset:]
    if args.limit:
        patterns = patterns[:args.limit]

    if args.list:
        for pat in patterns:
            path = model_dir / f"{pat}.tflite"
            print(f"{_fit_cell(pat)} {path.stat().st_size / (1024 * 1024):6.1f} MB")
        return

    if not patterns:
        raise SystemExit(f"no .tflite models found in {model_dir} "
                         f"(filter={args.filter!r})")
    if not MODEL_RUNNER.exists():
        raise SystemExit(f"mdla7_model_runner not built: {MODEL_RUNNER}\n"
                         f"  run `make -C systemc -s` from repo root")

    if args.compare_rtl_fast:
        compare_csv = default_csv.with_name(
            f"{default_csv.stem}.rtl_compare{default_csv.suffix}")
        csv_path = Path(args.csv_out)
        if csv_path == default_csv:
            csv_path = compare_csv
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        prior_full = {} if args.rerun_all else _load_prior_csv(csv_path)
        prior_ok = {
            p: r for p, r in prior_full.items()
            if r.get("status") == "ok" and r.get("fast_ms") and r.get("rtl_ms")
        }
        profile_path = Path(profile_html)
        compare_html = f"{profile_path.stem}.rtl_compare{profile_path.suffix}"
        mdla6_cx_ms_by_pattern = _load_mdla6_cx_ms(order_source)

        def _mdla6_cx_ms_for(pattern: str) -> float | None:
            return mdla6_cx_ms_by_pattern.get(_normalise_pattern(pattern))

        def _fill_compare_ms(row: dict) -> dict:
            out = dict(row)
            mdla6_cx_ms = out.get("mdla6_cx_ms")
            if not mdla6_cx_ms:
                value = _mdla6_cx_ms_for(out.get("pattern", ""))
                out["mdla6_cx_ms"] = f"{value:.3f}" if value is not None else ""
            if not out.get("rtl_over_fast"):
                out["rtl_over_fast"] = _ratio_from_ms(
                    out.get("rtl_ms", ""), out.get("fast_ms", ""))
            if not out.get("f_over_cx"):
                out["f_over_cx"] = out.get("fast_over_mdla6_cx", "") or _ratio_from_ms(
                    out.get("fast_ms", ""), out.get("mdla6_cx_ms", ""))
            if not out.get("rtl_over_mdla6_cx"):
                out["rtl_over_mdla6_cx"] = _ratio_from_ms(
                    out.get("rtl_ms", ""), out.get("mdla6_cx_ms", ""))
            return out

        def _checkpoint_compare(rows: list[dict]) -> None:
            seen = {r["pattern"] for r in rows}
            merged = [_fill_compare_ms(r) for r in rows]
            for pat, prow in prior_full.items():
                if pat not in seen:
                    merged.append(_fill_compare_ms(prow))
            fields = [
                "pattern", "mdla6_cx_ms", "fast_ms", "rtl_ms",
                "f_over_cx", "rtl_over_fast", "rtl_over_mdla6_cx",
                "status", "fast_status", "rtl_status",
            ]
            with csv_path.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
                w.writeheader()
                w.writerows(merged)
            if not args.no_html:
                _write_rtl_compare_index(f"{profile_title} — fast vs rtl",
                                         compare_html, merged, csv_path)

        try:
            rel_model_dir = model_dir.relative_to(REPO_ROOT)
        except ValueError:
            rel_model_dir = model_dir
        print(f"==== MDLA7 {corpus_name} fast vs rtl regression: {len(patterns)} models "
              f"(from {rel_model_dir}) ====", flush=True)
        rows_out = []
        t_total = time.time()
        for i, pat in enumerate(patterns, 1):
            if pat in prior_ok and (args.no_html or _compare_report_exists_for(pat, model_dir)):
                cached = prior_ok[pat]
                cached_filled = _fill_compare_ms(cached)
                display_pat = _fit_cell(pat)
                _row_print(f"[{i:>2}/{len(patterns)}] {display_pat} "
                           f"mdla6_cx={_ms_cell(cached_filled.get('mdla6_cx_ms', ''))} "
                           f"fast={_ms_cell(cached.get('fast_ms', ''))} "
                           f"rtl={_ms_cell(cached.get('rtl_ms', ''))} "
                           f"f/mdla6_cx={_ratio_cell(cached_filled.get('f_over_cx', cached_filled.get('fast_over_mdla6_cx', '')))} "
                           f"rtl/fast={_ratio_cell(cached_filled.get('rtl_over_fast', ''))} "
                           f"rtl/mdla6_cx={_ratio_cell(cached_filled.get('rtl_over_mdla6_cx', ''))} cached  "
                           f"{cached.get('status', 'ok')}")
                rows_out.append(cached_filled)
                _checkpoint_compare(rows_out)
                continue

            t0 = time.time()

            def _progress(stage: str) -> None:
                elapsed = time.time() - t0
                display_pat = _fit_cell(pat)
                _row_update(f"[{i:>2}/{len(patterns)}] {display_pat} "
                            f"{'—':>10s}      ({elapsed:5.1f}s)  "
                            f"running {stage}...")

            _, fast_ms, _, _, fast_status, _, _ = run_one(
                pat, model_dir, progress=lambda s: _progress(f"fast {s}"),
                fast_only=True, skip_html=args.no_html, engine_model="fast",
                l1_timing="fast")
            _, rtl_ms, _, _, rtl_status, _, _ = run_one(
                pat, model_dir, progress=lambda s: _progress(f"rtl {s}"),
                fast_only=True, skip_html=args.no_html, engine_model="rtl",
                l1_timing="fast")
            mdla6_cx_ms = _mdla6_cx_ms_for(pat)
            fast_mdla6_cx = _ratio_from_ms(fast_ms, mdla6_cx_ms)
            rtl_fast = _ratio_from_ms(rtl_ms, fast_ms)
            rtl_mdla6_cx = _ratio_from_ms(rtl_ms, mdla6_cx_ms)
            status = "ok" if fast_status == "ok" and rtl_status == "ok" else f"{fast_status}/{rtl_status}"
            model_path = model_dir / f"{_normalise_pattern(pat)}.tflite"
            if not args.no_html and fast_ms is not None and rtl_ms is not None:
                try:
                    _write_rtl_compare_html(
                        model_path,
                        OUT_DIR / f"{model_path.stem}.fast.html",
                        _mode_paths(model_path, "rtl")["html"],
                        mdla6_cx_ms, fast_ms, rtl_ms, fast_status, rtl_status)
                except Exception as e:
                    status = f"{status}; html-fail: {str(e)[:80]}"
            elapsed = time.time() - t0
            display_pat = _fit_cell(pat)
            _row_print(f"[{i:>2}/{len(patterns)}] {display_pat} "
                       f"mdla6_cx={f'{mdla6_cx_ms:>8.2f} ms' if mdla6_cx_ms is not None else f'{chr(8212):>8s}    '} "
                       f"fast={f'{fast_ms:>8.2f} ms' if fast_ms is not None else f'{chr(8212):>8s}    '} "
                       f"rtl={f'{rtl_ms:>8.2f} ms' if rtl_ms is not None else f'{chr(8212):>8s}    '} "
                       f"f/mdla6_cx={_ratio_cell(fast_mdla6_cx)} "
                       f"rtl/fast={_ratio_cell(rtl_fast)} rtl/mdla6_cx={_ratio_cell(rtl_mdla6_cx)}  "
                       f"({elapsed:5.1f}s)  {status}")
            row = {
                "pattern": pat,
                "mdla6_cx_ms": f"{mdla6_cx_ms:.3f}" if mdla6_cx_ms is not None else "",
                "fast_ms": f"{fast_ms:.3f}" if fast_ms is not None else "",
                "rtl_ms": f"{rtl_ms:.3f}" if rtl_ms is not None else "",
                "f_over_cx": fast_mdla6_cx,
                "rtl_over_fast": rtl_fast,
                "rtl_over_mdla6_cx": rtl_mdla6_cx,
                "status": status,
                "fast_status": fast_status,
                "rtl_status": rtl_status,
            }
            rows_out.append(row)
            _checkpoint_compare(rows_out)
            if not args.keep_bin:
                for bin_path in (
                    _artefact_paths(model_path)["prog"],
                    _mode_paths(model_path, "rtl")["prog"],
                    _mode_paths(model_path, "rtl_compare")["prog"],
                ):
                    try:
                        if bin_path.exists():
                            bin_path.unlink()
                    except OSError:
                        pass

        n_both = sum(1 for r in rows_out if r.get("fast_ms") and r.get("rtl_ms"))
        total_fast = sum(float(r["fast_ms"]) for r in rows_out if r.get("fast_ms"))
        total_rtl = sum(float(r["rtl_ms"]) for r in rows_out if r.get("rtl_ms"))
        total_s = time.time() - t_total
        print(f"\n==== summary: compared {n_both}/{len(rows_out)}, "
              f"fast total {total_fast:.1f} ms, rtl total {total_rtl:.1f} ms, "
              f"wall {total_s:.0f}s ====", flush=True)
        print(f"csv: {csv_path}", flush=True)
        if not args.no_html:
            print(f"html: {HERE / compare_html}", flush=True)
        _raise_if_not_clean(rows_out)
        return

    if args.compare_cx_rtl:
        compare_csv = default_csv.with_name(
            f"{default_csv.stem}.cx_rtl_compare{default_csv.suffix}")
        csv_path = Path(args.csv_out)
        if csv_path == default_csv:
            csv_path = compare_csv
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        prior_full = {} if args.rerun_all else _load_prior_csv(csv_path)
        prior_ok = {
            p: r for p, r in prior_full.items()
            if r.get("status") == "ok" and r.get("rtl_ms") and r.get("cx_ms")
        }
        profile_path = Path(profile_html)
        compare_html = f"{profile_path.stem}.cx_rtl_compare{profile_path.suffix}"
        mdla6_cx_ms_by_pattern = _load_mdla6_cx_ms(order_source)
        rtl_suffix = _mode_suffix("rtl", "rtl")
        cx_suffix = _mode_suffix("cx", "cx")

        def _mdla6_cx_ms_for(pattern: str) -> float | None:
            return mdla6_cx_ms_by_pattern.get(_normalise_pattern(pattern))

        def _fill_synth_compare_ms(row: dict) -> dict:
            out = dict(row)
            mdla6_cx_ms = out.get("mdla6_cx_ms")
            if not mdla6_cx_ms:
                value = _mdla6_cx_ms_for(out.get("pattern", ""))
                out["mdla6_cx_ms"] = f"{value:.3f}" if value is not None else ""
            if not out.get("cx_over_rtl"):
                out["cx_over_rtl"] = _ratio_from_ms(
                    out.get("cx_ms", ""), out.get("rtl_ms", ""))
            if not out.get("rtl_over_mdla6_cx"):
                out["rtl_over_mdla6_cx"] = _ratio_from_ms(
                    out.get("rtl_ms", ""), out.get("mdla6_cx_ms", ""))
            if not out.get("cx_over_mdla6_cx"):
                out["cx_over_mdla6_cx"] = _ratio_from_ms(
                    out.get("cx_ms", ""), out.get("mdla6_cx_ms", ""))
            return out

        def _checkpoint_synth_compare(rows: list[dict]) -> None:
            seen = {r["pattern"] for r in rows}
            merged = [_fill_synth_compare_ms(r) for r in rows]
            for pat, prow in prior_full.items():
                if pat not in seen:
                    merged.append(_fill_synth_compare_ms(prow))
            fields = [
                "pattern", "mdla6_cx_ms", "rtl_ms", "cx_ms",
                "cx_over_rtl", "rtl_over_mdla6_cx", "cx_over_mdla6_cx",
                "status", "rtl_status", "cx_status",
            ]
            with csv_path.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
                w.writeheader()
                w.writerows(merged)
            if not args.no_html:
                _write_cx_rtl_compare_index(f"{profile_title} — rtl vs cx",
                                               compare_html, merged, csv_path)

        try:
            rel_model_dir = model_dir.relative_to(REPO_ROOT)
        except ValueError:
            rel_model_dir = model_dir
        print(f"==== MDLA7 {corpus_name} rtl vs cx regression: {len(patterns)} models "
              f"(from {rel_model_dir}) ====", flush=True)
        rows_out = []
        t_total = time.time()
        for i, pat in enumerate(patterns, 1):
            if pat in prior_ok and (
                    args.no_html or _cx_rtl_compare_report_exists_for(pat, model_dir)):
                cached = prior_ok[pat]
                cached_filled = _fill_synth_compare_ms(cached)
                display_pat = _fit_cell(pat)
                _row_print(f"[{i:>2}/{len(patterns)}] {display_pat} "
                           f"mdla6_cx={_ms_cell(cached_filled.get('mdla6_cx_ms', ''))} "
                           f"rtl={_ms_cell(cached.get('rtl_ms', ''))} "
                           f"cx={_ms_cell(cached.get('cx_ms', ''))} "
                           f"cx/rtl={_ratio_cell(cached_filled.get('cx_over_rtl', ''))} "
                           f"cx/mdla6_cx={_ratio_cell(cached_filled.get('cx_over_mdla6_cx', ''))} cached  "
                           f"{cached.get('status', 'ok')}")
                rows_out.append(cached_filled)
                _checkpoint_synth_compare(rows_out)
                continue

            t0 = time.time()

            def _progress(stage: str) -> None:
                elapsed = time.time() - t0
                display_pat = _fit_cell(pat)
                _row_update(f"[{i:>2}/{len(patterns)}] {display_pat} "
                            f"{'—':>10s}      ({elapsed:5.1f}s)  "
                            f"running {stage}...")

            _, rtl_ms, _, _, rtl_status, _, _ = run_one(
                pat, model_dir, progress=lambda s: _progress(f"rtl {s}"),
                fast_only=True, skip_html=args.no_html, engine_model="rtl",
                l1_timing="rtl")
            _, cx_ms, _, _, cx_status, _, _ = run_one(
                pat, model_dir, progress=lambda s: _progress(f"cx {s}"),
                fast_only=True, skip_html=args.no_html, engine_model="cx",
                l1_timing="cx")
            mdla6_cx_ms = _mdla6_cx_ms_for(pat)
            cx_rtl = _ratio_from_ms(cx_ms, rtl_ms)
            rtl_mdla6_cx = _ratio_from_ms(rtl_ms, mdla6_cx_ms)
            cx_mdla6_cx = _ratio_from_ms(cx_ms, mdla6_cx_ms)
            status = "ok" if rtl_status == "ok" and cx_status == "ok" else f"{rtl_status}/{cx_status}"
            model_path = model_dir / f"{_normalise_pattern(pat)}.tflite"
            if not args.no_html and rtl_ms is not None and cx_ms is not None:
                try:
                    _write_cx_rtl_compare_html(
                        model_path,
                        _mode_paths(model_path, rtl_suffix)["html"],
                        _mode_paths(model_path, cx_suffix)["html"],
                        mdla6_cx_ms, rtl_ms, cx_ms, rtl_status, cx_status)
                except Exception as e:
                    status = f"{status}; html-fail: {str(e)[:80]}"
            elapsed = time.time() - t0
            display_pat = _fit_cell(pat)
            _row_print(f"[{i:>2}/{len(patterns)}] {display_pat} "
                       f"mdla6_cx={f'{mdla6_cx_ms:>8.2f} ms' if mdla6_cx_ms is not None else f'{chr(8212):>8s}    '} "
                       f"rtl={f'{rtl_ms:>8.2f} ms' if rtl_ms is not None else f'{chr(8212):>8s}    '} "
                       f"cx={f'{cx_ms:>8.2f} ms' if cx_ms is not None else f'{chr(8212):>8s}    '} "
                       f"cx/rtl={_ratio_cell(cx_rtl)} cx/mdla6_cx={_ratio_cell(cx_mdla6_cx)}  "
                       f"({elapsed:5.1f}s)  {status}")
            row = {
                "pattern": pat,
                "mdla6_cx_ms": f"{mdla6_cx_ms:.3f}" if mdla6_cx_ms is not None else "",
                "rtl_ms": f"{rtl_ms:.3f}" if rtl_ms is not None else "",
                "cx_ms": f"{cx_ms:.3f}" if cx_ms is not None else "",
                "cx_over_rtl": cx_rtl,
                "rtl_over_mdla6_cx": rtl_mdla6_cx,
                "cx_over_mdla6_cx": cx_mdla6_cx,
                "status": status,
                "rtl_status": rtl_status,
                "cx_status": cx_status,
            }
            rows_out.append(row)
            _checkpoint_synth_compare(rows_out)
            if not args.keep_bin:
                for bin_path in (
                    _mode_paths(model_path, rtl_suffix)["prog"],
                    _mode_paths(model_path, cx_suffix)["prog"],
                    _mode_paths(model_path, "cx_rtl_compare")["prog"],
                ):
                    try:
                        if bin_path.exists():
                            bin_path.unlink()
                    except OSError:
                        pass

        n_both = sum(1 for r in rows_out if r.get("rtl_ms") and r.get("cx_ms"))
        total_rtl = sum(float(r["rtl_ms"]) for r in rows_out if r.get("rtl_ms"))
        total_cx = sum(float(r["cx_ms"]) for r in rows_out if r.get("cx_ms"))
        total_s = time.time() - t_total
        print(f"\n==== summary: compared {n_both}/{len(rows_out)}, "
              f"rtl total {total_rtl:.1f} ms, cx total {total_cx:.1f} ms, "
              f"wall {total_s:.0f}s ====", flush=True)
        print(f"csv: {csv_path}", flush=True)
        if not args.no_html:
            print(f"html: {HERE / compare_html}", flush=True)
        _raise_if_not_clean(rows_out)
        return

    csv_path = Path(args.csv_out)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    prior_full = {} if args.rerun_all else _load_prior_csv(csv_path)
    prior_ok = {} if args.rerun_all else _load_prior_results(csv_path, fast_only=args.fast_only)
    if prior_ok:
        print(f"  (cache: {len(prior_ok)} prior ok rows in {csv_path.name}; "
              f"per-model artefacts also reusable; --rerun-all to ignore)", flush=True)
    elif not args.rerun_all:
        print(f"  (cache: no prior ok rows in {csv_path.name}; "
              f"will reuse per-model artefacts when present)", flush=True)
    mdla6_cx_ms_by_pattern = _load_mdla6_cx_ms(order_source)
    has_mdla6_cx = bool(mdla6_cx_ms_by_pattern)
    primary_label = "cx" if (args.l1_timing == "cx" and args.engine_model == "cx") else "fast"
    ratio_label = "cx/mdla6_cx" if primary_label == "cx" else "f/mdla6_cx"

    def _mdla6_cx_ms_for(pattern: str) -> float | None:
        return mdla6_cx_ms_by_pattern.get(_normalise_pattern(pattern))

    def _attach_mdla6_cx(row: dict) -> dict:
        if not has_mdla6_cx:
            return row
        out = dict(row)
        mdla6_cx_ms = out.get("mdla6_cx")
        if not mdla6_cx_ms:
            value = _mdla6_cx_ms_for(out.get("pattern", ""))
            out["mdla6_cx"] = _format_ms(value)
        if not out.get("f_over_cx"):
            out["f_over_cx"] = out.get("fast_over_mdla6_cx", "") or _ratio_from_ms(
                out.get("mdla7_ms", ""), out.get("mdla6_cx", ""))
        return out

    def _checkpoint(rows: list[dict]) -> None:
        seen = {r["pattern"] for r in rows}
        merged = [_attach_mdla6_cx(r) for r in rows]
        for pat, prow in prior_full.items():
            if pat not in seen:
                merged.append(_attach_mdla6_cx(prow))
        fields = [
            "pattern",
            *(["mdla6_cx", "f_over_cx"] if has_mdla6_cx else []),
            "mdla7_ms",
            "status",
        ]
        if microblock_metrics:
            fields.extend([
                "fuse_hit", "fuse_flows", "streamed_layers",
                "mb_hit", "mb_count", "mb_layers", "mb_stages",
            ])
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(merged)

    try:
        rel_model_dir = model_dir.relative_to(REPO_ROOT)
    except ValueError:
        rel_model_dir = model_dir
    model_label = ""
    if args.l1_timing != "fast" or args.engine_model != "fast":
        model_label = f", mode={_mode_display(args.l1_timing, args.engine_model)}"
    print(f"==== MDLA7 {corpus_name} regression: {len(patterns)} models "
          f"(from {rel_model_dir}{model_label}) ====", flush=True)

    rows_out = []
    t_total = time.time()
    for i, pat in enumerate(patterns, 1):
        cached = prior_ok.get(pat)
        cache_source = "csv"
        if cached is not None and not (args.no_html or _report_exists_for(
                pat, model_dir, engine_model=args.engine_model,
                l1_timing=args.l1_timing)):
            cached = None
        if cached is None and not args.rerun_all:
            cached = _load_cached_result_from_artefacts(
                pat, model_dir, engine_model=args.engine_model,
                l1_timing=args.l1_timing, require_html=not args.no_html)
            cache_source = "artefact"
        if cached is not None:
            cached = _attach_mdla6_cx(cached)
            status_text = cached.get("status", "ok")
            model_path = model_dir / f"{pat}.tflite"
            mb = (_microblock_metrics_for(model_path, args.l1_timing, args.engine_model)
                  if microblock_metrics else {})
            mb_suffix = (f" fuse={mb.get('fuse_hit', 'no')}"
                         f" mb={mb.get('mb_count', '0')}:{mb.get('mb_stages', '')}"
                         if microblock_metrics else "")
            display_pat = _fit_cell(pat)
            _row_print(f"[{i:>2}/{len(patterns)}] {display_pat} "
                       f"{'mdla6_cx=' + _ms_cell(cached.get('mdla6_cx', '')) + ' ' if has_mdla6_cx else ''}"
                       f"{primary_label}={_ms_cell(cached.get('mdla7_ms', ''))} "
                       f"{ratio_label + '=' + _ratio_cell(cached.get('f_over_cx', cached.get('fast_over_mdla6_cx', ''))) + ' ' if has_mdla6_cx else ''}"
                       f"cached:{cache_source}  "
                       f"{status_text}{mb_suffix}")
            row = {
                "pattern": pat,
                "mdla6_cx": cached.get("mdla6_cx", ""),
                "f_over_cx": cached.get("f_over_cx", cached.get("fast_over_mdla6_cx", "")),
                "mdla7_ms": cached.get("mdla7_ms", ""),
                "status": cached.get("status", "ok"),
            }
            row.update(mb)
            rows_out.append(row)
            _checkpoint(rows_out)
            continue

        t0 = time.time()

        def _progress(stage: str) -> None:
            elapsed = time.time() - t0
            display_pat = _fit_cell(pat)
            _row_update(f"[{i:>2}/{len(patterns)}] {display_pat} "
                        f"{'—':>10s}      ({elapsed:5.1f}s)  "
                        f"running {stage}...")

        _, ms, conflict_ms, mesh_ms, status, conflict_status, mesh_status = run_one(
            pat, model_dir, progress=_progress, fast_only=args.fast_only,
            skip_html=args.no_html, engine_model=args.engine_model,
            l1_timing=args.l1_timing)
        elapsed = time.time() - t0
        ms_str = f"{ms:>8.2f} ms" if ms is not None else f"{'—':>8s}    "
        model_path = model_dir / f"{pat}.tflite"
        mb = (_microblock_metrics_for(model_path, args.l1_timing, args.engine_model)
              if microblock_metrics else {})
        mb_suffix = (f" fuse={mb.get('fuse_hit', 'no')}"
                     f" mb={mb.get('mb_count', '0')}:{mb.get('mb_stages', '')}"
                     if microblock_metrics else "")
        display_pat = _fit_cell(pat)
        mdla6_cx_ms = _mdla6_cx_ms_for(pat)
        fast_mdla6_cx = _ratio_from_ms(ms, mdla6_cx_ms)
        _row_print(f"[{i:>2}/{len(patterns)}] {display_pat} "
                   f"{'mdla6_cx=' + (f'{mdla6_cx_ms:>8.2f} ms' if mdla6_cx_ms is not None else f'{chr(8212):>8s}    ') + ' ' if has_mdla6_cx else ''}"
                   f"{primary_label}={ms_str} "
                   f"{ratio_label + '=' + _ratio_cell(fast_mdla6_cx) + ' ' if has_mdla6_cx else ''}"
                   f"({elapsed:5.1f}s)  "
                   f"{status}{mb_suffix}")
        row = {
            "pattern": pat,
            "mdla6_cx": _format_ms(mdla6_cx_ms),
            "f_over_cx": fast_mdla6_cx,
            "mdla7_ms": f"{ms:.3f}" if ms is not None else "",
            "status": status,
        }
        row.update(mb)
        rows_out.append(row)
        _checkpoint(rows_out)

        if not args.keep_bin:
            mode_suffix = _mode_suffix(args.l1_timing, args.engine_model)
            if mode_suffix:
                bin_paths = (
                    _mode_paths(model_path, mode_suffix)["prog"],
                    _mode_paths(model_path, f"{mode_suffix}.conflict")["prog"],
                    _mode_paths(model_path, f"{mode_suffix}.mesh")["prog"],
                )
            else:
                bin_paths = (
                    _artefact_paths(model_path)["prog"],
                    OUT_DIR / f"{pat}.conflict.bin",
                    OUT_DIR / f"{pat}.mesh.bin",
                )
            for bin_path in bin_paths:
                try:
                    if bin_path.exists():
                        bin_path.unlink()
                except OSError:
                    pass

    n_fast = sum(1 for r in rows_out if r.get("mdla7_ms"))
    n_clean = sum(1 for r in rows_out if r.get("status") == "ok")
    total_ms = sum(float(r["mdla7_ms"]) for r in rows_out if r.get("mdla7_ms"))
    total_s = time.time() - t_total
    print(f"\n==== summary: {primary_label} {n_fast}/{len(rows_out)} ran, "
          f"clean {n_clean}/{len(rows_out)}, "
          f"sim total {total_ms:.1f} ms, wall {total_s:.0f}s ====",
          flush=True)
    print(f"csv: {csv_path}", flush=True)
    if not args.no_html:
        _refresh_profile_index(_profile_title_for_mode(profile_title, primary_label),
                               profile_html, csv_path,
                               show_mdla6_cx=has_mdla6_cx,
                               primary_label=primary_label,
                               ratio_label=ratio_label,
                               report_suffix=suffix)
        print(f"html: {HERE / profile_html}", flush=True)
    _raise_if_not_clean(rows_out)

# ---- Unified public CLI ----

CORPORA = {
    "ethz": {
        "name": "ETHZ_v6",
        "model_dir": REPO_ROOT / "model" / "ETHZ_v6",
        "csv": OUT_DIR / "ethz_v6_regression.csv",
        "profile": "profile_ethz_v6.html",
        "title": "MDLA7 ETHZ_v6 Profiles",
        "order": "ethz_v6",
    },
    "ethz_v6": {
        "name": "ETHZ_v6",
        "model_dir": REPO_ROOT / "model" / "ETHZ_v6",
        "csv": OUT_DIR / "ethz_v6_regression.csv",
        "profile": "profile_ethz_v6.html",
        "title": "MDLA7 ETHZ_v6 Profiles",
        "order": "ethz_v6",
    },
    "ethz_v5": {
        "name": "ETHZ_v5",
        "model_dir": REPO_ROOT / "model" / "ETHZ_v5",
        "csv": OUT_DIR / "ethz_v5_regression.csv",
        "profile": "profile_ethz_v5.html",
        "title": "MDLA7 ETHZ_v5 Profiles",
        "order": None,
    },
    "hotspot": {
        "name": "Hotspot",
        "model_dir": REPO_ROOT / "model" / "Hotspot",
        "csv": OUT_DIR / "hotspot_regression.csv",
        "profile": "profile_hotspot.html",
        "title": "MDLA7 Hotspot Profiles",
        "order": None,
    },
    "slice": {
        "name": "MB_Path_Slice",
        "model_dir": REPO_ROOT / "model" / "MB_Path_Slice",
        "csv": OUT_DIR / "mb_path_regression.csv",
        "profile": "profile_mb_path.html",
        "title": "MDLA7 MB Path Slice Profiles",
        "order": None,
        "recursive": True,
        "microblock_metrics": True,
    },
    "mlperf": {
        "name": "MLPerf_Tiny",
        "model_dir": REPO_ROOT / "model" / "MLPerf_Tiny",
        "csv": OUT_DIR / "mlperf_regression.csv",
        "profile": "profile_mlperf.html",
        "title": "MDLA7 MLPerf_Tiny Profiles",
        "order": None,
    },
    "bmm": {
        "name": "BMM",
        "model_dir": REPO_ROOT / "model" / "BMM",
        "csv": OUT_DIR / "bmm_regression.csv",
        "profile": "profile_bmm.html",
        "title": "MDLA7 BMM Profiles",
        "order": None,
    },
    "unit": {
        "name": "UnitTest",
        "model_dir": REPO_ROOT / "model" / "UnitTest",
        "csv": OUT_DIR / "unittest_regression.csv",
        "profile": "profile_unittest.html",
        "title": "MDLA7 UnitTest TFLite Profiles",
        "order": None,
        "recursive": True,
    },
    "unittest": {
        "name": "UnitTest",
        "model_dir": REPO_ROOT / "model" / "UnitTest",
        "csv": OUT_DIR / "unittest_regression.csv",
        "profile": "profile_unittest.html",
        "title": "MDLA7 UnitTest TFLite Profiles",
        "order": None,
        "recursive": True,
    },
}


def _has_option(args: list[str], *names: str) -> bool:
    return any(arg == name or arg.startswith(f"{name}=")
               for arg in args for name in names)


def main() -> None:
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument("--filter", default="ethz",
                    help="corpus selector: ethz, ethz_v6, ethz_v5, hotspot, slice, mlperf, bmm")
    ap.add_argument("--model-filter", default="",
                    help="substring filter inside the selected corpus")
    ap.add_argument("-h", "--help", action="store_true")
    ns, rest = ap.parse_known_args()

    corpus_key = ns.filter.lower()
    if corpus_key == "ethz":
        corpus_key = "ethz_v6"
    if ns.help:
        print(__doc__.strip())
        print("\nCommon options: --limit N --offset N --rerun-all --cx")
        print("Corpus keys:", ", ".join(sorted(k for k in CORPORA if k != "ethz")))
        return
    if corpus_key not in CORPORA:
        valid = "/".join(k for k in ("ethz_v6", "ethz_v5", "hotspot", "slice", "mlperf", "bmm", "unit"))
        raise SystemExit(f"unknown --filter corpus {ns.filter!r}; use {valid}")

    runner_args = list(rest)
    if ns.model_filter:
        runner_args.extend(["--filter", ns.model_filter])

    if not _has_option(runner_args, "--fast-only"):
        runner_args.append("--fast-only")

    sys.argv = [sys.argv[0], *runner_args]
    cfg = CORPORA[corpus_key]
    print(f"[run_systemc] corpus={cfg['name']} mode_args={' '.join(runner_args)}", flush=True)
    run_corpus(
        corpus_name=str(cfg["name"]),
        default_model_dir=Path(cfg["model_dir"]),
        default_csv_out=Path(cfg["csv"]),
        profile_html=f"profile/{cfg['profile']}",
        profile_title=str(cfg["title"]),
        pattern_order_csv=cfg.get("order"),
        recursive=bool(cfg.get("recursive", False)),
        microblock_metrics=bool(cfg.get("microblock_metrics", False)),
    )


if __name__ == "__main__":
    main()
