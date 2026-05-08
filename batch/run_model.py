#!/usr/bin/env python3
"""End-to-end MDLA7 SystemC runner.

Default flow (= what running just `run_model.py <model>` does):

  .tflite ─▶ compile_model.py ─▶ program.bin ─▶ test_model
                                                   │
                  Mdla7System: Host → CmdEng → CONV / UDMA / ...
                                                   │
                                  bit-true verify + cycles report

Usage:
  ./run_model.py                       # default model, all CONV layers
  ./run_model.py <pattern>             # substring/typo match against model/
  ./run_model.py --list                # list bundled models
  ./run_model.py --all                 # try every INT8 model
  ./run_model.py --layer N             # legacy: run only one CONV layer
  ./run_model.py <pattern> --l1-timing conflict
  ./run_model.py <pattern> --l1-timing mesh
"""

from __future__ import annotations

import argparse
import difflib
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE        = Path(__file__).resolve().parent          # .../batch/
REPO_ROOT   = HERE.parent
SYSTEMC_DIR = REPO_ROOT / "systemc"
MODEL_ROOT  = REPO_ROOT / "model"
BUILD_DIR   = SYSTEMC_DIR / "build"
OUTPUT_DIR  = HERE / "output"                         # per-model artefacts
EXTRACT_PY  = SYSTEMC_DIR / "scripts" / "extract_conv.py"   # legacy single-layer
COMPILE_PY  = SYSTEMC_DIR / "scripts" / "compile_model.py"  # full-model compiler
PLOT_PY     = SYSTEMC_DIR / "scripts" / "plot_profile.py"   # gantt renderer
MODEL_PROFILE_PY = HERE / "gen_model_profile.py"
TEST_BIN    = BUILD_DIR / "test_tflite_conv"            # legacy single-layer
MODEL_BIN   = BUILD_DIR / "test_model"                  # full-model runner
BLOB_PATH   = BUILD_DIR / "conv_layer.bin"

def _artefact_paths(model: Path) -> dict[str, Path]:
    """Per-model output paths under output/<stem>.* — test_model derives
    .profile.json / .profile.csv from the program.bin stem so we just need
    to pass it a stable per-model name.

    Note: model names like 'mobilenet_v1_0.25_128_quant.tflite' contain dots
    in the stem, so we use f-strings rather than Path.with_suffix() (which
    treats '.25_128_quant' as a suffix and strips it)."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    stem = model.stem
    return {
        "prog":  OUTPUT_DIR / f"{stem}.bin",
        "prof":  OUTPUT_DIR / f"{stem}.profile.json",
        "csv":   OUTPUT_DIR / f"{stem}.profile.csv",
        "gantt": OUTPUT_DIR / f"{stem}.profile.png",
        "html":  OUTPUT_DIR / f"{stem}.html",
    }

def _refresh_model_profile_index() -> None:
    try:
        subprocess.run([sys.executable, str(MODEL_PROFILE_PY),
                        "--html-out", "profile_mdla6_pattern.html",
                        "--title", "MDLA7 MDLA6 Pattern Profiles",
                        "--only-metrics-rows"],
                       cwd=str(HERE), capture_output=True, text=True)
    except Exception:
        pass

DEFAULT_MODEL = REPO_ROOT / "model/INT8/efficientnet_lite0_int8.tflite"

# venv lives outside the repo so it survives volumes that don't support
# extended attrs (the 4T_OFFICE volume creates AppleDouble "._*" sidecars
# that crash Python's site importer).  Override with $MDLA7_VENV.
VENV_DIR = Path(os.environ.get("MDLA7_VENV") or
                Path.home() / ".venvs/mdla7").expanduser()
VENV_PY  = VENV_DIR / "bin" / "python"

# v8.1: regression artefact policy. After a successful run, .bin and
# .profile.json are deleted (they're only used to bridge compile→sim→Gantt→HTML
# and aren't user-facing). Set to True via --keep-intermediate to retain them
# for debugging.
_keep_intermediate = False


def _reexec_in_venv():
    """If a venv exists and we're not already running in it, re-exec there."""
    if not VENV_PY.exists():
        return  # no venv; fall back to system python
    # `Path.resolve()` chases the venv's bin/python symlink back to the
    # system Python and falsely reports "already in venv". Use sys.prefix
    # instead — it always points at the active venv root.
    if Path(sys.prefix).resolve() == VENV_DIR.resolve():
        return
    os.execv(str(VENV_PY), [str(VENV_PY), __file__, *sys.argv[1:]])


_reexec_in_venv()


# Priority order for resolving substring matches. ETHZ_v6 is the active
# regression-sweep bundle (deeplab/efficientnet_b4/etc.) so it wins over older
# ETHZ_v5 and the per-dtype dirs. Within the per-dtype tier, INT8 still wins
# (the most thoroughly-tested path).
_DTYPE_PRIORITY = {"ETHZ_v6": 0, "ETHZ_V6": 0,
                   "ETHZ_v5": 1, "ETHZ_V5": 1,
                   "INT8": 2, "UINT8": 3, "INT16": 4,
                   "INT4": 5, "FP16": 6, "BF16": 7, "FP32": 8}


def all_models() -> list[Path]:
    """Every .tflite under model/, filtering macOS AppleDouble sidecars."""
    out = []
    for sub in sorted(MODEL_ROOT.iterdir()):
        if not sub.is_dir():
            continue
        for m in sorted(sub.glob("*.tflite")):
            if m.name.startswith("._"):
                continue
            out.append(m)
    return out


def _match_key(p: Path) -> tuple[int, str]:
    return (_DTYPE_PRIORITY.get(p.parent.name, 99), str(p))


def resolve_model(arg: str) -> Path:
    """Resolve a CLI arg to a concrete .tflite path.

    Resolution order:
      1. Exact file path (absolute or relative to cwd / repo root).
      2. Case-insensitive substring match against every .tflite under model/.
         If multiple match, take the first by sorted path; mention the rest.
      3. Fuzzy "did you mean" via difflib if no substring matches.
    """
    p = Path(arg)
    for cand in (p, REPO_ROOT / arg, HERE / arg):
        if cand.exists() and cand.is_file():
            return cand.resolve()

    needle = arg.lower()
    pool = all_models()
    matches = sorted((m for m in pool if needle in m.name.lower()), key=_match_key)
    if matches:
        chosen = matches[0]
        rel = chosen.relative_to(REPO_ROOT)
        if len(matches) > 1:
            others = ", ".join(m.name for m in matches[1:5])
            tail = "" if len(matches) <= 5 else f" (+{len(matches) - 5} more)"
            print(f"[match] '{arg}' → {rel}   (also matched: {others}{tail})")
        else:
            print(f"[match] '{arg}' → {rel}")
        return chosen

    # Fuzzy fallback: typo-tolerant match by filename. Respect difflib's
    # similarity order so the closest *spelling* wins (the user's intent),
    # not the dtype priority — they can re-type with a precise pattern if
    # they want a specific quantization.
    suggestions = difflib.get_close_matches(arg, [m.name for m in pool], n=5, cutoff=0.5)
    if suggestions:
        first_name = suggestions[0]
        # If multiple files share that basename across dtype dirs, pick the
        # supported one (INT8 first) inside that name group only.
        same_name = sorted((m for m in pool if m.name == first_name), key=_match_key)
        chosen = same_name[0]
        rel = chosen.relative_to(REPO_ROOT)
        others = ", ".join(suggestions[1:])
        tail = f"   (alternatives: {others})" if others else ""
        print(f"[fuzzy] '{arg}' → {rel}{tail}")
        return chosen

    print(f"no model matches pattern '{arg}'  (try --list)")
    sys.exit(2)


def list_models():
    print("Available models under", MODEL_ROOT)
    last_dir = None
    for m in all_models():
        if m.parent != last_dir:
            print(f"\n  [{m.parent.name}]")
            last_dir = m.parent
        size_kb = m.stat().st_size // 1024
        print(f"    {m.relative_to(REPO_ROOT)}   ({size_kb} KB)")


def run(cmd, **kw):
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    return subprocess.run(cmd, **kw)


def check_python_deps():
    """numpy + (tflite-runtime OR tensorflow) needed by extract_conv.py."""
    # Suppress TF logger spam during the import probe.
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import warnings
    warnings.filterwarnings("ignore")

    missing = []
    try:
        import numpy            # noqa: F401
    except ImportError:
        missing.append("numpy")
    try:
        import tflite_runtime   # noqa: F401
    except ImportError:
        try:
            import tensorflow   # noqa: F401
        except ImportError:
            missing.append("tflite-runtime  (or tensorflow on Apple Silicon)")
    if missing:
        import platform
        py = f"python{sys.version_info.major}.{sys.version_info.minor}"
        arch = platform.machine()
        tf_pkg = "tensorflow" if arch == "arm64" else "tflite-runtime"
        print("missing python deps:")
        for m in missing:
            print(f"   - {m}")
        print("\ninstall with:")
        print(f"   {py} -m pip install --user numpy {tf_pkg}")
        print(f"   (or: pip install -r {SYSTEMC_DIR / 'requirements.txt'})")
        sys.exit(1)


def build():
    print("[step 1/3] make")
    r = subprocess.run(["make", "-s", "-C", str(SYSTEMC_DIR)])
    if r.returncode:
        sys.exit(f"build failed (exit {r.returncode})")


def count_convs(model: Path) -> int:
    """Number of CONV_2D ops in the model (DEPTHWISE_CONV_2D etc not counted)."""
    r = subprocess.run(
        [sys.executable, str(EXTRACT_PY), "--count", str(model)],
        capture_output=True, text=True,
    )
    if r.returncode:
        return 0
    try:
        return int(r.stdout.strip())
    except ValueError:
        return 0


def extract(model: Path, layer: int = 0, verbose: bool = True) -> tuple[bool, str]:
    """Return (success, error_summary). Never sys.exit."""
    if verbose:
        label = f"layer {layer}" if layer else "first CONV"
        print(f"[step 2/3] extract {label} from {model.name}")
    BUILD_DIR.mkdir(exist_ok=True)
    # Use the same interpreter we're running under (already venv-resolved).
    r = subprocess.run(
        [sys.executable, str(EXTRACT_PY), "--layer", str(layer),
         str(model), str(BLOB_PATH)],
        capture_output=True, text=True,
    )
    if verbose:
        keep = [ln for ln in (r.stdout or "").splitlines()
                if any(ln.startswith(k) for k in
                       ("model", "layer", "shape", "output", "wrote", "  "))]
        if keep:
            print("\n".join(keep))
    if r.returncode:
        # Extract a one-line error summary from stderr if possible.
        err = (r.stderr or "").strip().splitlines()
        summary = err[-1] if err else f"exit {r.returncode}"
        # Trim verbose flatbuffers errors.
        if "Model provided has model identifier" in summary:
            summary = "corrupt .tflite (likely HTML download error)"
        return (False, summary)
    return (True, "")


def simulate(verbose: bool = True, l1_timing: str = "fast") -> tuple[bool, float, str]:
    if verbose:
        print("[step 3/3] simulate on Mdla7System")
    t0 = time.time()
    r = subprocess.run(
        [str(TEST_BIN), str(BLOB_PATH), f"--l1-timing={l1_timing}"],
        capture_output=True, text=True,
    )
    elapsed = time.time() - t0
    out = r.stdout or ""
    if verbose:
        keep = [ln for ln in out.splitlines()
                if ln.startswith(("===", "    in=", "[CONV]", "PASS", "FAIL"))]
        for ln in keep[-6:]:
            print("   ", ln)
    last = out.splitlines()[-1] if out.strip() else "(no output)"
    return (r.returncode == 0 and "PASS" in out, elapsed, last)


def run_one(model: Path, layer: int = 0, l1_timing: str = "fast") -> bool:
    if layer == 0:
        print(f"\n========  {model.relative_to(REPO_ROOT)}  ========")
    if not model.exists():
        print(f"  → SKIP  (not found: {model})")
        return False
    ok, err = extract(model, layer)
    if not ok:
        print(f"  layer {layer}: SKIP  (extract failed: {err})")
        return False
    ok, elapsed, last = simulate(l1_timing=l1_timing)
    status = "PASS" if ok else "FAIL"
    if layer == 0:
        print(f"  → {status}   sim wall={elapsed:.2f}s   ({last})")
    return ok


def compile_and_run(model: Path, l1_timing: str = "fast") -> bool:
    """Default flow: compile entire model → one program.bin → one sc_start.

    Exercises the real Host → CommandEngine → CONV/UDMA dispatch chain.
    Artefacts (.bin / .profile.json / .csv / .png / .html) land under
    output/<model_stem>.* so multiple models don't clobber each other.
    """
    if not model.exists():
        print(f"\n========  {model}  ========")
        print(f"  → SKIP  (not found: {model})")
        return False

    BUILD_DIR.mkdir(exist_ok=True)
    paths = _artefact_paths(model)
    prog_path  = paths["prog"]
    prof_path  = paths["prof"]
    gantt_path = paths["gantt"]

    # Capture every line we *display* (filtered) so the HTML report contains
    # exactly the same console view the user saw.
    log_lines: list[str] = []
    def emit(line: str) -> None:
        print(line)
        log_lines.append(line)

    emit(f"\n========  {model.relative_to(REPO_ROOT)}  ========")
    emit(f"[step 2/3] compile {model.name} → {prog_path.relative_to(HERE)}")
    r = subprocess.run(
        [sys.executable, str(COMPILE_PY), str(model), str(prog_path)],
        capture_output=True, text=True,
    )
    if r.returncode:
        err = (r.stderr or "").strip().splitlines()
        summary = err[-1] if err else f"exit {r.returncode}"
        if "Model provided has model identifier" in summary:
            summary = "corrupt .tflite (likely HTML download error)"
        emit(f"  → SKIP  (compile failed: {summary})")
        return False
    # Surface compile output verbatim (compile_model already uses the canonical
    # "  layer NN  in=...  ready" line shape; we don't want to reflow it).
    for ln in (r.stdout or "").splitlines():
        if ln.startswith(("compile_model:", "  layer", "  →")):
            emit(ln)

    emit(f"[step 3/3] simulate Mdla7System")
    t0 = time.time()
    r = subprocess.run(
        [str(MODEL_BIN), str(prog_path), "--quiet", f"--l1-timing={l1_timing}"],
        capture_output=True, text=True,
    )
    elapsed = time.time() - t0
    out = r.stdout or ""
    # test_model uses the same canonical "  layer NN  in=...  PASS/FAIL" line,
    # so relaying verbatim makes it diffable against the compile lines above.
    for ln in out.splitlines():
        if (ln.startswith(("test_model:", "  layer", "  summary",
                           "  sim time", "  DRAM ", "  SRAM ", "  L1Mesh ",
                           "  per-engine", "  utilization", "    ",
                           "  profile", "  csv"))):
            emit(ln)
    emit(f"  ({elapsed:.2f}s wall)")

    # v4.3: render Gantt chart from profile.json if it was emitted.
    if prof_path.exists():
        gr = subprocess.run(
            [sys.executable, str(PLOT_PY), str(prof_path), "-o", str(gantt_path)],
            capture_output=True, text=True,
        )
        for ln in (gr.stdout or "").splitlines():
            if ln.startswith("  gantt"): emit(ln)
        if gr.returncode and gr.stderr:
            emit(f"  (gantt render failed: {gr.stderr.strip()})")

    # v6: write a self-contained HTML report bundling console + profile + gantt.
    if prof_path.exists():
        try:
            html_path = _write_html_report(model, paths, log_lines)
            emit(f"  html:    {html_path}")
            _refresh_model_profile_index()
        except Exception as e:
            emit(f"  (html report failed: {e})")

    # v8.1: clean up regression intermediates. .bin is only consumed by
    # test_model; .profile.json is only consumed by the Gantt plotter and the
    # HTML writer (both already done above). The user-facing artefacts that
    # survive are .profile.csv (tabular data) + .profile.png + .html. Pass
    # --keep-intermediate to keep .bin / .profile.json around for debugging.
    if r.returncode == 0 and not _keep_intermediate:
        for k in ("prog", "prof"):
            try:
                if paths[k].exists(): paths[k].unlink()
            except OSError:
                pass

    return r.returncode == 0


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

def _parse_compile_log(log_lines: list[str]) -> list[dict]:
    """Extract the per-layer compile_model.py lines (canonical "ready" line +
    early-skip line) into structured records. Returns one entry per compile-
    stage layer, including layers the simulator never sees because compile
    skipped them (e.g., FP ADD)."""
    rows: list[dict] = []
    for ln in log_lines:
        m = _COMPILE_FULL_RE.match(ln)
        if m:
            rows.append({
                "id":     int(m.group(1)),
                "op":     m.group(2),
                "in":     (int(m.group(3)), int(m.group(4)), int(m.group(5))),
                "k":      (int(m.group(6)), int(m.group(7))),
                "s":      (int(m.group(8)), int(m.group(9))),
                "group":  int(m.group(10)),
                "out":    (int(m.group(11)), int(m.group(12)), int(m.group(13))),
                "nelem":  int(m.group(14)),
                "dtype":  m.group(15),
                "status": m.group(16).strip(),
            })
            continue
        m = _COMPILE_SKIP_RE.match(ln)
        if m:
            rows.append({
                "id":     int(m.group(1)),
                "op":     m.group(2),
                "in":     (int(m.group(3)), int(m.group(4)), int(m.group(5))),
                "k":      None, "s": None, "group": None, "out": None,
                "nelem":  None, "dtype": "",
                "status": f"skipped ({m.group(6)})",
            })
    return rows


def _write_html_report(model: Path, paths: dict[str, Path],
                       log_lines: list[str]) -> Path:
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
        # Mirrors test_model.cpp's normal params blob. INT correlation blobs are
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

    # v8.2: build an interactive SVG Gantt instead of embedding the matplotlib PNG.
    # Engines render as horizontal lanes; tasks as colored rects; horizontal
    # mouse wheel zooms; click+drag pans; hover shows tooltip.
    eng_payload = {}
    conv_wait_tasks = []
    for name, e in engines.items():
        tasks = list(e.get("tasks") or [])
        if name == "udma_r":
            tasks = _annotate_udma_read_tasks(tasks)
            conv_wait_tasks = _conv_wait_tasks_from_udma(tasks)
        elif name == "udma_w":
            tasks = _annotate_udma_write_tasks(tasks)
        eng_payload[name] = {
            "busy": int(e.get("busy_cycles", 0) or 0),
            "tasks": tasks,
        }
    layer_marks = [
        {"id": int(L.get("id", i)),
         "flow": int(L.get("flow", L.get("id", i)) if L.get("flow") is not None else L.get("id", i)),
         "op": str(L.get("op", "")).strip(),
         "end": int(L.get("cycles_cum", 0) or 0)}
        for i, L in enumerate(layers)
    ]
    gantt_data_json = json.dumps({
        "engines": eng_payload,
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

    def _ceil_div(a: int, b: int) -> int:
        return (a + b - 1) // b if b > 0 else 0

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
        if op in ("add", "mul", "sub", "h_swsh", "gelu"):
            return _ceil_div(out_elems, _dtype_elem_lanes(dtype))
        if op == "softmax":
            return 3 * _ceil_div(out_elems, _dtype_elem_lanes(dtype))
        return 0

    ideal_rows: list[tuple[int, int]] = []
    ideal_cum = 0
    for idx, L in enumerate(layers):
        c = ready_compile_rows[idx] if idx < len(ready_compile_rows) else None
        ideal = _ideal_layer_cycles(L, c)
        ideal_cum += ideal
        ideal_rows.append((ideal, ideal_cum))

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
        return ("<tr>" +
            f"<td>{layer_idx}</td>" +
            f"<td>F{flow_idx}</td>" +
            f"<td>{html.escape(str(L.get('op','')).strip())}</td>" +
            f"<td>{ih}</td><td>{iw}</td><td>{ic}</td>" +
            f"<td>{oh}</td><td>{ow}</td><td>{oc}</td>" +
            f"<td>{kh}</td><td>{kw}</td><td>{sh}</td><td>{sw}</td>" +
            f"<td>{L.get('group','')}</td>" +
            f"<td>{th}×{toc}</td>" +
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
        return ("<tr>" +
                f"<td>{c['id']}</td>" +
                f"<td>{html.escape(c['op'])}</td>" +
                shape_cells +
                f"<td style='text-align:right'>{nelem}</td>" +
                f"<td>{html.escape(dtype)}</td>" +
                status_cell + "</tr>")
    compile_rows = "".join(_compile_row(c) for c in compile_rows_data)
    n_compile = len(compile_rows_data)
    n_skipped_compile = sum(1 for c in compile_rows_data
                            if not c["status"].startswith("ready"))

    title = f"MDLA7 profile — {model.name}"
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
</style></head>
<body>
<h1>{html.escape(title)}</h1>
<div>
  <span class="kv"><b>Model:</b> {html.escape(str(model.relative_to(REPO_ROOT)))}</span>
  <span class="kv"><b>Layers:</b> {n_total} (PASS {n_pass} / FAIL {n_fail})</span>
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
  <g id="gantt-axis"></g>
  <rect id="gantt-selrect" x="0" y="0" width="0" height="0"
        fill="rgba(66,135,245,0.18)" stroke="#4287f5" stroke-width="1"
        stroke-dasharray="3,2" style="display:none; pointer-events:none;"/>
</svg>
<script id="gantt-data" type="application/json">{gantt_data_json}</script>
<script>
(function() {{
  const data = JSON.parse(document.getElementById('gantt-data').textContent);
  // Pre-compute layer windows (start = previous layer's end).
  data.layers.forEach((L, i) => {{
    L.start = i > 0 ? data.layers[i-1].end : 0;
  }});

  const ENG_COLORS = {{
    udma:    '#4287f5',           // legacy (kept for older bins)
    udma_r:  '#4287f5',           // DRAM → L1 (load): blue
    udma_w:  '#7c4dff',           // L1  → DRAM (store): purple-blue
    conv:    '#e84545',
    requant: '#f5a142',
    ewe:     '#3ec56e',
    pool:    '#a945e8',
  }};
  const svg     = document.getElementById('gantt-svg');
  const gGrid   = document.getElementById('gantt-grid');
  const gOps    = document.getElementById('gantt-ops');
  const gLayers = document.getElementById('gantt-layers');
  const gScope  = document.getElementById('gantt-scope-hi');
  const gBars   = document.getElementById('gantt-bars');
  const gAxis   = document.getElementById('gantt-axis');
  const selRect = document.getElementById('gantt-selrect');
  const tip     = document.getElementById('gantt-tip');
  const cursorInfo = document.getElementById('gantt-cursor');
  const windowInfo = document.getElementById('gantt-window');
  const matchInfo  = document.getElementById('gantt-matches');
  const scopeInput = document.getElementById('gantt-scope');

  const eng_names = Object.keys(data.engines);
  const N_LANES = eng_names.length;
  // v8.3 / v8.9 geometry: TOP → OP_ROW (28 px) → engine lanes (56 px each)
  // → axis (32 px). The OP_ROW shows one rect per layer with the op id and
  // name; click an op rect to zoom to that layer.
  const LANE_H = 56, LANE_PAD = 8, LEFT = 72, RIGHT = 12, TOP = 8, AXIS_H = 32;
  const OP_ROW_H = 34;
  const total = Math.max(1, data.total | 0);

  // viewState: visible cycle window.
  let view0 = 0, view1 = total;
  // scope highlight: which layer (if any) is currently focused.
  let highlight = null;       // {{start, end, id, op}} or null

  function fmt(n) {{ return n.toLocaleString(); }}
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
        bars += `<rect class="gantt-task" x="${{x0}}" y="${{yy}}" width="${{w}}" height="${{hh}}" `
              + `fill="${{color}}" data-eng="${{name}}" data-s="${{s}}" data-e="${{e}}" `
              + `data-label="${{esc(label)}}" data-bytes="${{esc(bytes)}}"/>`;
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

  // ----- Mouse-drag: zoom-to-range (default), or pan with alt/right-button -----
  let mode = 'idle';   // 'select' | 'pan'
  let startX = 0, startView0 = 0;

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
      cursorInfo.textContent = 'cycle: ' + fmt(Math.round(pxToCyc(px, W)));
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
      }}
      selRect.style.display = 'none';
    }}
    mode = 'idle';
    svg.style.cursor = 'crosshair';
  }});

  // Tooltip on task / op-rect hover (uses bubbling so it works while drag is active).
  svg.addEventListener('mousemove', (ev) => {{
    if (mode !== 'idle') {{ tip.style.opacity = '0'; return; }}
    const t = ev.target;
    if (t && t.classList && t.classList.contains('gantt-task')) {{
      const s = +t.getAttribute('data-s');
      const e = +t.getAttribute('data-e');
      const label = t.getAttribute('data-label') || '';
      const bytes = +(t.getAttribute('data-bytes') || 0);
      let text = `${{t.getAttribute('data-eng')}}`;
      if (label) text += `\\n${{label}}`;
      text += `\\n${{fmt(s)}} → ${{fmt(e)}} cyc\\nΔ ${{fmt(e - s)}} cyc`;
      if (bytes) text += `\\n${{(bytes / 1024).toFixed(1)}} KB`;
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
  }});
  svg.addEventListener('mouseleave', () => {{ tip.style.opacity = '0'; }});

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
<th>id</th><th>flow</th><th>op</th><th>iH</th><th>iW</th><th>iC</th><th>oH</th><th>oW</th><th>oC</th>
<th>kH</th><th>kW</th><th>sH</th><th>sW</th><th>group</th>
<th>tiles<br>(H×OC)</th>
<th>cyc/layer</th><th>cyc/cum</th>
<th>ideal<br>cyc/layer</th><th>ideal<br>cyc/cum</th>
<th>conv<br>MAC util</th><th>conv<br>occupancy</th>
<th>DRAM r</th><th>DRAM w</th><th>SRAM r</th><th>SRAM w</th><th>verify</th>
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


def run_all_layers(model: Path, l1_timing: str = "fast") -> bool:
    """Run every CONV_2D layer in the model independently and aggregate."""
    print(f"\n========  {model.relative_to(REPO_ROOT)}  (all CONV_2D layers)  ========")
    if not model.exists():
        print(f"  → SKIP  (not found: {model})")
        return False
    n = count_convs(model)
    if n == 0:
        print(f"  → SKIP  (no CONV_2D ops or extractor failed)")
        return False
    print(f"  {n} CONV_2D layers found\n")
    pass_n = fail_n = skip_n = 0
    total_t = 0.0
    for i in range(n):
        ok, err = extract(model, i, verbose=False)
        if not ok:
            print(f"   layer {i:>2d}: SKIP  ({err})")
            skip_n += 1
            continue
        sim_ok, t, last = simulate(verbose=False, l1_timing=l1_timing)
        total_t += t
        # one-line per layer
        meta = _peek_blob_meta(BLOB_PATH) or "?"
        if sim_ok:
            print(f"   layer {i:>2d}: PASS  {meta}   ({t:.2f}s)")
            pass_n += 1
        else:
            print(f"   layer {i:>2d}: FAIL  {meta}   ({t:.2f}s, {last})")
            fail_n += 1
    print(f"\n  summary: {pass_n} PASS, {fail_n} FAIL, {skip_n} SKIP   "
          f"(total sim wall {total_t:.2f}s)")
    return fail_n == 0


def _peek_blob_meta(p: Path) -> str | None:
    """Return a compact 'in=... k=... out=...' string from the blob header."""
    try:
        import struct
        with open(p, "rb") as f:
            magic, _v = struct.unpack("<II", f.read(8))
            if magic != 0x374C444D:
                return None
            H, W, Cin   = struct.unpack("<HHH", f.read(6))
            OH, OW, OC  = struct.unpack("<HHH", f.read(6))
            kh, kw, *_  = struct.unpack("<8B",  f.read(8))
            return f"in={H}x{W}x{Cin}  k={kh}x{kw}  out={OH}x{OW}x{OC}"
    except OSError:
        return None


def main():
    ap = argparse.ArgumentParser(description="MDLA7 end-to-end runner")
    ap.add_argument("model", nargs="?", default=None,
                    help="Path to .tflite (default: efficientnet_lite0_int8)")
    ap.add_argument("-m", "--model-flag", dest="model_flag", default=None,
                    help="Same as positional, but explicit flag")
    ap.add_argument("--no-build", action="store_true",
                    help="Skip running `make` first")
    ap.add_argument("--list", action="store_true",
                    help="List all bundled models")
    ap.add_argument("--all", action="store_true",
                    help="Run every INT8 model (regression sweep)")
    ap.add_argument("--all-layers", action="store_true",
                    help="Run every CONV_2D layer of the chosen model")
    ap.add_argument("-l", "--layer", type=int, default=0,
                    help="Run a specific CONV_2D layer index (default: 0)")
    ap.add_argument("--keep-intermediate", action="store_true",
                    help="Keep .bin and .profile.json intermediates "
                         "(default: deleted on successful run; .csv/.png/.html stay)")
    ap.add_argument("--l1-timing", choices=("fast", "conflict", "mesh"), default="fast",
                    help="L1Mesh timing mode: fast aggregate estimate (default) "
                         "or per-bank SRAM port conflict model, or 4x4 mesh router/link conflict model")
    args = ap.parse_args()
    global _keep_intermediate
    _keep_intermediate = bool(args.keep_intermediate)

    if args.list:
        list_models(); return

    check_python_deps()

    if not args.no_build:
        build()
        if not TEST_BIN.exists():
            sys.exit(f"binary missing after build: {TEST_BIN}")

    if args.all:
        models = sorted(m for m in (MODEL_ROOT / "INT8").glob("*.tflite")
                        if not m.name.startswith("._"))
        ok_count, fail_count = 0, 0
        for m in models:
            if run_one(m, l1_timing=args.l1_timing):
                ok_count += 1
            else:
                fail_count += 1
        print(f"\n=== summary: {ok_count} PASS, {fail_count} FAIL ===")
        sys.exit(0 if fail_count == 0 else 1)

    arg = args.model_flag or args.model
    if arg is None:
        target = Path(DEFAULT_MODEL).resolve()
    else:
        target = resolve_model(arg)
    # Default = compile entire model and run one full sim (the architecture
    # the user asked for: tflite → host → cmdeng → engines, bit-true + cycles).
    # --layer N reverts to the legacy single-layer path.
    if args.layer:
        ok = run_one(target, layer=args.layer, l1_timing=args.l1_timing)
    elif args.all_layers:
        ok = run_all_layers(target, l1_timing=args.l1_timing)  # legacy: N independent sims
    else:
        ok = compile_and_run(target, l1_timing=args.l1_timing)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
