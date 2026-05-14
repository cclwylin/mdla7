#!/usr/bin/env python3
"""Run verilog host-driven byte-moving programs from MDL7 .bin files."""

from __future__ import annotations

import argparse
import math
import glob
import html
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


FILTER_ALIASES = {
    "slice": "ETHZ_v6_slice",
    "ethz_slice": "ETHZ_v6_slice",
    "ethz-v6-slice": "ETHZ_v6_slice",
    "ethz_v6_slice": "ETHZ_v6_slice",
    "ethz": "ETHZ_v6",
    "ethz_v6": "ETHZ_v6",
    "ethz-v6": "ETHZ_v6",
    "hotspot": "Hotspot",
    "bmm": "BMM",
    "unit": "UnitTest",
    "unittest": "UnitTest",
    "unit_test": "UnitTest",
    "unit_tflite": "UnitTest",
}


AUTO_COMPILE_CORPORA = {"BMM", "ETHZ_v6"}
COMPILE_FAILURES: list[str] = []


def record_compile_failure(model: Path, out_bin: Path, reason: str, output: str = "") -> None:
    msg = f"{model.name} -> {out_bin}: {reason}"
    COMPILE_FAILURES.append(msg)
    print(
        f"[run_verilog] ERROR: compile_model failed for {model.name}; "
        f"refusing stale {out_bin}",
        file=sys.stderr,
    )
    if output.strip():
        print(output.strip(), file=sys.stderr)


def compile_corpus_bins(repo_root: Path, rtl_dir: Path, corpus: str,
                        only_stems: set[str] | None = None) -> bool:
    """Keep corpus Verilog bins reproducible when compile_model.py changed."""
    if corpus not in AUTO_COMPILE_CORPORA:
        return True
    model_dir = repo_root / "model" / corpus
    out_dir = rtl_dir / "bin" / corpus
    compile_py = repo_root / "systemc" / "scripts" / "compile_model.py"
    venv_py = Path(os.environ.get("MDLA7_VENV", Path.home() / ".venvs/mdla7")).expanduser() / "bin" / "python"
    py = venv_py if venv_py.exists() else Path(sys.executable)
    if not model_dir.is_dir() or not compile_py.exists():
        return True
    out_dir.mkdir(parents=True, exist_ok=True)
    ok = True
    for model in sorted(model_dir.glob("*.tflite")):
        if model.name.startswith("._"):
            continue
        if only_stems is not None and model.stem not in only_stems:
            continue
        out_bin = out_dir / f"{model.stem}.bin"
        try:
            newest_input = max(model.stat().st_mtime_ns,
                               compile_py.stat().st_mtime_ns)
            if out_bin.exists() and out_bin.stat().st_mtime_ns >= newest_input:
                continue
            proc = subprocess.run(
                [str(py), str(compile_py), str(model), str(out_bin)],
                cwd=repo_root,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
        except OSError as exc:
            ok = False
            record_compile_failure(model, out_bin, f"could not start compiler: {exc}")
            continue
        if proc.returncode != 0:
            ok = False
            record_compile_failure(model, out_bin, f"rc={proc.returncode}", proc.stdout)
    return ok


def refresh_known_corpus_bin(repo_root: Path, rtl_dir: Path, candidate: Path) -> bool:
    try:
        rel = candidate.resolve().relative_to((rtl_dir / "bin").resolve())
    except ValueError:
        return True
    if len(rel.parts) != 2:
        return True
    corpus, filename = rel.parts
    if corpus not in AUTO_COMPILE_CORPORA or not filename.endswith(".bin"):
        return True
    return compile_corpus_bins(repo_root, rtl_dir, corpus, {Path(filename).stem})
PASS_RE = re.compile(
    r"PASS: verilog host-driven .* issued=([0-9]+) done=([0-9]+)"
    r"(?:\s+(?:verilog_cycles|vf_cycles)=([0-9]+))?"
)
PERF_RE = re.compile(
    r"perf_total=([0-9]+)\s+perf_conv=([0-9]+)\s+"
    r"perf_requant=([0-9]+)\s+perf_ewe=([0-9]+)\s+"
    r"perf_pool=([0-9]+)\s+perf_tnps=([0-9]+)\s+"
    r"perf_udma_r=([0-9]+)\s+perf_udma_w=([0-9]+)"
)
FAIL_RE = re.compile(r"(HOST_VERILOG_FAIL:.*|FAIL: verilog host program.*)")
SIM_FINISH_RE = re.compile(
    r"Verilator:\s+\$finish at\s+([0-9]+(?:\.[0-9]+)?)\s*([munpf]?s)",
    re.IGNORECASE,
)
GEN_STATS_RE = re.compile(
    r"commands=([0-9]+)"
    r"(?:\s+conv=([0-9]+))?"
    r"(?:\s+pool=([0-9]+))?"
    r"(?:\s+requant=([0-9]+))?"
    r"(?:\s+ewe=([0-9]+))?"
    r"\s+tnps=([0-9]+)\s+udma=([0-9]+)"
    r"(?:\s+refcrc=([0-9]+))?"
    r"(?:\s+sramcrc=([0-9]+))?"
    r"(?:\s+refbytes=([0-9]+))?"
    r"(?:\s+srambytes=([0-9]+))?"
    r"(?:\s+finalcrc=([0-9]+))?"
    r"(?:\s+finalbytes=([0-9]+))?"
)
CACHE_VERSION = 16
WORDS_PER_COMMAND = 32
DEFAULT_MAX_COMMANDS = 4096
MDL7_MAGIC = 0x374C444D
SMF_FINAL_TILE = 0x10
PROGRAM_COL_WIDTH = 34
RATIO_COL_WIDTH = 14
SYNTH_CLOCK_HZ = 1_900_000_000.0
VERILOG_CLOCK_HZ = 100_000_000.0
OP_NAMES = {
    0: "DONE",
    1: "CONV",
    2: "REQUANT",
    3: "EWE",
    4: "POOL",
    5: "TNPS",
    6: "UDMA",
    7: "L1CRC",
}

TIME_UNIT_TO_MS = {
    "s": 1000.0,
    "ms": 1.0,
    "us": 0.001,
    "ns": 0.000001,
    "ps": 0.000000001,
    "fs": 0.000000000001,
}


@dataclass(frozen=True)
class BinInfo:
    size_bytes: int
    layers: int
    tensor_bytes: int
    ref_bytes: int
    final_ref_bytes: int
    pattern_class: str
    timeout_s: float


def repo_paths() -> tuple[Path, Path]:
    script = Path(__file__).resolve()
    repo_root = script.parents[1]
    return repo_root, repo_root / "rtl"


def display(path: Path, base: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(path)


def has_glob(pattern: str) -> bool:
    return any(ch in pattern for ch in "*?[]")


def collect_from_base(base: Path, pattern: str) -> list[Path]:
    if has_glob(pattern):
        return [Path(p) for p in glob.glob(str(base / pattern), recursive=True)]
    candidate = base / pattern
    return [candidate] if candidate.is_file() else []


def collect_bins(filters: list[str], rtl_dir: Path, repo_root: Path, cwd: Path) -> list[Path]:
    bin_root = rtl_dir / "bin"
    patterns = filters or ["slice"]
    found: dict[Path, Path] = {}

    for raw in patterns:
        pattern = raw.strip()
        if not pattern:
            continue
        alias = FILTER_ALIASES.get(pattern.lower())
        candidates: list[Path] = []
        if alias is not None:
            if alias in AUTO_COMPILE_CORPORA:
                compile_corpus_bins(repo_root, rtl_dir, alias)
            candidates.extend((bin_root / alias).rglob("*.bin"))
        else:
            p = Path(pattern)
            has_path = p.is_absolute() or "/" in pattern or os.sep in pattern
            if p.is_absolute():
                if has_glob(pattern):
                    candidates.extend(Path(x) for x in glob.glob(pattern, recursive=True))
                elif p.is_file():
                    candidates.append(p)
            elif has_path:
                for base in (repo_root, bin_root, cwd):
                    candidates.extend(collect_from_base(base, pattern))
            else:
                if has_glob(pattern):
                    candidates.extend(bin_root.rglob(pattern))
                elif pattern.endswith(".bin"):
                    candidates.extend(bin_root.rglob(pattern))
                else:
                    candidates.extend(bin_root.rglob(f"*{pattern}*.bin"))
                    local = cwd / pattern
                    if local.is_file():
                        candidates.append(local)

        for candidate in candidates:
            resolved = candidate.resolve()
            refresh_ok = refresh_known_corpus_bin(repo_root, rtl_dir, resolved)
            if resolved.is_file() and resolved.suffix == ".bin" and not resolved.name.startswith("._"):
                if refresh_ok:
                    found[resolved] = resolved

    return sorted(found.values(), key=lambda x: str(x))


def file_signature(path: Path) -> dict[str, int | str]:
    st = path.stat()
    return {
        "path": str(path.resolve()),
        "size": st.st_size,
        "mtime_ns": st.st_mtime_ns,
    }


def rd32(data: bytes, off: int) -> int:
    if off + 4 > len(data):
        return 0
    return int.from_bytes(data[off:off + 4], "little")


def classify_bin(program: Path) -> BinInfo:
    st = program.stat()
    size_bytes = st.st_size
    layers = 0
    tensor_bytes = 0
    ref_bytes = 0
    final_ref_bytes = 0
    try:
        with program.open("rb") as f:
            header = f.read(16)
            if len(header) == 16 and rd32(header, 0) == MDL7_MAGIC:
                layers = rd32(header, 8)
                table = f.read(layers * 64)
                for idx in range(layers):
                    off = idx * 64
                    if off + 44 > len(table):
                        break
                    tensor_bytes += rd32(table, off + 32)
                    tensor_bytes += rd32(table, off + 36)
                    layer_ref_bytes = rd32(table, off + 40)
                    tensor_bytes += layer_ref_bytes
                    ref_bytes += layer_ref_bytes
                    final_ref_bytes = layer_ref_bytes
    except OSError:
        pass

    score = max(size_bytes, tensor_bytes)
    if layers >= 200 or score >= 256 * 1024 * 1024:
        pattern_class = "huge"
        timeout_s = 3600.0
    elif layers >= 80 or score >= 96 * 1024 * 1024:
        pattern_class = "large"
        timeout_s = 1800.0
    elif layers >= 25 or score >= 24 * 1024 * 1024:
        pattern_class = "medium"
        timeout_s = 900.0
    else:
        pattern_class = "small"
        timeout_s = 300.0
    return BinInfo(size_bytes, layers, tensor_bytes, ref_bytes,
                   final_ref_bytes, pattern_class, timeout_s)


def fmt_timeout(value: float) -> str:
    return f"{int(value)}s" if float(value).is_integer() else f"{value:.1f}s"


def fmt_program_name(stem: str) -> str:
    if len(stem) <= PROGRAM_COL_WIDTH:
        return f"{stem:<{PROGRAM_COL_WIDTH}}"
    return stem[:PROGRAM_COL_WIDTH - 3] + "..."


def fmt_ms(value: float | None) -> str:
    return "" if value is None else f"{value:.3f}"


def fmt_cycles(value: int | None) -> str:
    if value is None:
        return ""
    abs_value = abs(value)
    if abs_value >= 1_000_000:
        scaled = value / 1_000_000.0
        text = f"{scaled:.1f}".rstrip("0").rstrip(".")
        return f"{text}M"
    if abs_value >= 1_000:
        scaled = value / 1_000.0
        text = f"{scaled:.1f}".rstrip("0").rstrip(".")
        return f"{text}K"
    return str(value)


def fmt_ratio(num: float | None, den: float | None) -> str:
    if num is None or den is None or den == 0:
        return ""
    return f"{num / den:.2f}x"


def parse_verilog_ms(output: str) -> float | None:
    matches = SIM_FINISH_RE.findall(output)
    if not matches:
        return None
    value_s, unit = matches[-1]
    scale = TIME_UNIT_TO_MS.get(unit.lower())
    if scale is None:
        return None
    return float(value_s) * scale


def parse_verilog_cycles(output: str) -> int | None:
    ms = parse_verilog_ms(output)
    if ms is None:
        return None
    return int(round((ms / 1000.0) * VERILOG_CLOCK_HZ))


def parse_perf_counters(output: str) -> dict[str, int]:
    match = PERF_RE.search(output)
    if not match:
        return {}
    keys = ["total", "conv", "requant", "ewe", "pool", "tnps", "udma_r", "udma_w"]
    return {key: int(match.group(idx + 1)) for idx, key in enumerate(keys)}


def has_tensor_crc(refcrc: int, sramcrc: int, finalcrc: int) -> bool:
    return (refcrc + sramcrc + finalcrc) > 0


def coverage_label(info: BinInfo, refcrc: int, sramcrc: int, finalbytes: int) -> str:
    if (refcrc + sramcrc + finalbytes) == 0:
        return "sample"
    if info.final_ref_bytes > 0 and finalbytes >= info.final_ref_bytes:
        return "full"
    return "partial"


def is_unit_test_bin(program: Path) -> bool:
    return "UnitTest" in program.parts


def is_bmm_bin(program: Path) -> bool:
    return "BMM" in program.parts


def is_ethz_bin(program: Path) -> bool:
    return "ETHZ_v6" in program.parts or "ETHZ_v6_slice" in program.parts


def should_check_materialized_layers(program: Path) -> bool:
    return is_bmm_bin(program) or is_ethz_bin(program)


def requires_full_output_coverage(program: Path) -> bool:
    return is_bmm_bin(program)


def is_perf_comparable(program: Path, info: BinInfo, finalbytes: int) -> bool:
    if is_unit_test_bin(program):
        return True
    return info.final_ref_bytes > 0 and finalbytes >= info.final_ref_bytes


def fmt_perf_ratio(program: Path, info: BinInfo, finalbytes: int,
                   verilog_cycles: int | None, synth_cycles: int | None) -> str:
    if not is_perf_comparable(program, info, finalbytes):
        return ""
    return fmt_ratio(verilog_cycles, synth_cycles)


def partial_coverage_reason(info: BinInfo, finalbytes: int) -> str:
    need = info.final_ref_bytes or info.ref_bytes
    if need > 0:
        return f"partial BMM coverage finalB={finalbytes}/{need}; full Verilog traversal not implemented"
    return f"partial BMM coverage finalB={finalbytes}; full Verilog traversal not implemented"


def profile_dir_for(program: Path) -> Path:
    return program.parent / "profile"


def infer_profile_corpus(bins: list[Path], rtl_dir: Path) -> str:
    corpora: set[str] = set()
    bin_root = (rtl_dir / "bin").resolve()
    for path in bins:
        try:
            rel = path.resolve().relative_to(bin_root)
        except ValueError:
            continue
        if rel.parts:
            corpora.add(rel.parts[0])
    return next(iter(corpora)) if len(corpora) == 1 else "verilog"


def profile_stem_for_corpus(corpus: str) -> str:
    mapping = {
        "BMM": "profile_bmm",
        "ETHZ_v6": "profile_ethz_v6",
        "ETHZ_v6_slice": "profile_ethz_v6_slice",
        "Hotspot": "profile_hotspot",
        "UnitTest": "profile_unittest",
        "verilog": "profile_verilog",
    }
    return mapping.get(corpus, f"profile_{corpus.lower()}")


def verilog_profile_path(args: argparse.Namespace, bins: list[Path],
                         options: set[str]) -> Path:
    corpus = infer_profile_corpus(bins, args.rtl_dir)
    stem = profile_stem_for_corpus(corpus)
    tags = ["verilog"]
    if "dpi" in options:
        tags.append("dpi")
    return args.repo_root / "batch" / "profile" / f"{stem}.{'.'.join(tags)}.html"


def verilog_output_dir(args: argparse.Namespace, options: set[str]) -> Path:
    return args.repo_root / "batch" / "output" / (
        "verilog-dpi" if "dpi" in options else "verilog"
    )


def verilog_report_path(args: argparse.Namespace, options: set[str], stem: str) -> Path:
    return verilog_output_dir(args, options) / f"{stem}.html"


def _estimate_command_cycles(words: list[int]) -> int:
    op = words[0] & 0xF
    byte_count = max(words[1] & 0xFFFFFFFF, words[4] & 0xFFFFFFFF,
                     words[29] & 0xFFFFFFFF)
    if op == 6:
        return max(1, math.ceil(byte_count / 16) + 8)
    if op == 7:
        return max(1, math.ceil(byte_count / 16) + 4)
    if op in (1, 2, 3, 4, 5):
        return max(1, min(byte_count, 1_000_000) + 8)
    return 1


def decode_verilog_commands(hex_path: Path, verilog_cycles: int | None) -> list[dict[str, object]]:
    if not hex_path.exists():
        return []
    words: list[int] = []
    for line in hex_path.read_text(encoding="ascii").splitlines():
        line = line.strip()
        if line:
            words.append(int(line, 16))
    raw_rows: list[dict[str, object]] = []
    raw_total = 0
    for idx, off in enumerate(range(0, len(words), WORDS_PER_COMMAND), 1):
        chunk = words[off:off + WORDS_PER_COMMAND]
        if len(chunk) < WORDS_PER_COMMAND:
            break
        op = chunk[0] & 0xF
        if op == 0:
            break
        est = _estimate_command_cycles(chunk)
        raw_rows.append({
            "idx": idx,
            "op": op,
            "op_name": OP_NAMES.get(op, f"OP{op}"),
            "layer": chunk[19] & 0xFFFFFFFF,
            "bytes": chunk[1] & 0xFFFFFFFF,
            "addr": chunk[2] & 0xFFFFFFFF,
            "flags": chunk[3] & 0xFFFFFFFF,
            "stage": (chunk[3] >> 24) & 0xFF,
            "offset": chunk[27] & 0xFFFFFFFF,
            "crc_bytes": chunk[29] & 0xFFFFFFFF,
            "raw_cycles": est,
        })
        raw_total += est
    if not raw_rows:
        return []
    scale = (float(verilog_cycles) / raw_total) if verilog_cycles and raw_total > 0 else 1.0
    cursor = 0.0
    for row in raw_rows:
        dur = max(1.0, float(row["raw_cycles"]) * scale)
        row["start"] = int(round(cursor))
        cursor += dur
        row["end"] = int(round(cursor))
        row["cycles"] = max(1, int(row["end"]) - int(row["start"]))
    return raw_rows


def write_verilog_report_html(path: Path, *, title: str, mode: str,
                              bin_path: Path, hex_path: Path,
                              systemc_report: Path | None,
                              ans: str, cov: str, reason: str,
                              synth_ms: float | None,
                              synth_cycles: int | None,
                              verilog_cycles: int | None,
                              perf_counters: dict[str, int] | None,
                              wall: float,
                              commands: list[dict[str, object]]) -> None:
    try:
        model_rel = bin_path.resolve().relative_to((bin_path.parents[2]).resolve())
    except (ValueError, IndexError):
        model_rel = bin_path.name

    def link_to(target: Path | None, label: str) -> str:
        if target is None:
            return ""
        href = os.path.relpath(target.resolve(), path.parent).replace(os.sep, "/")
        return f'<a href="{html.escape(href)}">{html.escape(label)}</a>'

    total_cycles = verilog_cycles or max((int(row.get("end") or 0) for row in commands), default=0)
    total_cycles = max(total_cycles, 1)
    max_table_rows = 500
    table_rows: list[str] = []
    engine_cycles: dict[str, int] = {}
    engine_tasks: dict[str, list[dict[str, object]]] = {}
    layer_stats: dict[int, dict[str, object]] = {}

    def engine_for(row: dict[str, object]) -> str:
        op_name = str(row.get("op_name") or "").lower()
        if op_name == "udma":
            return "udma_w" if (int(row.get("flags") or 0) & 1) else "udma_r"
        if op_name == "l1crc":
            return "cmd"
        return op_name

    for row in commands:
        op_name = str(row.get("op_name") or "")
        cycles = int(row.get("cycles") or 0)
        engine = engine_for(row)
        layer = int(row.get("layer") or 0)
        bytes_n = int(row.get("bytes") or 0)
        crc_bytes = int(row.get("crc_bytes") or 0)
        engine_cycles[engine] = engine_cycles.get(engine, 0) + cycles
        engine_tasks.setdefault(engine, []).append(row)
        stats = layer_stats.setdefault(layer, {
            "ops": set(), "commands": 0, "cycles": 0,
            "dram_r": 0, "dram_w": 0, "sram_r": 0, "sram_w": 0,
            "verify": 0,
        })
        stats["ops"].add(op_name)  # type: ignore[index, union-attr]
        stats["commands"] = int(stats["commands"]) + 1
        stats["cycles"] = int(stats["cycles"]) + cycles
        flags = int(row.get("flags") or 0)
        is_write = bool(flags & 1)
        if op_name == "UDMA":
            if is_write:
                stats["dram_w"] = int(stats["dram_w"]) + bytes_n
                stats["sram_r"] = int(stats["sram_r"]) + bytes_n
            else:
                stats["dram_r"] = int(stats["dram_r"]) + bytes_n
                stats["sram_w"] = int(stats["sram_w"]) + bytes_n
        elif op_name == "L1CRC" or crc_bytes:
            stats["verify"] = int(stats["verify"]) + max(crc_bytes, bytes_n)
        elif op_name in ("CONV", "REQUANT", "EWE", "POOL", "TNPS"):
            stats["sram_r"] = int(stats["sram_r"]) + bytes_n
            stats["sram_w"] = int(stats["sram_w"]) + bytes_n

    engine_order = ["cmd", "udma_r", "udma_w", "conv", "requant", "ewe", "pool", "tnps"]
    for engine in engine_order:
        engine_cycles.setdefault(engine, 0)
        engine_tasks.setdefault(engine, [])
    if perf_counters:
        total_from_perf = int(perf_counters.get("total", 0) or 0)
        if total_from_perf > 0:
            total_cycles = total_from_perf
        for engine in engine_order:
            if engine == "cmd":
                continue
            engine_cycles[engine] = int(perf_counters.get(engine, 0) or 0)

    for row in commands[:max_table_rows]:
        op_name = str(row.get("op_name") or "")
        table_rows.append(
            "<tr>"
            f"<td class='num'>{html.escape(str(row.get('idx')))}</td>"
            f"<td>{html.escape(op_name)}</td>"
            f"<td class='num'>{html.escape(str(row.get('layer')))}</td>"
            f"<td class='num'>{html.escape(str(row.get('start')))}</td>"
            f"<td class='num'>{html.escape(str(row.get('end')))}</td>"
            f"<td class='num'>{html.escape(str(row.get('cycles')))}</td>"
            f"<td class='num'>{html.escape(str(row.get('bytes')))}</td>"
            f"<td class='num'>0x{int(row.get('addr') or 0):x}</td>"
            f"<td class='num'>0x{int(row.get('flags') or 0):08x}</td>"
            f"<td class='num'>{html.escape(str(row.get('stage')))}</td>"
            f"<td class='num'>{html.escape(str(row.get('offset')))}</td>"
            f"<td class='num'>{html.escape(str(row.get('crc_bytes')))}</td>"
            "</tr>"
        )
    svg_w = 1200
    left_pad = 86
    right_pad = 24
    top_pad = 18
    lane_h = 22
    plot_w = svg_w - left_pad - right_pad
    svg_h = top_pad + len(engine_order) * lane_h + 30
    axis_rows = []
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        x = left_pad + frac * plot_w
        cyc = int(round(frac * total_cycles))
        axis_rows.append(
            f"<line x1='{x:.1f}' y1='{top_pad}' x2='{x:.1f}' y2='{svg_h-24}' stroke='#e5e5e5'/>"
            f"<text x='{x:.1f}' y='{svg_h-8}' text-anchor='middle' font-size='10' fill='#777'>{cyc:,}</text>"
        )
    gantt_bars = []
    colors = {
        "cmd": "#777777", "udma_r": "#c75555", "udma_w": "#d9854f",
        "conv": "#3f7fbf", "requant": "#6f9f45", "ewe": "#9a6cc8",
        "pool": "#bf7a30", "tnps": "#2f9d8f",
    }
    for lane_idx, engine in enumerate(engine_order):
        y = top_pad + lane_idx * lane_h
        gantt_bars.append(
            f"<text x='8' y='{y+14}' font-size='11' fill='#333'>{html.escape(engine)}</text>"
            f"<rect x='{left_pad}' y='{y+2}' width='{plot_w}' height='{lane_h-5}' fill='#f1f1f1'/>"
        )
        for row in engine_tasks.get(engine, [])[:max_table_rows]:
            start = int(row.get("start") or 0)
            cycles = max(1, int(row.get("cycles") or 1))
            x = left_pad + (start * plot_w / total_cycles)
            w = max(1.0, cycles * plot_w / total_cycles)
            op_name = str(row.get("op_name") or "")
            title_text = (
                f"cmd {row.get('idx')} {op_name} L{row.get('layer')} "
                f"{start}-{start + cycles} cyc"
            )
            gantt_bars.append(
                f"<rect x='{x:.2f}' y='{y+3}' width='{w:.2f}' height='{lane_h-7}' "
                f"fill='{colors.get(engine, '#7aa6d9')}' opacity='0.9'>"
                f"<title>{html.escape(title_text)}</title></rect>"
            )
    gantt_svg = (
        f"<svg id='gantt-svg' width='100%' viewBox='0 0 {svg_w} {svg_h}' "
        "style='border:1px solid #ddd; background:#fafafa; cursor:crosshair; display:block; user-select:none;'>"
        f"<g id='gantt-grid'>{''.join(axis_rows)}</g>"
        f"<g id='gantt-bars'>{''.join(gantt_bars)}</g>"
        "</svg>"
    )
    engine_rows = "".join(
        "<tr>"
        f"<td>{html.escape(name)}</td>"
        f"<td style='text-align:right'>{cycles:,}</td>"
        f"<td style='text-align:right'>{(cycles * 100.0 / total_cycles):.1f}%</td>"
        "</tr>"
        for name, cycles in ((name, engine_cycles.get(name, 0)) for name in engine_order)
    )
    # Map descriptor op_name to the engine lane label used in the timeline.
    OP_NAME_TO_ENGINE = {
        "CONV": "conv", "REQUANT": "requant", "EWE": "ewe",
        "POOL": "pool", "TNPS": "tnps",
        "UDMA": "udma", "L1CRC": "cmd",
    }
    layer_rows = []
    cum = 0
    for layer, stats in sorted(layer_stats.items()):
        cyc = int(stats["cycles"])
        cum += cyc
        ops_sorted = sorted(stats["ops"])  # type: ignore[arg-type]
        ops = ",".join(ops_sorted)
        engines = sorted({OP_NAME_TO_ENGINE.get(op, op.lower()) for op in ops_sorted})
        engine_cell = ",".join(engines) if engines else "—"
        layer_rows.append(
            "<tr>"
            f"<td>{layer}</td><td>{html.escape(ops)}</td><td>{int(stats['commands'])}</td>"
            f"<td>{html.escape(engine_cell)}</td>"
            f"<td style='text-align:right'>{cyc:,}</td>"
            f"<td style='text-align:right'>{cum:,}</td>"
            "<td></td><td></td><td></td><td></td>"
            f"<td style='text-align:right'>{int(stats['dram_r']) / 1024:.1f}</td>"
            f"<td style='text-align:right'>{int(stats['dram_w']) / 1024:.1f}</td>"
            f"<td style='text-align:right'>{int(stats['sram_r']) / 1024:.1f}</td>"
            f"<td style='text-align:right'>{int(stats['sram_w']) / 1024:.1f}</td>"
            "<td></td><td></td><td></td><td></td><td></td><td></td>"
            "<td></td><td></td><td></td><td></td><td></td>"
            f"<td>{int(stats['commands'])}</td>"
            f"<td style='text-align:right'>{int(stats['verify']) / 1024:.1f}</td>"
            "</tr>"
        )
    links = " ".join(x for x in (
        link_to(bin_path, "bin"), link_to(hex_path, "hex"), link_to(systemc_report, "systemc")
    ) if x)
    by_op: dict[str, int] = {}
    for row in commands:
        by_op[str(row.get("op_name"))] = by_op.get(str(row.get("op_name")), 0) + 1
    op_summary = " ".join(
        f"<span><b>{html.escape(k)}:</b> {v}</span>"
        for k, v in sorted(by_op.items())
    )
    truncated = "" if len(commands) <= max_table_rows else (
        f"<p class='note'>showing first {max_table_rows} of {len(commands)} commands</p>"
    )
    verilog_ms = (float(verilog_cycles) / (VERILOG_CLOCK_HZ / 1000.0)
                  if verilog_cycles is not None else None)
    pass_flag = 1 if ans in ("PASS", "CACHED") else 0
    fail_flag = 1 if ans == "FAIL" else 0
    has_compute_engine = any(engine_cycles.get(e, 0) > 0
                             for e in ("conv", "requant", "ewe", "pool", "tnps"))
    graph_coverage = (
        "Generated Verilog descriptors include compute-engine commands"
        if has_compute_engine else
        "result-check only — generated host program contains UDMA/L1CRC commands, "
        "so compute-engine lanes have no bars"
    )
    doc = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>
body {{ font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
        max-width:1200px; margin:24px auto; padding:0 16px; color:#222; }}
h1 {{ font-size:20px; margin-bottom:4px; }}
h2 {{ font-size:15px; border-bottom:1px solid #ddd; padding-bottom:4px; margin-top:28px; }}
h3 {{ font-size:13px; margin:10px 0 4px; }}
table {{ border-collapse:collapse; font-size:12px; width:100%; }}
th,td {{ border:1px solid #e4e4e4; padding:4px 8px; text-align:left;
         font-variant-numeric:tabular-nums; }}
th {{ background:#f4f4f4; }}
table.sortable th {{ cursor:pointer; user-select:none; position:relative; padding-right:16px; }}
.kv {{ display:block; margin:2px 0; }}
.kv b {{ color:#556; font-weight:600; }}
.warn {{ color:#a06000; font-weight:600; }}
a {{ color:#1f5fa8; text-decoration:none; }}
a:hover {{ text-decoration:underline; }}
.num {{ text-align:right; white-space:nowrap; }}
.pass {{ color:#0a7d23; font-weight:600; }} .fail {{ color:#b00020; font-weight:600; }}
.note {{ color:#666; font-size:12px; }}
</style></head><body>
<h1>{html.escape(title)}</h1>
<div>
  <span class="kv"><b>Model:</b> {html.escape(str(model_rel))}</span>
  <span class="kv"><b>Mode:</b> {html.escape(mode)}</span>
  <span class="kv"><b>Compiled layers:</b> {len(layer_stats)} (PASS {pass_flag} / FAIL {fail_flag})</span>
  <span class="kv"><b>Graph coverage:</b> {html.escape(graph_coverage)}</span>
  <span class="kv"><b>Verification:</b> {html.escape(cov)} CRC coverage from generated host descriptors</span>
  <span class="kv"><b>Sim time:</b> {fmt_ms(verilog_ms)} ms @ 100 MHz ({verilog_cycles or 0:,} cycles)</span>
  <span class="kv"><b>CX reference:</b> {html.escape(fmt_ms(synth_ms))} ms @ 1.9 GHz ({html.escape(fmt_cycles(synth_cycles))} cycles)</span>
  <span class="kv"><b>Result:</b> <span class="{html.escape(ans.lower())}">{html.escape(ans)}</span> {html.escape(reason)}</span>
  <span class="kv"><b>Artifacts:</b> {links}</span>
</div>

<h2>Engine utilization</h2>
<table><thead><tr><th>engine</th><th>busy cycles</th><th>utilization</th></tr></thead>
<tbody>{engine_rows}</tbody></table>

<h2>Gantt timeline (interactive)</h2>
<div id="gantt-controls" style="font-size:12px; margin:6px 0;">
  <button id="gantt-zoomin">+</button>
  <button id="gantt-zoomout">-</button>
  <button id="gantt-reset">Reset</button>
  &nbsp;|&nbsp;
  <label>scope:
    <input id="gantt-scope" placeholder="op name - L&lt;id&gt; - range 3-6"
           style="width:200px; padding:1px 4px; font: 11px ui-monospace,Menlo,Consolas,monospace;"
           autocomplete="off"/></label>
  <button id="gantt-prev" title="previous match">&lt;</button>
  <button id="gantt-next" title="next match">&gt;</button>
  <span id="gantt-matches" style="margin-left:6px; color:#666;"></span>
  <br/>
  <span id="gantt-cursor" style="color:#555;">cycle: -</span>
  <span id="gantt-window" style="margin-left:12px; color:#888;">0 - {total_cycles:,}</span>
  <span style="float:right; color:#888;">hover bars for command details</span>
</div>
<h3>Original engine timeline</h3>
{gantt_svg}
{truncated}

<h2>Per-layer profile (sizes in KB)</h2>
<table id="profile-tbl" class="sortable"><thead><tr>
<th>id</th><th>flow</th><th>op</th><th>engine</th>
<th>cyc/layer</th><th>cyc/cum</th>
<th>ideal<br>cyc/layer</th><th>ideal<br>cyc/cum</th>
<th>conv<br>MAC util</th><th>conv<br>occupancy</th>
<th>DRAM r</th><th>DRAM w</th><th>SRAM r</th><th>SRAM w</th>
<th>iH</th><th>iW</th><th>iC</th><th>oH</th><th>oW</th><th>oC</th>
<th>kH</th><th>kW</th><th>sH</th><th>sW</th><th>group</th>
<th>tiles<br>(HxOC)</th>
<th>verify</th>
</tr></thead>
<tbody>{''.join(layer_rows)}</tbody></table>

<h2>Compile log ({len(commands)} commands, 0 skipped)</h2>
<table id="compile-tbl" class="sortable">
  <thead><tr><th>idx</th><th>op</th><th>layer</th><th>start</th><th>end</th>
  <th>cycles</th><th>bytes</th><th>addr</th><th>flags</th><th>stage</th>
  <th>offset</th><th>crc_bytes</th></tr></thead>
  <tbody>{''.join(table_rows)}</tbody>
</table>
</body></html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(doc)


def write_verilog_profile_html(path: Path, title: str, rows: list[dict[str, object]],
                               *, mode: str, cache_hits: int,
                               totals: dict[str, object]) -> None:
    def text(value: object) -> str:
        return "" if value is None else str(value)

    def rel_link(value: object, label: str) -> str:
        raw = text(value)
        if not raw:
            return ""
        raw_path = Path(raw)
        if raw_path.is_absolute():
            raw = os.path.relpath(raw_path, path.parent).replace(os.sep, "/")
        return f'<a href="{html.escape(raw)}">{html.escape(label)}</a>'

    body: list[str] = []
    for row in rows:
        program = text(row.get("program"))
        main_link = rel_link(row.get("verilog_link"), program) or html.escape(program)
        verilog_cycles = row.get("verilog_cycles_raw")
        verilog_ms = row.get("verilog_ms")
        metric = fmt_ms(verilog_ms if isinstance(verilog_ms, float) else None)
        if verilog_cycles is not None:
            metric = f"{metric} ({html.escape(text(row.get('verilog_cyc')))})" if metric else html.escape(text(row.get("verilog_cyc")))
        body.append(
            "<tr>"
            f"<td>{html.escape(program)}</td>"
            f"<td>{main_link}</td>"
            f"<td class='num'>{metric}</td>"
            "</tr>"
        )

    summary = " ".join(
        f"<span><b>{html.escape(k)}:</b> {html.escape(text(v))}</span>"
        for k, v in totals.items()
    )
    doc = f"""<!doctype html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
:root {{ color-scheme:light; --bg:#f7f8fa; --panel:#ffffff; --line:#d8dde6;
  --text:#17202a; --muted:#657080; --head:#eef2f7; --link:#0b5cad; }}
body {{ margin:0; font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
       color:var(--text); background:var(--bg); }}
main {{ max-width:1180px; margin:0 auto; padding:24px; }}
h1 {{ margin:0 0 12px; font-size:24px; }}
.bar {{ display:flex; align-items:center; gap:10px; margin:0 0 14px; flex-wrap:wrap; }}
.meta {{ color:var(--muted); font-size:12px; display:flex; flex-wrap:wrap; gap:8px 18px; }}
.meta b {{ color:var(--text); }}
table {{ width:100%; border-collapse:collapse; background:var(--panel);
        border:1px solid var(--line); }}
th,td {{ padding:8px 10px; border-bottom:1px solid var(--line); text-align:left; vertical-align:top; }}
th {{ background:var(--head); position:sticky; top:0; z-index:1; }}
th.pattern {{ width:32%; min-width:260px; }}
td.num, th.num {{ text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap; }}
a {{ color:var(--link); text-decoration:none; }}
a:hover {{ text-decoration:underline; }}
tr:hover td {{ background:#f4f7fb; }}
</style></head><body>
<main>
<h1>{html.escape(title)}</h1>
<div class="bar meta">
  <span><b>mode:</b> {html.escape(mode)}</span>
  <span><b>rows:</b> {len(rows)}</span>
  <span><b>cache_hits:</b> {cache_hits}</span>
  {summary}
</div>
<table>
  <thead><tr>
    <th class="pattern">pattern</th>
    <th>link</th>
    <th class="num">verilog</th>
  </tr></thead>
  <tbody>{''.join(body)}</tbody>
</table>
</main>
</body></html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(doc)


def profile_candidates(program: Path, profile_root: Path) -> list[Path]:
    stem = program.stem
    paths = [
        profile_dir_for(program) / f"{stem}.profile.json",
        profile_dir_for(program) / f"{stem}.synth.profile.json",
        profile_dir_for(program) / f"{stem}.mesh.profile.json",
        profile_root / f"{stem}.profile.json",
        profile_root / f"{stem}.cx.profile.json",
        profile_root / f"{stem}.synth.profile.json",
        profile_root / f"{stem}.mesh.profile.json",
        program.with_suffix(".profile.json"),
        program.with_suffix(".synth.profile.json"),
        program.with_suffix(".mesh.profile.json"),
    ]
    for mode in ("cx", "fast"):
        paths.extend([
            profile_root / mode / f"{stem}.profile.json",
            profile_root / mode / f"{stem}.{mode}.profile.json",
        ])
    return paths


def load_synth_cycles(program: Path, profile_root: Path) -> int | None:
    for path in profile_candidates(program, profile_root):
        if not path.exists() or path.name.startswith("._"):
            continue
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        summary = data.get("summary") if isinstance(data, dict) else None
        cycles = summary.get("total_cycles") if isinstance(summary, dict) else None
        if cycles is None:
            continue
        try:
            return int(cycles)
        except (TypeError, ValueError):
            continue
    return None


def synth_ms_from_cycles(cycles: int | None) -> float | None:
    if cycles is None:
        return None
    return float(cycles) / (SYNTH_CLOCK_HZ / 1000.0)


def load_synth_ms(program: Path, profile_root: Path) -> float | None:
    return synth_ms_from_cycles(load_synth_cycles(program, profile_root))


def latest_tree_mtime_ns(root: Path) -> int:
    mtimes: list[int] = []
    for pattern in ("*.v", "*.sv", "*.f"):
        mtimes.extend(p.stat().st_mtime_ns for p in root.glob(pattern) if p.is_file())
    return max(mtimes) if mtimes else 0


def load_cache(path: Path) -> dict[str, object]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"version": CACHE_VERSION, "entries": {}}
    if not isinstance(data, dict) or data.get("version") != CACHE_VERSION:
        return {"version": CACHE_VERSION, "entries": {}}
    if not isinstance(data.get("entries"), dict):
        data["entries"] = {}
    return data


def save_cache(path: Path, cache: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(cache, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def cache_hit(
    entry: object,
    bin_path: Path,
    run_mode: str,
    runner_mtime_ns: int,
    generator_mtime_ns: int,
    source_mtime_ns: int,
) -> dict[str, object] | None:
    if not isinstance(entry, dict):
        return None
    if entry.get("status") not in ("PASS", "SKIP"):
        return None
    if entry.get("bin_sig") != file_signature(bin_path):
        return None
    if entry.get("run_mode") != run_mode:
        return None
    if entry.get("runner_mtime_ns") != runner_mtime_ns:
        return None
    if entry.get("generator_mtime_ns") != generator_mtime_ns:
        return None
    if entry.get("source_mtime_ns") != source_mtime_ns:
        return None
    return entry


def run(cmd: list[str], cwd: Path, timeout: float | None = None) -> tuple[int, str, float]:
    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
        return proc.returncode, proc.stdout or "", time.time() - t0
    except subprocess.TimeoutExpired as exc:
        return 124, exc.stdout or f"TIMEOUT after {timeout:.1f}s", time.time() - t0


def parse_args(argv: list[str]) -> argparse.Namespace:
    repo_root, rtl_dir = repo_paths()
    ap = argparse.ArgumentParser()
    ap.add_argument("--filter", action="append", default=[],
                    help="bin filter, glob, path, or alias: unittest, slice, ethz, hotspot, bmm")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=None,
                    help="Manual per-program timeout. Default: auto by bin size/layer class.")
    ap.add_argument("--max-commands", type=int, default=DEFAULT_MAX_COMMANDS)
    ap.add_argument("--no-build", action="store_true")
    ap.add_argument("--closed-loop-perf-target", action="store_true",
                    help="Pad closed-loop verilog cycles toward synth profile total_cycles for calibration/debug only.")
    ap.add_argument("--require-crc-coverage", action="store_true",
                    help="Fail if the run produces no refcrc/sramcrc coverage.")
    ap.add_argument("--min-ref-bytes", type=int, default=0,
                    help="Fail if total refB coverage is below this value.")
    ap.add_argument("--min-sram-bytes", type=int, default=0,
                    help="Fail if total sramB coverage is below this value.")
    ap.add_argument("--require-final-output-crc", action="store_true",
                    help="Fail a generated program if it has no final-layer SRAM/L1 CRC coverage.")
    ap.add_argument("--min-final-bytes", type=int, default=0,
                    help="Fail if total final-layer SRAM/L1 CRC bytes are below this value.")
    ap.add_argument("--conv-sram-window-commands", type=int, default=0,
                    help="Pass through SRAM-window command budget for oversized INT8 CONV layers.")
    ap.add_argument("--conv-sram-window-count", type=int, default=0,
                    help="Pass through max SRAM-window count for oversized INT8 CONV layers.")
    ap.add_argument("--conv-sample-count", type=int, default=0,
                    help="Pass through INT8 CONV closed-loop output sample count. Default: generator default.")
    ap.add_argument("--full-conv-coverage", action="store_true",
                    help="Pass through full INT8 CONV output coverage.")
    ap.add_argument("--check-all-layers", action="store_true",
                    help="Pass through per-layer closed-loop checks. Default checks final layer only.")
    ap.add_argument("--check-materialized-layers", action="store_true",
                    help="Pass through compact checks for materialized fallback layers.")
    ap.add_argument("--rerun-all", action="store_true",
                    help="Ignore cached PASS/SKIP results and rerun every matched .bin.")
    ap.add_argument("--cache-file", type=Path,
                    default=rtl_dir / "obj" / "verilog" / "cache.json",
                    help="Regression cache JSON. Default: rtl/obj/verilog/cache.json")
    ap.add_argument("--profile-root", type=Path, default=repo_root / "batch" / "output",
                    help="Profile directory for cx_ms/cx_cyc. Default: batch/output")
    ap.add_argument("--no-html", action="store_true",
                    help="Skip writing batch/profile/profile_*.verilog.html index.")
    ap.add_argument("--dpi", action="store_true",
                    help="Enable DPI datapath helpers (shorthand for --option dpi).")
    ap.add_argument("--option", action="append", default=[],
                    help="verilog option. Use dpi to enable DPI datapath helpers.")
    ap.set_defaults(repo_root=repo_root, rtl_dir=rtl_dir)
    args = ap.parse_args(argv)
    if args.dpi:
        args.option.append("dpi")
    return args


def normalized_options(options: list[str]) -> set[str]:
    out: set[str] = set()
    for raw in options:
        for item in raw.split(","):
            opt = item.strip().lower()
            if not opt:
                continue
            out.add(opt)
    return out


def run_mode(args: argparse.Namespace) -> str:
    options = normalized_options(args.option)
    suffix = "+dpi" if "dpi" in options else ""
    layer_mode = "+all-layers" if args.check_all_layers else "+final-layer"
    conv_mode = "+full-conv" if args.full_conv_coverage else ""
    sample_mode = f"+conv-samples-{args.conv_sample_count}" if args.conv_sample_count > 0 else ""
    return "closed_loop_dataflow" + suffix + layer_mode + conv_mode + sample_mode


def count_commands(hex_path: Path) -> tuple[int, int, int, int, int, int, int, int, int, int, int, int, int]:
    words: list[int] = []
    for line in hex_path.read_text(encoding="ascii").splitlines():
        line = line.strip()
        if not line:
            continue
        words.append(int(line, 16))
    count = 0
    conv = 0
    pool = 0
    requant = 0
    ewe = 0
    tnps = 0
    udma = 0
    refcrc = 0
    sramcrc = 0
    refbytes = 0
    srambytes = 0
    finalcrc = 0
    finalbytes = 0

    def has_final_crc_flag(desc_off: int) -> bool:
        return ((words[desc_off + 3] >> 24) & SMF_FINAL_TILE) != 0

    for off in range(0, len(words), WORDS_PER_COMMAND):
        op = words[off] & 0xF
        if op == 0:
            break
        count += 1
        if op == 5:
            tnps += 1
            if words[off + 3] & (1 << 10):
                sramcrc += 1
                srambytes += words[off + 29]
                if has_final_crc_flag(off):
                    finalcrc += 1
                    finalbytes += words[off + 29]
        elif op == 6:
            udma += 1
            if words[off + 3] & (1 << 10):
                sramcrc += 1
                srambytes += words[off + 29]
                if has_final_crc_flag(off):
                    finalcrc += 1
                    finalbytes += words[off + 29]
        elif op == 7:
            if words[off + 3] & (1 << 10):
                sramcrc += 1
                srambytes += words[off + 29]
                if has_final_crc_flag(off):
                    finalcrc += 1
                    finalbytes += words[off + 29]
        elif op == 1:
            conv += 1
            if words[off + 3] & (1 << 9):
                refcrc += 1
                refbytes += words[off + 29]
            if words[off + 3] & (1 << 10):
                sramcrc += 1
                srambytes += words[off + 29]
                if has_final_crc_flag(off):
                    finalcrc += 1
                    finalbytes += words[off + 29]
        elif op == 2:
            requant += 1
            if words[off + 3] & (1 << 10):
                sramcrc += 1
                srambytes += words[off + 29]
                if has_final_crc_flag(off):
                    finalcrc += 1
                    finalbytes += words[off + 29]
        elif op == 3:
            ewe += 1
            if words[off + 3] & (1 << 10):
                sramcrc += 1
                srambytes += words[off + 29]
                if has_final_crc_flag(off):
                    finalcrc += 1
                    finalbytes += words[off + 29]
        elif op == 4:
            pool += 1
            if words[off + 3] & (1 << 9):
                refcrc += 1
                refbytes += words[off + 29]
            if words[off + 3] & (1 << 10):
                sramcrc += 1
                srambytes += words[off + 29]
                if has_final_crc_flag(off):
                    finalcrc += 1
                    finalbytes += words[off + 29]
    return count, conv, pool, requant, ewe, tnps, udma, refcrc, sramcrc, refbytes, srambytes, finalcrc, finalbytes


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    repo_root: Path = args.repo_root
    rtl_dir: Path = args.rtl_dir
    cwd = Path.cwd()
    gen = repo_root / "batch" / "gen_verilog_program.py"
    smoke = repo_root / "batch" / "run_verilog_smoke.py"
    program_dir = rtl_dir / "obj" / "verilog" / "programs"
    program_dir.mkdir(parents=True, exist_ok=True)
    args.cache_file = args.cache_file.resolve()
    args.profile_root = args.profile_root.resolve()

    bins = collect_bins(args.filter, rtl_dir, repo_root, cwd)
    if COMPILE_FAILURES:
        print(
            f"[run_verilog] ERROR: refusing to run with {len(COMPILE_FAILURES)} "
            "compile failure(s); stale .bin PASS is forbidden.",
            file=sys.stderr,
        )
        for failure in COMPILE_FAILURES:
            print(f"[run_verilog] compile_fail: {failure}", file=sys.stderr)
        return 1
    if args.limit > 0:
        bins = bins[: args.limit]
    if not bins:
        print("[run_verilog] ERROR: no .bin matched", file=sys.stderr)
        return 2

    print(f"[run_verilog] matched: {len(bins)}")
    print(f"[run_verilog] program_dir: {display(program_dir, repo_root)}")
    print(f"[run_verilog] cache: {display(args.cache_file, repo_root)}")
    print(f"[run_verilog] profile_root: {display(args.profile_root, repo_root)}")
    mode = run_mode(args)
    print(f"[run_verilog] mode: {mode}")
    options = normalized_options(args.option)
    ratio_label = "verilog-dpi/cx" if "dpi" in options else "verilog/cx"
    verilog_cyc_label = "verilog-dpi_cyc" if "dpi" in options else "verilog_cyc"
    profile_html = verilog_profile_path(args, bins, options)
    profile_corpus = infer_profile_corpus(bins, rtl_dir)
    profile_title = f"MDLA7 {profile_corpus} Verilog Profiles"
    if "dpi" in options:
        profile_title += " (DPI)"
    profile_rows: list[dict[str, object]] = []

    def first_systemc_report(stem: str) -> Path | None:
        for mode in ("cx", "fast"):
            for suffix in (".html", f".{mode}.html"):
                path = args.profile_root / mode / f"{stem}{suffix}"
                if path.exists():
                    return path
        for suffix in (".cx.html", ".html", ".fast.html"):
            path = args.profile_root / f"{stem}{suffix}"
            if path.exists():
                return path
        return None

    def remember_result(idx: int, bin_path: Path, info: BinInfo, ans: str,
                        cov: str = "", synth_ms: float | None = None,
                        synth_cycles: int | None = None,
                        verilog_cycles: int | None = None,
                        finalbytes: int = 0,
                        wall: float = 0.0,
                        reason: str = "",
                        perf_counters: dict[str, int] | None = None) -> None:
        hex_path = program_dir / f"{bin_path.stem}.verilog.hex"
        systemc_report = first_systemc_report(bin_path.stem)
        verilog_report = verilog_report_path(args, options, bin_path.stem)
        if not args.no_html:
            commands = decode_verilog_commands(hex_path, verilog_cycles)
            write_verilog_report_html(
                verilog_report,
                title=(
                    f"MDLA7 profile — {bin_path.stem}.tflite "
                    f"({'verilog-dpi' if 'dpi' in options else 'verilog'})"
                ),
                mode=mode,
                bin_path=bin_path,
                hex_path=hex_path,
                systemc_report=systemc_report,
                ans=ans,
                cov=cov,
                reason=reason,
                synth_ms=synth_ms,
                synth_cycles=synth_cycles,
                verilog_cycles=verilog_cycles,
                perf_counters=perf_counters,
                wall=wall,
                commands=commands,
            )
        profile_rows.append({
            "idx": idx,
            "program": bin_path.stem,
            "class": info.pattern_class,
            "timeout": fmt_timeout(args.timeout if args.timeout is not None else info.timeout_s),
            "ans": ans,
            "cov": cov,
            "cx_ms": fmt_ms(synth_ms),
            "cx_cyc": fmt_cycles(synth_cycles),
            "verilog_cyc": fmt_cycles(verilog_cycles),
            "verilog_ms": (
                float(verilog_cycles) / (VERILOG_CLOCK_HZ / 1000.0)
                if verilog_cycles is not None else None
            ),
            "verilog_cycles_raw": verilog_cycles,
            "perf_total": (perf_counters or {}).get("total"),
            "ratio": fmt_perf_ratio(bin_path, info, finalbytes,
                                    verilog_cycles, synth_cycles),
            "wall_s": f"{wall:.2f}" if wall else "",
            "reason": reason,
            "bin_link": str(bin_path.resolve()),
            "hex_link": str(hex_path.resolve()) if hex_path.exists() else "",
            "systemc_link": str(systemc_report.resolve()) if systemc_report else "",
            "verilog_link": str(verilog_report.resolve()) if verilog_report.exists() else "",
        })
    if options:
        print(f"[run_verilog] option: {','.join(sorted(options))}")
    if args.rerun_all:
        print("[run_verilog] cache_mode: rerun-all")
    print(
        f"{'idx':>3}  {fmt_program_name('program')} {'class':<6} {'tout':>6} {'ans':<6} {'cov':<7} "
        f"{'cx_ms':>10} {'cx_cyc':>12} {verilog_cyc_label:>20} "
        f"{ratio_label:>{RATIO_COL_WIDTH}} {'wall_s':>8}"
    )
    print("-" * (3 + 2 + PROGRAM_COL_WIDTH + 1 + 6 + 1 + 6 + 1 + 6 + 1 + 7 + 1 + 10 + 1 + 12 + 1 + 20 + 1 + RATIO_COL_WIDTH + 1 + 8))

    passed = 0
    failed = 0
    skipped = 0
    cache_hits = 0
    total_refcrc = 0
    total_sramcrc = 0
    total_refbytes = 0
    total_srambytes = 0
    total_finalcrc = 0
    total_finalbytes = 0
    synth_total = 0.0
    synth_cycle_total = 0
    verilog_cycle_total = 0
    comparable = 0
    build_done = args.no_build
    cache = load_cache(args.cache_file)
    cache_entries = cache.setdefault("entries", {})
    if not isinstance(cache_entries, dict):
        cache_entries = {}
        cache["entries"] = cache_entries
    runner_mtime_ns = Path(__file__).resolve().stat().st_mtime_ns
    generator_mtime_ns = gen.stat().st_mtime_ns if gen.exists() else 0
    source_mtime_ns = latest_tree_mtime_ns(rtl_dir / "verilog")
    for idx, bin_path in enumerate(bins, 1):
        rel_bin = display(bin_path, repo_root)
        info = classify_bin(bin_path)
        run_timeout = args.timeout if args.timeout is not None else info.timeout_s
        synth_cycles_for_target = load_synth_cycles(bin_path, args.profile_root)
        cached = None
        if not args.rerun_all:
            cached = cache_hit(
                cache_entries.get(rel_bin),
                bin_path,
                mode,
                runner_mtime_ns,
                generator_mtime_ns,
                source_mtime_ns,
            )
        if cached is not None:
            cache_hits += 1
            status = str(cached.get("status"))
            command_count = int(cached.get("cmds") or 0)
            conv_count = int(cached.get("conv") or 0)
            pool_count = int(cached.get("pool") or 0)
            requant_count = int(cached.get("requant") or 0)
            ewe_count = int(cached.get("ewe") or 0)
            tnps_count = int(cached.get("tnps") or 0)
            udma_count = int(cached.get("udma") or 0)
            refcrc_count = int(cached.get("refcrc") or 0)
            sramcrc_count = int(cached.get("sramcrc") or 0)
            refcrc_bytes = int(cached.get("refbytes") or 0)
            sramcrc_bytes = int(cached.get("srambytes") or 0)
            finalcrc_count = int(cached.get("finalcrc") or 0)
            finalcrc_bytes = int(cached.get("finalbytes") or 0)
            synth_ms = cached.get("synth_ms")
            synth_ms = synth_ms if isinstance(synth_ms, float) else None
            synth_cycles = cached.get("synth_cycles")
            synth_cycles = synth_cycles if isinstance(synth_cycles, int) else None
            verilog_cycles = cached.get("verilog_cycles")
            verilog_cycles = verilog_cycles if isinstance(verilog_cycles, int) else None
            perf_counters_obj = cached.get("perf_counters")
            perf_counters = (
                perf_counters_obj if isinstance(perf_counters_obj, dict) else {}
            )
            done = int(cached.get("done") or 0)
            total_refcrc += refcrc_count
            total_sramcrc += sramcrc_count
            total_refbytes += refcrc_bytes
            total_srambytes += sramcrc_bytes
            total_finalcrc += finalcrc_count
            total_finalbytes += finalcrc_bytes
            comparable_row = is_perf_comparable(bin_path, info, finalcrc_bytes)
            if (
                comparable_row and
                synth_ms is not None and synth_cycles is not None and verilog_cycles is not None
            ):
                synth_total += synth_ms
                synth_cycle_total += synth_cycles
                verilog_cycle_total += verilog_cycles
                comparable += 1
            if status == "SKIP":
                skipped += 1
                remember_result(idx, bin_path, info, "SKIP",
                                synth_ms=synth_ms, synth_cycles=synth_cycles,
                                verilog_cycles=verilog_cycles,
                                perf_counters=perf_counters,
                                reason="no byte-moving command (cached)")
                print(
                    f"{idx:>3}  {fmt_program_name(bin_path.stem)} {info.pattern_class:<6} "
                    f"{fmt_timeout(run_timeout):>6} {'SKIP':<6} {'':<7} "
                    f"{fmt_ms(synth_ms):>10} {fmt_cycles(synth_cycles):>12} "
                    f"{fmt_cycles(verilog_cycles):>20} "
                    f"{fmt_perf_ratio(bin_path, info, finalcrc_bytes, verilog_cycles, synth_cycles):>{RATIO_COL_WIDTH}} {0.0:>8.2f}"
                )
                print("     reason: no byte-moving command (cached)")
            else:
                passed += 1
                ans = "CACHED"
                cov = coverage_label(info, refcrc_count, sramcrc_count, finalcrc_bytes)
                remember_result(idx, bin_path, info, ans, cov, synth_ms, synth_cycles,
                                verilog_cycles, finalcrc_bytes,
                                perf_counters=perf_counters,
                                reason="cached")
                print(
                    f"{idx:>3}  {fmt_program_name(bin_path.stem)} {info.pattern_class:<6} "
                    f"{fmt_timeout(run_timeout):>6} {ans:<6} {cov:<7} "
                    f"{fmt_ms(synth_ms):>10} {fmt_cycles(synth_cycles):>12} "
                    f"{fmt_cycles(verilog_cycles):>20} "
                    f"{fmt_perf_ratio(bin_path, info, finalcrc_bytes, verilog_cycles, synth_cycles):>{RATIO_COL_WIDTH}} {0.0:>8.2f}"
                )
            continue
        hex_path = program_dir / f"{bin_path.stem}.verilog.hex"
        gen_cmd = [
            sys.executable, str(gen), str(bin_path), "-o", str(hex_path),
            "--max-commands", str(args.max_commands),
        ]
        if synth_cycles_for_target is not None and args.closed_loop_perf_target:
            gen_cmd.extend(["--closed-loop-target-cycles", str(synth_cycles_for_target)])
        if args.conv_sram_window_commands > 0:
            gen_cmd.extend(["--conv-sram-window-commands", str(args.conv_sram_window_commands)])
        if args.conv_sram_window_count > 0:
            gen_cmd.extend(["--conv-sram-window-count", str(args.conv_sram_window_count)])
        if args.conv_sample_count > 0:
            gen_cmd.extend(["--conv-sample-count", str(args.conv_sample_count)])
        if args.full_conv_coverage:
            gen_cmd.append("--full-conv-coverage")
        if is_bmm_bin(bin_path):
            gen_cmd.append("--full-final-ref")
        if args.check_materialized_layers or should_check_materialized_layers(bin_path):
            gen_cmd.append("--check-materialized-layers")
        if args.check_all_layers:
            gen_cmd.append("--check-all-layers")
        rc, gen_out, _ = run(gen_cmd, repo_root, timeout=run_timeout)
        if rc != 0:
            failed += 1
            remember_result(idx, bin_path, info, "FAIL", reason=f"generator failed rc={rc}")
            print(
                f"{idx:>3}  {fmt_program_name(bin_path.stem)} {info.pattern_class:<6} "
                f"{fmt_timeout(run_timeout):>6} {'FAIL':<6} {'':<7} "
                f"{'':>10} {'':>12} {'':>20} {'':>{RATIO_COL_WIDTH}} {0.0:>8.2f}"
            )
            print(f"     reason: generator failed rc={rc}: {gen_out.strip()}")
            continue
        stats_match = GEN_STATS_RE.search(gen_out)
        if stats_match:
            command_count = int(stats_match.group(1))
            conv_count = int(stats_match.group(2) or 0)
            pool_count = int(stats_match.group(3) or 0)
            requant_count = int(stats_match.group(4) or 0)
            ewe_count = int(stats_match.group(5) or 0)
            tnps_count = int(stats_match.group(6))
            udma_count = int(stats_match.group(7))
            refcrc_count = int(stats_match.group(8) or 0)
            sramcrc_count = int(stats_match.group(9) or 0)
            refcrc_bytes = int(stats_match.group(10) or 0)
            sramcrc_bytes = int(stats_match.group(11) or 0)
            finalcrc_count = int(stats_match.group(12) or 0)
            finalcrc_bytes = int(stats_match.group(13) or 0)
        else:
            (
                command_count, conv_count, pool_count, requant_count, ewe_count,
                tnps_count, udma_count, refcrc_count, sramcrc_count,
                refcrc_bytes, sramcrc_bytes, finalcrc_count, finalcrc_bytes,
            ) = count_commands(hex_path)
        if command_count == 0:
            skipped += 1
            synth_cycles = load_synth_cycles(bin_path, args.profile_root)
            synth_ms = synth_ms_from_cycles(synth_cycles)
            remember_result(idx, bin_path, info, "SKIP", synth_ms=synth_ms,
                            synth_cycles=synth_cycles, reason="no final command")
            print(
                f"{idx:>3}  {fmt_program_name(bin_path.stem)} {info.pattern_class:<6} "
                f"{fmt_timeout(run_timeout):>6} {'SKIP':<6} {'':<7} "
                f"{fmt_ms(synth_ms):>10} {fmt_cycles(synth_cycles):>12} "
                f"{'':>20} {'':>{RATIO_COL_WIDTH}} {0.0:>8.2f}"
            )
            print("     reason: no final command")
            cache_entries[rel_bin] = {
                "status": "SKIP",
                "bin_sig": file_signature(bin_path),
                "run_mode": mode,
                "runner_mtime_ns": runner_mtime_ns,
                "generator_mtime_ns": generator_mtime_ns,
                "source_mtime_ns": source_mtime_ns,
                "cmds": 0,
                "conv": 0,
                "pool": 0,
                "requant": 0,
                "ewe": 0,
                "tnps": 0,
                "udma": 0,
                "refcrc": 0,
                "sramcrc": 0,
                "refbytes": 0,
                "srambytes": 0,
                "finalcrc": 0,
                "finalbytes": 0,
                "synth_cycles": synth_cycles,
                "synth_ms": synth_ms,
                "verilog_cycles": None,
                "done": 0,
            }
            save_cache(args.cache_file, cache)
            continue
        total_refcrc += refcrc_count
        total_sramcrc += sramcrc_count
        total_refbytes += refcrc_bytes
        total_srambytes += sramcrc_bytes
        total_finalcrc += finalcrc_count
        total_finalbytes += finalcrc_bytes

        cmd = [str(smoke), "--test", "host", "--program", str(hex_path), "--ref-program", str(bin_path)]
        for opt in sorted(options):
            cmd.extend(["--option", opt])
        if build_done:
            cmd.append("--no-build")
        rc, out, wall = run(cmd, repo_root, timeout=run_timeout)
        build_done = True
        match = PASS_RE.search(out)
        verilog_ms = parse_verilog_ms(out)
        verilog_cycles = int(match.group(3)) if match and match.group(3) is not None else parse_verilog_cycles(out)
        perf_counters = parse_perf_counters(out)
        perf_total = perf_counters.get("total")
        if perf_total:
            verilog_cycles = perf_total
        synth_cycles = load_synth_cycles(bin_path, args.profile_root)
        synth_ms = synth_ms_from_cycles(synth_cycles)
        comparable_row = is_perf_comparable(bin_path, info, finalcrc_bytes)
        if (
            comparable_row and
            synth_ms is not None and synth_cycles is not None and verilog_cycles is not None
        ):
            synth_total += synth_ms
            synth_cycle_total += synth_cycles
            verilog_cycle_total += verilog_cycles
            comparable += 1
        if rc == 0 and match:
            issued = int(match.group(1))
            done = int(match.group(2))
            cov = coverage_label(info, refcrc_count, sramcrc_count, finalcrc_bytes)
            if requires_full_output_coverage(bin_path) and cov != "full":
                failed += 1
                reason = partial_coverage_reason(info, finalcrc_bytes)
                remember_result(idx, bin_path, info, "FAIL", cov, synth_ms, synth_cycles,
                                verilog_cycles, finalcrc_bytes, wall, reason,
                                perf_counters=perf_counters)
                print(
                    f"{idx:>3}  {fmt_program_name(bin_path.stem)} {info.pattern_class:<6} "
                    f"{fmt_timeout(run_timeout):>6} {'FAIL':<6} {cov:<7} "
                    f"{fmt_ms(synth_ms):>10} {fmt_cycles(synth_cycles):>12} "
                    f"{fmt_cycles(verilog_cycles):>20} "
                    f"{fmt_perf_ratio(bin_path, info, finalcrc_bytes, verilog_cycles, synth_cycles):>{RATIO_COL_WIDTH}} {wall:>8.2f}"
                )
                print(f"     reason: {reason}")
                continue
            passed += 1
            ans = "PASS"
            remember_result(idx, bin_path, info, ans, cov, synth_ms, synth_cycles,
                            verilog_cycles, finalcrc_bytes, wall,
                            perf_counters=perf_counters)
            print(
                f"{idx:>3}  {fmt_program_name(bin_path.stem)} {info.pattern_class:<6} "
                f"{fmt_timeout(run_timeout):>6} {ans:<6} {cov:<7} "
                f"{fmt_ms(synth_ms):>10} {fmt_cycles(synth_cycles):>12} "
                f"{fmt_cycles(verilog_cycles):>20} "
                f"{fmt_perf_ratio(bin_path, info, finalcrc_bytes, verilog_cycles, synth_cycles):>{RATIO_COL_WIDTH}} {wall:>8.2f}"
            )
            cache_entries[rel_bin] = {
                "status": "PASS",
                "bin_sig": file_signature(bin_path),
                "run_mode": mode,
                "runner_mtime_ns": runner_mtime_ns,
                "generator_mtime_ns": generator_mtime_ns,
                "source_mtime_ns": source_mtime_ns,
                "cmds": issued,
                "conv": conv_count,
                "pool": pool_count,
                "requant": requant_count,
                "ewe": ewe_count,
                "tnps": tnps_count,
                "udma": udma_count,
                "refcrc": refcrc_count,
                "sramcrc": sramcrc_count,
                "refbytes": refcrc_bytes,
                "srambytes": sramcrc_bytes,
                "finalcrc": finalcrc_count,
                "finalbytes": finalcrc_bytes,
                "synth_ms": synth_ms,
                "synth_cycles": synth_cycles,
                "verilog_ms": verilog_ms,
                "verilog_cycles": verilog_cycles,
                "perf_counters": perf_counters,
                "done": done,
                "wall_s": wall,
            }
            save_cache(args.cache_file, cache)
        else:
            failed += 1
            reason = "simulation failed"
            fail_match = FAIL_RE.search(out)
            if fail_match:
                reason = fail_match.group(1)
            elif rc == 124:
                reason = f"TIMEOUT after {run_timeout:.1f}s"
            remember_result(idx, bin_path, info, "FAIL",
                            coverage_label(info, refcrc_count,
                                           sramcrc_count, finalcrc_bytes),
                            synth_ms, synth_cycles, verilog_cycles,
                            finalcrc_bytes, wall, reason,
                            perf_counters=perf_counters)
            print(
                f"{idx:>3}  {fmt_program_name(bin_path.stem)} {info.pattern_class:<6} "
                f"{fmt_timeout(run_timeout):>6} {'FAIL':<6} {coverage_label(info, refcrc_count, sramcrc_count, finalcrc_bytes):<7} "
                f"{fmt_ms(synth_ms):>10} {fmt_cycles(synth_cycles):>12} "
                f"{fmt_cycles(verilog_cycles):>20} "
                f"{fmt_perf_ratio(bin_path, info, finalcrc_bytes, verilog_cycles, synth_cycles):>{RATIO_COL_WIDTH}} {wall:>8.2f}"
            )
            print(f"     reason: {reason}")
        if args.require_final_output_crc and command_count != 0 and finalcrc_count == 0:
            failed += 1
            print("     reason: no final-layer SRAM/L1 CRC coverage")
    print(
        f"[run_verilog] summary: pass={passed} fail={failed} "
        f"skip={skipped} total={len(bins)}"
    )
    print(
        f"[run_verilog] coverage: refcrc={total_refcrc} sramcrc={total_sramcrc} "
        f"finalcrc={total_finalcrc} refB={total_refbytes} sramB={total_srambytes} "
        f"finalB={total_finalbytes}"
    )
    print(
        "[run_verilog] compare: "
        f"comparable={comparable}/{passed + failed} "
        f"cx_total_ms={fmt_ms(synth_total if comparable else None)} "
        f"cx_total_cyc={fmt_cycles(synth_cycle_total if comparable else None)} "
        f"verilog_total_cyc={fmt_cycles(verilog_cycle_total if comparable else None)} "
        f"{ratio_label}={fmt_ratio(verilog_cycle_total if comparable else None, synth_cycle_total if comparable else None)}"
    )
    if args.require_crc_coverage and total_refcrc == 0 and total_sramcrc == 0:
        failed += 1
        print("[run_verilog] coverage_fail: required CRC coverage, but no CRC descriptors ran")
    if args.min_ref_bytes > 0 and total_refbytes < args.min_ref_bytes:
        failed += 1
        print(
            f"[run_verilog] coverage_fail: refB={total_refbytes} "
            f"below min_ref_bytes={args.min_ref_bytes}"
        )
    if args.min_sram_bytes > 0 and total_srambytes < args.min_sram_bytes:
        failed += 1
        print(
            f"[run_verilog] coverage_fail: sramB={total_srambytes} "
            f"below min_sram_bytes={args.min_sram_bytes}"
        )
    if args.min_final_bytes > 0 and total_finalbytes < args.min_final_bytes:
        failed += 1
        print(
            f"[run_verilog] coverage_fail: finalB={total_finalbytes} "
            f"below min_final_bytes={args.min_final_bytes}"
        )
    if cache_hits:
        print(f"[run_verilog] cache_hits: {cache_hits}")
    if not args.no_html:
        totals = {
            "pass": passed,
            "fail": failed,
            "skip": skipped,
            "refcrc": total_refcrc,
            "sramcrc": total_sramcrc,
            "finalcrc": total_finalcrc,
            "refB": total_refbytes,
            "sramB": total_srambytes,
            "finalB": total_finalbytes,
            "comparable": f"{comparable}/{passed + failed}",
            "cx_total_cyc": fmt_cycles(synth_cycle_total if comparable else None),
            "verilog_total_cyc": fmt_cycles(verilog_cycle_total if comparable else None),
            ratio_label: fmt_ratio(verilog_cycle_total if comparable else None,
                                   synth_cycle_total if comparable else None),
        }
        write_verilog_profile_html(profile_html, profile_title, profile_rows,
                                   mode=mode, cache_hits=cache_hits, totals=totals)
        print(f"[run_verilog] html: {profile_html}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
