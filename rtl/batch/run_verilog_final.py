#!/usr/bin/env python3
"""Run verilog_final host-driven byte-moving programs from MDL7 .bin files."""

from __future__ import annotations

import argparse
import glob
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
}
PASS_RE = re.compile(
    r"PASS: verilog_final host-driven .* issued=([0-9]+) done=([0-9]+)"
    r"(?:\s+vf_cycles=([0-9]+))?"
)
FAIL_RE = re.compile(r"(HOST_FINAL_FAIL:.*|FAIL: verilog_final host program.*)")
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
CACHE_VERSION = 13
WORDS_PER_COMMAND = 32
DEFAULT_MAX_COMMANDS = 4096
MDL7_MAGIC = 0x374C444D
PROGRAM_COL_WIDTH = 34
SYNTH_CLOCK_HZ = 1_900_000_000.0
VERILOG_FINAL_CLOCK_HZ = 100_000_000.0

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
    pattern_class: str
    timeout_s: float


def repo_paths() -> tuple[Path, Path]:
    script = Path(__file__).resolve()
    rtl_dir = script.parents[1]
    return rtl_dir.parent, rtl_dir


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
            candidates.extend((bin_root / alias).glob("*.bin"))
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
            if resolved.is_file() and resolved.suffix == ".bin" and not resolved.name.startswith("._"):
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
    return BinInfo(size_bytes, layers, tensor_bytes, ref_bytes, pattern_class, timeout_s)


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
    return int(round((ms / 1000.0) * VERILOG_FINAL_CLOCK_HZ))


def has_tensor_crc(refcrc: int, sramcrc: int, finalcrc: int) -> bool:
    return (refcrc + sramcrc + finalcrc) > 0


def coverage_label(info: BinInfo, refcrc: int, sramcrc: int, finalbytes: int) -> str:
    if (refcrc + sramcrc + finalbytes) == 0:
        return "sample"
    if info.ref_bytes > 0 and finalbytes >= info.ref_bytes:
        return "full"
    return "partial"


def profile_dir_for(program: Path) -> Path:
    return program.parent / "profile"


def profile_candidates(program: Path, profile_root: Path) -> list[Path]:
    stem = program.stem
    return [
        profile_dir_for(program) / f"{stem}.profile.json",
        profile_dir_for(program) / f"{stem}.synth.profile.json",
        profile_dir_for(program) / f"{stem}.mesh.profile.json",
        profile_root / f"{stem}.profile.json",
        profile_root / f"{stem}.synth.profile.json",
        profile_root / f"{stem}.mesh.profile.json",
        program.with_suffix(".profile.json"),
        program.with_suffix(".synth.profile.json"),
        program.with_suffix(".mesh.profile.json"),
    ]


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
    tmp = path.with_suffix(path.suffix + ".tmp")
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
                    help="bin filter, glob, path, or alias: slice, ethz, hotspot")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=None,
                    help="Manual per-program timeout. Default: auto by bin size/layer class.")
    ap.add_argument("--max-commands", type=int, default=DEFAULT_MAX_COMMANDS)
    ap.add_argument("--no-build", action="store_true")
    ap.add_argument("--crc-coverage", action="store_true",
                    help="Convenience mode for CRC coverage; enables --emit-conv-partial-psum.")
    ap.add_argument("--closed-loop-dataflow", action="store_true",
                    help="Generate real .bin probes for DRAM->UDMA->L1->engine->L1->UDMA->DRAM->L1CRC.")
    ap.add_argument("--closed-loop-perf-target", action="store_true",
                    help="Pad closed-loop vf_cycles toward synth profile total_cycles for calibration/debug only.")
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
    ap.add_argument("--emit-conv-partial-psum", action="store_true",
                    help="Pass through generator opt-in for INT8 CONV psum first/accumulate pairs.")
    ap.add_argument("--full-tensor", action="store_true",
                    help="Prefer full output tensor traversal when it fits in --max-commands.")
    ap.add_argument("--sample-descriptors", action="store_true",
                    help="Use legacy per-engine sample descriptors instead of synth-style microblock descriptors.")
    ap.add_argument("--conv-sram-window-commands", type=int, default=0,
                    help="Pass through SRAM-window command budget for oversized INT8 CONV layers.")
    ap.add_argument("--conv-sram-window-count", type=int, default=0,
                    help="Pass through max SRAM-window count for oversized INT8 CONV layers.")
    ap.add_argument("--rerun-all", action="store_true",
                    help="Ignore cached PASS/SKIP results and rerun every matched .bin.")
    ap.add_argument("--cache-file", type=Path,
                    default=rtl_dir / "obj" / "verilog_final" / "cache.json",
                    help="Regression cache JSON. Default: rtl/obj/verilog_final/cache.json")
    ap.add_argument("--profile-root", type=Path, default=repo_root / "batch" / "output",
                    help="Profile directory for synth_ms. Default: batch/output")
    ap.set_defaults(repo_root=repo_root, rtl_dir=rtl_dir)
    args = ap.parse_args(argv)
    if args.crc_coverage:
        args.emit_conv_partial_psum = True
        args.sample_descriptors = True
    if args.full_tensor:
        args.emit_conv_partial_psum = True
        args.sample_descriptors = True
    return args


def run_mode(args: argparse.Namespace) -> str:
    if args.closed_loop_dataflow:
        return "closed_loop_dataflow"
    if args.emit_conv_partial_psum:
        return "crc_coverage"
    if args.sample_descriptors:
        return "sample"
    return "microblock_control"


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
                if words[off + 3] & (1 << 12):
                    finalcrc += 1
                    finalbytes += words[off + 29]
        elif op == 6:
            udma += 1
            if words[off + 3] & (1 << 10):
                sramcrc += 1
                srambytes += words[off + 29]
                if words[off + 3] & (1 << 12):
                    finalcrc += 1
                    finalbytes += words[off + 29]
        elif op == 7:
            if words[off + 3] & (1 << 10):
                sramcrc += 1
                srambytes += words[off + 29]
                if words[off + 3] & (1 << 12):
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
                if words[off + 3] & (1 << 12):
                    finalcrc += 1
                    finalbytes += words[off + 29]
        elif op == 2:
            requant += 1
            if words[off + 3] & (1 << 10):
                sramcrc += 1
                srambytes += words[off + 29]
                if words[off + 3] & (1 << 12):
                    finalcrc += 1
                    finalbytes += words[off + 29]
        elif op == 3:
            ewe += 1
            if words[off + 3] & (1 << 10):
                sramcrc += 1
                srambytes += words[off + 29]
                if words[off + 3] & (1 << 12):
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
                if words[off + 3] & (1 << 12):
                    finalcrc += 1
                    finalbytes += words[off + 29]
    return count, conv, pool, requant, ewe, tnps, udma, refcrc, sramcrc, refbytes, srambytes, finalcrc, finalbytes


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    repo_root: Path = args.repo_root
    rtl_dir: Path = args.rtl_dir
    cwd = Path.cwd()
    gen = rtl_dir / "batch" / "gen_verilog_final_program.py"
    smoke = rtl_dir / "batch" / "run_verilog_final_smoke.py"
    program_dir = rtl_dir / "obj" / "verilog_final" / "programs"
    program_dir.mkdir(parents=True, exist_ok=True)
    args.cache_file = args.cache_file.resolve()
    args.profile_root = args.profile_root.resolve()

    bins = collect_bins(args.filter, rtl_dir, repo_root, cwd)
    if args.limit > 0:
        bins = bins[: args.limit]
    if not bins:
        print("[run_verilog_final] ERROR: no .bin matched", file=sys.stderr)
        return 2

    print(f"[run_verilog_final] matched: {len(bins)}")
    print(f"[run_verilog_final] program_dir: {display(program_dir, repo_root)}")
    print(f"[run_verilog_final] cache: {display(args.cache_file, repo_root)}")
    print(f"[run_verilog_final] profile_root: {display(args.profile_root, repo_root)}")
    mode = run_mode(args)
    print(f"[run_verilog_final] mode: {mode}")
    if args.rerun_all:
        print("[run_verilog_final] cache_mode: rerun-all")
    print(
        f"{'idx':>3}  {fmt_program_name('program')} {'class':<6} {'tout':>6} {'ans':<6} {'cov':<7} "
        f"{'synth_ms':>10} {'synth_cycles':>12} {'verilog_final_cycles':>20} "
        f"{'vf/synth':>9} {'wall_s':>8}"
    )
    print("-" * (3 + 2 + PROGRAM_COL_WIDTH + 1 + 6 + 1 + 6 + 1 + 6 + 1 + 7 + 1 + 10 + 1 + 12 + 1 + 20 + 1 + 9 + 1 + 8))

    passed = 0
    failed = 0
    skipped = 0
    sample_only = 0
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
    source_mtime_ns = latest_tree_mtime_ns(rtl_dir / "verilog_final")
    for idx, bin_path in enumerate(bins, 1):
        rel_bin = display(bin_path, repo_root)
        info = classify_bin(bin_path)
        run_timeout = args.timeout if args.timeout is not None else info.timeout_s
        cached = None
        if not args.rerun_all and not args.emit_conv_partial_psum:
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
            verilog_cycles = cached.get("verilog_final_cycles")
            verilog_cycles = verilog_cycles if isinstance(verilog_cycles, int) else None
            done = int(cached.get("done") or 0)
            total_refcrc += refcrc_count
            total_sramcrc += sramcrc_count
            total_refbytes += refcrc_bytes
            total_srambytes += sramcrc_bytes
            total_finalcrc += finalcrc_count
            total_finalbytes += finalcrc_bytes
            if synth_ms is not None and synth_cycles is not None and verilog_cycles is not None:
                synth_total += synth_ms
                synth_cycle_total += synth_cycles
                verilog_cycle_total += verilog_cycles
                comparable += 1
            if status == "SKIP":
                skipped += 1
                print(
                    f"{idx:>3}  {fmt_program_name(bin_path.stem)} {info.pattern_class:<6} "
                    f"{fmt_timeout(run_timeout):>6} {'SKIP':<6} {'':<7} "
                    f"{fmt_ms(synth_ms):>10} {fmt_cycles(synth_cycles):>12} "
                    f"{fmt_cycles(verilog_cycles):>20} "
                    f"{fmt_ratio(verilog_cycles, synth_cycles):>9} {0.0:>8.2f}"
                )
                print("     reason: no byte-moving command (cached)")
            else:
                passed += 1
                cached_sample_only = not has_tensor_crc(refcrc_count, sramcrc_count, finalcrc_count)
                if cached_sample_only:
                    sample_only += 1
                ans = "SAMPLE" if cached_sample_only else "CACHED"
                cov = coverage_label(info, refcrc_count, sramcrc_count, finalcrc_bytes)
                print(
                    f"{idx:>3}  {fmt_program_name(bin_path.stem)} {info.pattern_class:<6} "
                    f"{fmt_timeout(run_timeout):>6} {ans:<6} {cov:<7} "
                    f"{fmt_ms(synth_ms):>10} {fmt_cycles(synth_cycles):>12} "
                    f"{fmt_cycles(verilog_cycles):>20} "
                    f"{fmt_ratio(verilog_cycles, synth_cycles):>9} {0.0:>8.2f}"
                )
                if cached_sample_only:
                    mode = "sample-only" if args.sample_descriptors else "microblock-control"
                    print(f"     reason: {mode} cached result; no tensor CRC coverage")
            continue
        hex_path = program_dir / f"{bin_path.stem}.final.hex"
        synth_cycles_for_target = load_synth_cycles(bin_path, args.profile_root)
        gen_cmd = [
            sys.executable, str(gen), str(bin_path), "-o", str(hex_path),
            "--max-commands", str(args.max_commands),
        ]
        if not args.sample_descriptors and not args.emit_conv_partial_psum and not args.closed_loop_dataflow:
            gen_cmd.append("--microblock-descriptors")
        if args.emit_conv_partial_psum:
            gen_cmd.append("--emit-conv-partial-psum")
        if args.closed_loop_dataflow:
            gen_cmd.append("--closed-loop-dataflow")
            if synth_cycles_for_target is not None and args.closed_loop_perf_target:
                gen_cmd.extend(["--closed-loop-target-cycles", str(synth_cycles_for_target)])
        if args.full_tensor:
            gen_cmd.append("--full-tensor")
        if args.conv_sram_window_commands > 0:
            gen_cmd.extend(["--conv-sram-window-commands", str(args.conv_sram_window_commands)])
        if args.conv_sram_window_count > 0:
            gen_cmd.extend(["--conv-sram-window-count", str(args.conv_sram_window_count)])
        rc, gen_out, _ = run(gen_cmd, repo_root, timeout=run_timeout)
        if rc != 0:
            failed += 1
            print(
                f"{idx:>3}  {fmt_program_name(bin_path.stem)} {info.pattern_class:<6} "
                f"{fmt_timeout(run_timeout):>6} {'FAIL':<6} {'':<7} "
                f"{'':>10} {'':>12} {'':>20} {'':>9} {0.0:>8.2f}"
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
            print(
                f"{idx:>3}  {fmt_program_name(bin_path.stem)} {info.pattern_class:<6} "
                f"{fmt_timeout(run_timeout):>6} {'SKIP':<6} {'':<7} "
                f"{fmt_ms(synth_ms):>10} {fmt_cycles(synth_cycles):>12} "
                f"{'':>20} {'':>9} {0.0:>8.2f}"
            )
            print("     reason: no final command")
            if not args.emit_conv_partial_psum:
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
                    "verilog_final_cycles": None,
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
        if build_done:
            cmd.append("--no-build")
        rc, out, wall = run(cmd, repo_root, timeout=run_timeout)
        build_done = True
        match = PASS_RE.search(out)
        verilog_ms = parse_verilog_ms(out)
        verilog_cycles = int(match.group(3)) if match and match.group(3) is not None else parse_verilog_cycles(out)
        synth_cycles = load_synth_cycles(bin_path, args.profile_root)
        synth_ms = synth_ms_from_cycles(synth_cycles)
        if synth_ms is not None and synth_cycles is not None and verilog_cycles is not None:
            synth_total += synth_ms
            synth_cycle_total += synth_cycles
            verilog_cycle_total += verilog_cycles
            comparable += 1
        if rc == 0 and match:
            issued = int(match.group(1))
            done = int(match.group(2))
            passed += 1
            current_sample_only = not has_tensor_crc(refcrc_count, sramcrc_count, finalcrc_count)
            if current_sample_only:
                sample_only += 1
            ans = "SAMPLE" if current_sample_only else "PASS"
            cov = coverage_label(info, refcrc_count, sramcrc_count, finalcrc_bytes)
            print(
                f"{idx:>3}  {fmt_program_name(bin_path.stem)} {info.pattern_class:<6} "
                f"{fmt_timeout(run_timeout):>6} {ans:<6} {cov:<7} "
                f"{fmt_ms(synth_ms):>10} {fmt_cycles(synth_cycles):>12} "
                f"{fmt_cycles(verilog_cycles):>20} "
                f"{fmt_ratio(verilog_cycles, synth_cycles):>9} {wall:>8.2f}"
            )
            if current_sample_only:
                mode = "sample-only" if args.sample_descriptors else "microblock-control"
                print(f"     reason: {mode}; no tensor CRC coverage")
            if not args.emit_conv_partial_psum:
                cache_entries[rel_bin] = {
                    "status": "SAMPLE" if current_sample_only else "PASS",
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
                    "verilog_final_ms": verilog_ms,
                    "verilog_final_cycles": verilog_cycles,
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
            print(
                f"{idx:>3}  {fmt_program_name(bin_path.stem)} {info.pattern_class:<6} "
                f"{fmt_timeout(run_timeout):>6} {'FAIL':<6} {coverage_label(info, refcrc_count, sramcrc_count, finalcrc_bytes):<7} "
                f"{fmt_ms(synth_ms):>10} {fmt_cycles(synth_cycles):>12} "
                f"{fmt_cycles(verilog_cycles):>20} "
                f"{fmt_ratio(verilog_cycles, synth_cycles):>9} {wall:>8.2f}"
            )
            print(f"     reason: {reason}")
        if args.require_final_output_crc and command_count != 0 and finalcrc_count == 0:
            failed += 1
            print("     reason: no final-layer SRAM/L1 CRC coverage")
    print(
        f"[run_verilog_final] summary: pass={passed} fail={failed} "
        f"skip={skipped} sample_only={sample_only} total={len(bins)}"
    )
    print(
        f"[run_verilog_final] coverage: refcrc={total_refcrc} sramcrc={total_sramcrc} "
        f"finalcrc={total_finalcrc} refB={total_refbytes} sramB={total_srambytes} "
        f"finalB={total_finalbytes}"
    )
    print(
        "[run_verilog_final] compare: "
        f"comparable={comparable}/{passed + failed} "
        f"synth_total_ms={fmt_ms(synth_total if comparable else None)} "
        f"synth_total_cycles={fmt_cycles(synth_cycle_total if comparable else None)} "
        f"verilog_final_total_cycles={fmt_cycles(verilog_cycle_total if comparable else None)} "
        f"vf/synth={fmt_ratio(verilog_cycle_total if comparable else None, synth_cycle_total if comparable else None)}"
    )
    if total_refcrc == 0 and total_sramcrc == 0 and not args.emit_conv_partial_psum and not args.closed_loop_dataflow:
        mode_hint = "legacy sample-path" if args.sample_descriptors else "microblock-control"
        print(
            "[run_verilog_final] coverage_hint: CRC coverage is generated by "
            f"--crc-coverage; current mode is {mode_hint} without tensor CRC."
        )
    if args.require_crc_coverage and total_refcrc == 0 and total_sramcrc == 0:
        failed += 1
        print("[run_verilog_final] coverage_fail: required CRC coverage, but no CRC descriptors ran")
    if args.min_ref_bytes > 0 and total_refbytes < args.min_ref_bytes:
        failed += 1
        print(
            f"[run_verilog_final] coverage_fail: refB={total_refbytes} "
            f"below min_ref_bytes={args.min_ref_bytes}"
        )
    if args.min_sram_bytes > 0 and total_srambytes < args.min_sram_bytes:
        failed += 1
        print(
            f"[run_verilog_final] coverage_fail: sramB={total_srambytes} "
            f"below min_sram_bytes={args.min_sram_bytes}"
        )
    if args.min_final_bytes > 0 and total_finalbytes < args.min_final_bytes:
        failed += 1
        print(
            f"[run_verilog_final] coverage_fail: finalB={total_finalbytes} "
            f"below min_final_bytes={args.min_final_bytes}"
        )
    if cache_hits:
        print(f"[run_verilog_final] cache_hits: {cache_hits}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
