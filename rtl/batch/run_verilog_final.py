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
PASS_RE = re.compile(r"PASS: verilog_final host-driven .* issued=([0-9]+) done=([0-9]+)")
FAIL_RE = re.compile(r"(HOST_FINAL_FAIL:.*|FAIL: verilog_final host program.*)")
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
)
CACHE_VERSION = 5
WORDS_PER_COMMAND = 32
DEFAULT_MAX_COMMANDS = 4096


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
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--max-commands", type=int, default=DEFAULT_MAX_COMMANDS)
    ap.add_argument("--no-build", action="store_true")
    ap.add_argument("--crc-coverage", action="store_true",
                    help="Convenience mode for CRC coverage; enables --emit-conv-partial-psum.")
    ap.add_argument("--require-crc-coverage", action="store_true",
                    help="Fail if the run produces no refcrc/sramcrc coverage.")
    ap.add_argument("--min-ref-bytes", type=int, default=0,
                    help="Fail if total refB coverage is below this value.")
    ap.add_argument("--min-sram-bytes", type=int, default=0,
                    help="Fail if total sramB coverage is below this value.")
    ap.add_argument("--emit-conv-partial-psum", action="store_true",
                    help="Pass through generator opt-in for INT8 CONV psum first/accumulate pairs.")
    ap.add_argument("--conv-sram-window-commands", type=int, default=0,
                    help="Pass through SRAM-window command budget for oversized INT8 CONV layers.")
    ap.add_argument("--conv-sram-window-count", type=int, default=0,
                    help="Pass through max SRAM-window count for oversized INT8 CONV layers.")
    ap.add_argument("--rerun-all", action="store_true",
                    help="Ignore cached PASS/SKIP results and rerun every matched .bin.")
    ap.add_argument("--cache-file", type=Path,
                    default=rtl_dir / "obj" / "verilog_final" / "cache.json",
                    help="Regression cache JSON. Default: rtl/obj/verilog_final/cache.json")
    ap.set_defaults(repo_root=repo_root, rtl_dir=rtl_dir)
    args = ap.parse_args(argv)
    if args.crc_coverage:
        args.emit_conv_partial_psum = True
    return args


def count_commands(hex_path: Path) -> tuple[int, int, int, int, int, int, int, int, int, int, int]:
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
        elif op == 6:
            udma += 1
        elif op == 1:
            conv += 1
            if words[off + 3] & (1 << 9):
                refcrc += 1
                refbytes += words[off + 29]
            if words[off + 3] & (1 << 10):
                sramcrc += 1
                srambytes += words[off + 29]
        elif op == 2:
            requant += 1
            if words[off + 3] & (1 << 10):
                sramcrc += 1
                srambytes += words[off + 29]
        elif op == 3:
            ewe += 1
            if words[off + 3] & (1 << 10):
                sramcrc += 1
                srambytes += words[off + 29]
        elif op == 4:
            pool += 1
            if words[off + 3] & (1 << 9):
                refcrc += 1
                refbytes += words[off + 29]
            if words[off + 3] & (1 << 10):
                sramcrc += 1
                srambytes += words[off + 29]
    return count, conv, pool, requant, ewe, tnps, udma, refcrc, sramcrc, refbytes, srambytes


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

    bins = collect_bins(args.filter, rtl_dir, repo_root, cwd)
    if args.limit > 0:
        bins = bins[: args.limit]
    if not bins:
        print("[run_verilog_final] ERROR: no .bin matched", file=sys.stderr)
        return 2

    print(f"[run_verilog_final] matched: {len(bins)}")
    print(f"[run_verilog_final] program_dir: {display(program_dir, repo_root)}")
    print(f"[run_verilog_final] cache: {display(args.cache_file, repo_root)}")
    if args.rerun_all:
        print("[run_verilog_final] cache_mode: rerun-all")
    print("idx  program                                  ans   cmds  conv  pool requant  ewe  tnps  udma refcrc sramcrc     refB    sramB  done  wall_s")
    print("-------------------------------------------------------------------------------------------------------------------------------------")

    passed = 0
    failed = 0
    skipped = 0
    cache_hits = 0
    total_refcrc = 0
    total_sramcrc = 0
    total_refbytes = 0
    total_srambytes = 0
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
        cached = None
        if not args.rerun_all and not args.emit_conv_partial_psum:
            cached = cache_hit(
                cache_entries.get(rel_bin),
                bin_path,
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
            done = int(cached.get("done") or 0)
            total_refcrc += refcrc_count
            total_sramcrc += sramcrc_count
            total_refbytes += refcrc_bytes
            total_srambytes += sramcrc_bytes
            if status == "SKIP":
                skipped += 1
                print(f"{idx:3d}  {bin_path.stem[:38]:38s} SKIP {command_count:5d} {conv_count:5d} {pool_count:5d} {requant_count:7d} {ewe_count:4d} {tnps_count:5d} {udma_count:5d} {refcrc_count:6d} {sramcrc_count:7d} {refcrc_bytes:8d} {sramcrc_bytes:8d} {done:5d} {0.0:7.2f}")
                print("     reason: no byte-moving command (cached)")
            else:
                passed += 1
                print(f"{idx:3d}  {bin_path.stem[:38]:38s} CACHE{command_count:5d} {conv_count:5d} {pool_count:5d} {requant_count:7d} {ewe_count:4d} {tnps_count:5d} {udma_count:5d} {refcrc_count:6d} {sramcrc_count:7d} {refcrc_bytes:8d} {sramcrc_bytes:8d} {done:5d} {0.0:7.2f}")
            continue
        hex_path = program_dir / f"{bin_path.stem}.final.hex"
        gen_cmd = [
            sys.executable, str(gen), str(bin_path), "-o", str(hex_path),
            "--max-commands", str(args.max_commands),
        ]
        if args.emit_conv_partial_psum:
            gen_cmd.append("--emit-conv-partial-psum")
        if args.conv_sram_window_commands > 0:
            gen_cmd.extend(["--conv-sram-window-commands", str(args.conv_sram_window_commands)])
        if args.conv_sram_window_count > 0:
            gen_cmd.extend(["--conv-sram-window-count", str(args.conv_sram_window_count)])
        rc, gen_out, _ = run(gen_cmd, repo_root, timeout=args.timeout)
        if rc != 0:
            failed += 1
            print(f"{idx:3d}  {bin_path.stem[:38]:38s} FAIL   n/a   n/a    0.00")
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
        else:
            (
                command_count, conv_count, pool_count, requant_count, ewe_count,
                tnps_count, udma_count, refcrc_count, sramcrc_count,
                refcrc_bytes, sramcrc_bytes,
            ) = count_commands(hex_path)
        if command_count == 0:
            skipped += 1
            print(f"{idx:3d}  {bin_path.stem[:38]:38s} SKIP     0     0     0       0    0     0     0      0       0        0        0     0    0.00")
            print("     reason: no final command")
            if not args.emit_conv_partial_psum:
                cache_entries[rel_bin] = {
                    "status": "SKIP",
                    "bin_sig": file_signature(bin_path),
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
                    "done": 0,
                }
                save_cache(args.cache_file, cache)
            continue
        total_refcrc += refcrc_count
        total_sramcrc += sramcrc_count
        total_refbytes += refcrc_bytes
        total_srambytes += sramcrc_bytes

        cmd = [str(smoke), "--test", "host", "--program", str(hex_path), "--ref-program", str(bin_path)]
        if build_done:
            cmd.append("--no-build")
        rc, out, wall = run(cmd, repo_root, timeout=args.timeout)
        build_done = True
        match = PASS_RE.search(out)
        if rc == 0 and match:
            issued = int(match.group(1))
            done = int(match.group(2))
            passed += 1
            print(
                f"{idx:3d}  {bin_path.stem[:38]:38s} PASS "
                f"{issued:5d} {conv_count:5d} {pool_count:5d} {requant_count:7d} {ewe_count:4d} {tnps_count:5d} {udma_count:5d} {refcrc_count:6d} {sramcrc_count:7d} {refcrc_bytes:8d} {sramcrc_bytes:8d} {done:5d} {wall:7.2f}"
            )
            if not args.emit_conv_partial_psum:
                cache_entries[rel_bin] = {
                    "status": "PASS",
                    "bin_sig": file_signature(bin_path),
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
                reason = f"TIMEOUT after {args.timeout:.1f}s"
            print(
                f"{idx:3d}  {bin_path.stem[:38]:38s} FAIL "
                f"{command_count:5d} {conv_count:5d} {pool_count:5d} {requant_count:7d} {ewe_count:4d} {tnps_count:5d} {udma_count:5d} {refcrc_count:6d} {sramcrc_count:7d} {refcrc_bytes:8d} {sramcrc_bytes:8d}   n/a  {wall:7.2f}"
            )
            print(f"     reason: {reason}")

    print(f"[run_verilog_final] summary: pass={passed} fail={failed} skip={skipped} total={len(bins)}")
    print(
        f"[run_verilog_final] coverage: refcrc={total_refcrc} sramcrc={total_sramcrc} "
        f"refB={total_refbytes} sramB={total_srambytes}"
    )
    if total_refcrc == 0 and total_sramcrc == 0 and not args.emit_conv_partial_psum:
        print(
            "[run_verilog_final] coverage_hint: CRC coverage is generated by "
            "--crc-coverage; default mode is sample-path only."
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
    if cache_hits:
        print(f"[run_verilog_final] cache_hits: {cache_hits}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
