#!/usr/bin/env python3
"""Run the MDLA7 Verilog control-path testbench over compiled .bin programs."""

from __future__ import annotations

import argparse
import glob
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path


PASS_MARKER = "PASS: Testbench host/mdla7_top/dram control+datapath path"
FAIL_MARKER = "FAIL:"
HOST_LOAD_MARKER = "HOST: loaded"
SIM_FINISH_RE = re.compile(
    r"Verilator:\s+\$finish at\s+([0-9]+(?:\.[0-9]+)?)\s*([munpf]?s)",
    re.IGNORECASE,
)
VERILOG_CYCLES_RE = re.compile(r"VERILOG_CYCLES:\s*([0-9]+)")


TIME_UNIT_TO_MS = {
    "s": 1000.0,
    "ms": 1.0,
    "us": 0.001,
    "ns": 0.000001,
    "ps": 0.000000001,
    "fs": 0.000000000001,
}


def repo_paths() -> tuple[Path, Path]:
    script_path = Path(__file__).resolve()
    rtl_dir = script_path.parents[1]
    repo_root = rtl_dir.parent
    return repo_root, rtl_dir


def display(path: Path, base: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(path)


def normalize_arch(machine: str, system: str) -> str:
    machine = machine.lower()
    if machine in ("amd64", "x64"):
        return "x86_64"
    if machine == "aarch64" and system == "Darwin":
        return "arm64"
    return machine


def host_info() -> tuple[str, str, str]:
    system = platform.system() or sys.platform
    arch = normalize_arch(platform.machine() or "unknown", system)
    if system == "Darwin" and arch == "x86_64":
        label = "macOS Intel"
    elif system == "Darwin" and arch == "arm64":
        label = "macOS Apple Silicon"
    elif system == "Linux":
        label = "Linux"
    else:
        label = system or "unknown OS"
    return system, arch, label


def validate_host(system: str, arch: str) -> bool:
    if system == "Darwin" and arch not in ("x86_64", "arm64"):
        print(
            f"[run_mdla7_verilog] ERROR: unsupported macOS architecture: {arch}",
            file=sys.stderr,
        )
        return False
    return True


def simulator_arches(sim: Path) -> tuple[set[str], str]:
    try:
        proc = subprocess.run(
            ["file", str(sim)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return set(), ""

    desc = proc.stdout.strip()
    desc_lower = desc.lower()
    arches: set[str] = set()
    if "x86_64" in desc_lower or "x86-64" in desc_lower:
        arches.add("x86_64")
    if "arm64" in desc_lower or "aarch64" in desc_lower:
        arches.add("arm64")
    return arches, desc


def simulator_matches_host(sim: Path, system: str, arch: str) -> tuple[bool, str]:
    if not sim.exists():
        return False, "simulator is missing"

    arches, desc = simulator_arches(sim)
    if not arches:
        return True, "simulator architecture unknown; skipping arch check"

    expected = arch
    if system == "Linux" and arch == "aarch64":
        expected = "arm64"

    if expected in arches:
        return True, "sim_arch=" + ",".join(sorted(arches))

    return (
        False,
        f"simulator arch {','.join(sorted(arches))} does not match host arch {arch}: {desc}",
    )


def has_glob(pattern: str) -> bool:
    return any(ch in pattern for ch in "*?[]")


def collect_from_base(base: Path, pattern: str) -> list[Path]:
    if has_glob(pattern):
        return [Path(p) for p in glob.glob(str(base / pattern), recursive=True)]
    candidate = base / pattern
    return [candidate] if candidate.is_file() else []


def collect_bins(patterns: list[str], bin_root: Path, repo_root: Path, cwd: Path) -> list[Path]:
    found: dict[Path, Path] = {}

    for pattern in patterns:
        pattern = pattern.strip()
        if not pattern:
            continue

        candidates: list[Path] = []
        pattern_path = Path(pattern)
        has_path = pattern_path.is_absolute() or "/" in pattern or os.sep in pattern

        if pattern_path.is_absolute():
            if has_glob(pattern):
                candidates.extend(Path(p) for p in glob.glob(pattern, recursive=True))
            elif pattern_path.is_file():
                candidates.append(pattern_path)
        elif has_path:
            for base in (repo_root, bin_root, cwd):
                candidates.extend(collect_from_base(base, pattern))
        else:
            if has_glob(pattern):
                candidates.extend(bin_root.rglob(pattern))
            else:
                candidates.extend(bin_root.rglob(pattern))
                local = cwd / pattern
                if local.is_file():
                    candidates.append(local)

        for candidate in candidates:
            resolved = candidate.resolve()
            if (
                resolved.is_file()
                and resolved.suffix == ".bin"
                and not resolved.name.startswith("._")
            ):
                found[resolved] = resolved

    return sorted(found.values(), key=lambda p: str(p))


def latest_source_mtime(synth_dir: Path) -> float:
    mtimes = [
        p.stat().st_mtime
        for pattern in ("*.v", "*.sv", "*.cpp", "*.cc", "*.h", "*.hpp")
        for p in synth_dir.glob(pattern)
    ]
    filelist = synth_dir / "filelist_system_tb.f"
    if filelist.exists():
        mtimes.append(filelist.stat().st_mtime)
    return max(mtimes) if mtimes else 0.0


def find_cxx_stdlib_include() -> Path | None:
    env_path = os.environ.get("MDLA7_CXX_STDLIB_INCLUDE")
    if env_path:
        candidate = Path(env_path)
        if (candidate / "cstdint").exists():
            return candidate

    candidates = [
        Path("/Library/Developer/CommandLineTools/usr/include/c++/v1"),
        Path("/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/c++/v1"),
        Path("/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/include/c++/v1"),
        Path("/usr/local/opt/llvm/include/c++/v1"),
        Path("/opt/homebrew/opt/llvm/include/c++/v1"),
    ]
    candidates.extend(
        sorted(
            Path("/Library/Developer/CommandLineTools/SDKs").glob("MacOSX*.sdk/usr/include/c++/v1"),
            reverse=True,
        )
    )
    candidates.extend(sorted(Path("/usr/local/Cellar/llvm").glob("*/include/c++/v1"), reverse=True))
    candidates.extend(sorted(Path("/opt/homebrew/Cellar/llvm").glob("*/include/c++/v1"), reverse=True))

    for candidate in candidates:
        if (candidate / "cstdint").exists():
            return candidate
    return None


def legacy_sim_path(rtl_dir: Path) -> Path:
    return rtl_dir / "verilator" / "mdla7" / "obj_dir" / "VTestbench"


def build_sim(args: argparse.Namespace, rtl_dir: Path, synth_dir: Path) -> tuple[int, str]:
    args.obj_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(args.verilator),
        "--binary",
        "--sv",
        "--timing",
        "-Wall",
        "-Wno-fatal",
        "-CFLAGS",
        "-O3",
    ]
    cxx_stdlib_include = args.cxx_stdlib_include
    if cxx_stdlib_include is None and args.auto_cxx_stdlib_include:
        cxx_stdlib_include = find_cxx_stdlib_include()
    if cxx_stdlib_include is not None:
        cmd.extend(["-CFLAGS", f"-isystem {cxx_stdlib_include}"])
        if args.verbose_build:
            print(f"[run_mdla7_verilog] cxx_stdlib_include: {cxx_stdlib_include}")

    cmd.extend([
        f"-I{synth_dir}",
        "-f",
        str(args.filelist),
    ])
    dpi_cpp = synth_dir / "mdla7_dpi.cpp"
    if dpi_cpp.exists():
        cmd.append(str(dpi_cpp))
        if args.verbose_build:
            print(f"[run_mdla7_verilog] dpi_cpp: {dpi_cpp}")
    cmd.extend([
        "--top-module",
        args.top,
        "--Mdir",
        str(args.obj_dir),
    ])
    if args.verbose_build:
        print("[run_mdla7_verilog] build: " + " ".join(cmd))
    proc = subprocess.run(
        cmd,
        cwd=rtl_dir.parent,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return proc.returncode, proc.stdout or ""


def run_one(
    sim: Path,
    program: Path,
    repo_root: Path,
    timeout: float,
    timing_file: Path | None = None,
) -> tuple[int, str, float]:
    cmd = [str(sim), f"+PROGRAM={program}"]
    if timing_file is not None:
        cmd.append(f"+TIMING={timing_file}")
    start = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            cwd=repo_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
        elapsed = time.monotonic() - start
        return proc.returncode, proc.stdout, elapsed
    except subprocess.TimeoutExpired as exc:
        elapsed = time.monotonic() - start
        output = exc.stdout or ""
        if isinstance(output, bytes):
            output = output.decode(errors="replace")
        return 124, output + f"\nTIMEOUT after {timeout:.1f}s\n", elapsed


def parse_verilog_ms(output: str) -> float | None:
    cycle_matches = VERILOG_CYCLES_RE.findall(output)
    if cycle_matches:
        return int(cycle_matches[-1]) / 1.9e6
    matches = SIM_FINISH_RE.findall(output)
    if not matches:
        return None
    value_s, unit = matches[-1]
    scale = TIME_UNIT_TO_MS.get(unit.lower())
    if scale is None:
        return None
    return float(value_s) * scale


def load_synth_ms(program: Path, profile_root: Path) -> float | None:
    stem = program.stem
    candidates = [
        profile_root / f"{stem}.profile.json",
        profile_root / f"{stem}.synth.profile.json",
        profile_root / f"{stem}.mesh.profile.json",
    ]
    for path in candidates:
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
            return float(cycles) / 1.9e6
        except (TypeError, ValueError):
            continue
    return None


def profile_path_for(program: Path, profile_root: Path) -> Path | None:
    stem = program.stem
    for suffix in (".profile.json", ".synth.profile.json", ".mesh.profile.json"):
        path = profile_root / f"{stem}{suffix}"
        if path.exists() and not path.name.startswith("._"):
            return path
    return None


def write_timing_sidecar(program: Path, profile_root: Path, timing_root: Path) -> Path | None:
    profile_path = profile_path_for(program, profile_root)
    if profile_path is None:
        return None
    try:
        data = json.loads(profile_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    layers = data.get("layers") if isinstance(data, dict) else None
    if not isinstance(layers, list):
        return None

    max_id = -1
    cycles_by_layer: dict[int, int] = {}
    for layer in layers:
        if not isinstance(layer, dict):
            continue
        try:
            layer_id = int(layer.get("id"))
            cycles = int(layer.get("cycles_layer") or 0)
        except (TypeError, ValueError):
            continue
        if layer_id < 0:
            continue
        max_id = max(max_id, layer_id)
        cycles_by_layer[layer_id] = max(0, min(cycles, 0xFFFFFFFF))
    if max_id < 0:
        return None

    timing_root.mkdir(parents=True, exist_ok=True)
    out = timing_root / f"{program.stem}.timing.hex"
    with out.open("w") as f:
        for layer_id in range(max_id + 1):
            f.write(f"{cycles_by_layer.get(layer_id, 0):08x}\n")
    return out


def first_failure_line(output: str) -> str:
    for line in output.splitlines():
        if "HOST_FAIL:" in line or line.startswith("FAIL:") or "DATAPATH_FAIL:" in line:
            return line.strip()
    for line in output.splitlines():
        if "ERROR:" in line or "%Error" in line or "TIMEOUT" in line:
            return line.strip()
    return ""


def fmt_ms(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def fmt_ratio(num: float | None, den: float | None) -> str:
    if num is None or den is None or den == 0.0:
        return "n/a"
    return f"{num / den:.3f}x"


def parse_args(argv: list[str]) -> argparse.Namespace:
    repo_root, rtl_dir = repo_paths()
    default_verilator = rtl_dir / "verilator" / "bin" / "verilator"
    if not default_verilator.exists():
        default_verilator = Path("verilator")

    parser = argparse.ArgumentParser(
        description="Run rtl/obj_dir/VTestbench over rtl/bin/*.bin programs."
    )
    parser.add_argument(
        "--filter",
        nargs="+",
        default=["*.bin"],
        help="Bin glob(s). Plain patterns search recursively under rtl/bin. Default: *.bin",
    )
    parser.add_argument(
        "--bin-root",
        type=Path,
        default=rtl_dir / "bin",
        help="Root directory for compiled .bin programs. Default: rtl/bin",
    )
    parser.add_argument(
        "--sim",
        type=Path,
        default=None,
        help="Verilator simulation executable. Default: <obj-dir>/VTestbench",
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="Force-build VTestbench with Verilator before running.",
    )
    parser.add_argument(
        "--verbose-build",
        action="store_true",
        help="Print full Verilator build command/output.",
    )
    parser.add_argument(
        "--no-build",
        action="store_true",
        help="Do not auto-build when the default simulator is missing.",
    )
    parser.add_argument(
        "--verilator",
        type=Path,
        default=default_verilator,
        help="Verilator command used with --build.",
    )
    parser.add_argument(
        "--obj-dir",
        type=Path,
        default=rtl_dir / "obj_dir",
        help="Verilator object directory used with --build. Default: rtl/obj_dir",
    )
    parser.add_argument(
        "--filelist",
        type=Path,
        default=rtl_dir / "synth" / "filelist_system_tb.f",
        help="Verilog filelist used with --build.",
    )
    parser.add_argument("--top", default="Testbench", help="Top module used with --build.")
    parser.add_argument(
        "--cxx-stdlib-include",
        type=Path,
        default=None,
        help="C++ standard library include directory to pass through Verilator -CFLAGS.",
    )
    parser.add_argument(
        "--no-auto-cxx-stdlib-include",
        dest="auto_cxx_stdlib_include",
        action="store_false",
        default=True,
        help="Disable automatic macOS libc++ include discovery for Verilator builds.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help=(
            "Timeout per .bin in seconds. Default: 30, or 300 with "
            "--compare-synth-verilog."
        ),
    )
    parser.add_argument("--limit", type=int, default=0, help="Run only the first N matched .bin files.")
    parser.add_argument("--list", action="store_true", help="List matched .bin files and exit.")
    parser.add_argument(
        "--stop-on-fail",
        action="store_true",
        help="Stop after the first failing simulation.",
    )
    parser.add_argument(
        "--require-host-load",
        action="store_true",
        help="Fail a run if simulator output does not include the host .bin load message.",
    )
    parser.add_argument(
        "--no-arch-check",
        action="store_true",
        help="Do not check the simulator binary architecture before running.",
    )
    parser.add_argument(
        "--compare-synth-verilog",
        action="store_true",
        help=(
            "Print a quiet synth-vs-Verilog comparison table. Synth ms is read "
            "from batch/output/<bin-stem>.profile.json when available; Verilog "
            "ms is parsed from the Verilator simulation report."
        ),
    )
    parser.add_argument(
        "--profile-root",
        type=Path,
        default=repo_root / "batch" / "output",
        help="Profile directory used by --compare-synth-verilog. Default: batch/output",
    )
    parser.add_argument(
        "--no-profile-timing",
        action="store_true",
        help="Do not pass profile-guided per-layer timing sidecars to Verilog.",
    )
    parser.add_argument(
        "--show-verilog-output",
        action="store_true",
        help="Print raw Verilator simulation output for each run.",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    repo_root, rtl_dir = repo_paths()
    synth_dir = rtl_dir / "synth"
    args = parse_args(argv)
    if args.timeout is None:
        args.timeout = 300.0 if args.compare_synth_verilog else 30.0
    system, arch, host_label = host_info()
    if not validate_host(system, arch):
        return 2

    sim_was_default = args.sim is None
    args.bin_root = args.bin_root.resolve()
    args.obj_dir = args.obj_dir.resolve()
    args.profile_root = args.profile_root.resolve()
    if args.sim is None:
        args.sim = args.obj_dir / "VTestbench"
    else:
        args.sim = args.sim.resolve()
    args.filelist = args.filelist.resolve()
    if args.cxx_stdlib_include is not None:
        args.cxx_stdlib_include = args.cxx_stdlib_include.resolve()
    if args.verilator != Path("verilator"):
        args.verilator = args.verilator.resolve()

    bins = collect_bins(args.filter, args.bin_root, repo_root, Path.cwd())
    if args.limit > 0:
        bins = bins[: args.limit]

    print(f"[run_mdla7_verilog] host: {host_label} ({system} {arch})")
    print(f"[run_mdla7_verilog] bin_root: {display(args.bin_root, repo_root)}")
    print(f"[run_mdla7_verilog] obj_dir: {display(args.obj_dir, repo_root)}")
    print(f"[run_mdla7_verilog] simulator: {display(args.sim, repo_root)}")
    if args.compare_synth_verilog:
        print(f"[run_mdla7_verilog] profile_root: {display(args.profile_root, repo_root)}")
    print(f"[run_mdla7_verilog] matched: {len(bins)}")

    if not bins:
        print("[run_mdla7_verilog] ERROR: no .bin files matched --filter", file=sys.stderr)
        return 2

    if args.list:
        for program in bins:
            print(display(program, repo_root))
        return 0

    arch_ok = True
    arch_detail = ""
    if args.sim.exists() and not args.no_arch_check:
        arch_ok, arch_detail = simulator_matches_host(args.sim, system, arch)

    should_build = args.build or (
        sim_was_default
        and not args.no_build
        and (not args.sim.exists() or not arch_ok)
    )
    if should_build:
        if args.build:
            print("[run_mdla7_verilog] build_reason: forced by --build")
        elif not args.sim.exists():
            print("[run_mdla7_verilog] build_reason: simulator missing")
        elif not arch_ok:
            print(f"[run_mdla7_verilog] build_reason: {arch_detail}")
        build_rc, build_output = build_sim(args, rtl_dir, synth_dir)
        if build_rc != 0:
            if build_output:
                print(build_output, end="" if build_output.endswith("\n") else "\n")
            print(f"[run_mdla7_verilog] ERROR: build failed rc={build_rc}", file=sys.stderr)
            return build_rc
        if args.verbose_build and build_output:
            print(build_output, end="" if build_output.endswith("\n") else "\n")

    if not args.sim.exists():
        legacy_sim = legacy_sim_path(rtl_dir)
        if legacy_sim.exists() and args.sim == args.obj_dir / "VTestbench":
            print(
                "[run_mdla7_verilog] NOTE: found an older simulator at "
                f"{display(legacy_sim, repo_root)}",
                file=sys.stderr,
            )
            print(
                "[run_mdla7_verilog] To move generated output out of rtl/verilator, run:\n"
                f"  mkdir -p {display(args.obj_dir, repo_root)} && "
                f"cp {display(legacy_sim, repo_root)} {display(args.sim, repo_root)}",
                file=sys.stderr,
            )
        print(
            f"[run_mdla7_verilog] ERROR: simulator not found: {args.sim}\n"
            "[run_mdla7_verilog] Hint: run with --build after the C++ toolchain is fixed, "
            "or pass --sim <path>.",
            file=sys.stderr,
        )
        return 2

    if not args.no_arch_check:
        arch_ok, arch_detail = simulator_matches_host(args.sim, system, arch)
        if not arch_ok:
            print(f"[run_mdla7_verilog] ERROR: {arch_detail}", file=sys.stderr)
            print(
                "[run_mdla7_verilog] Hint: run with --build to rebuild for this host, "
                "or use --no-arch-check if you intentionally want this simulator.",
                file=sys.stderr,
            )
            return 2
        print(f"[run_mdla7_verilog] {arch_detail}")

    latest_src = latest_source_mtime(synth_dir)
    if args.sim.stat().st_mtime < latest_src:
        print(
            "[run_mdla7_verilog] WARN: simulator is older than rtl/synth sources; "
            "use --build when the Verilator C++ toolchain is available."
        )

    failures: list[str] = []
    missing_host_load = 0
    rows: list[dict[str, object]] = []
    timing_root = args.obj_dir / "timing"

    if args.compare_synth_verilog:
        print(
            f"{'idx':>3}  {'program':<42} {'ans':<6} "
            f"{'synth_ms':>10} {'verilog_ms':>11} {'v/synth':>9} {'wall_s':>8}"
        )
        print("-" * 98)

    for idx, program in enumerate(bins, start=1):
        rel_program = display(program, repo_root)
        if not args.compare_synth_verilog:
            print(f"[run_mdla7_verilog] RUN {idx}/{len(bins)} {rel_program}")
        timing_file = None
        if args.compare_synth_verilog and not args.no_profile_timing:
            timing_file = write_timing_sidecar(program, args.profile_root, timing_root)
        rc, output, elapsed = run_one(
            args.sim,
            program,
            repo_root,
            args.timeout,
            timing_file=timing_file,
        )
        if output and args.show_verilog_output:
            print(output, end="" if output.endswith("\n") else "\n")

        has_pass = PASS_MARKER in output
        has_fail = FAIL_MARKER in output
        has_host_load = HOST_LOAD_MARKER in output
        passed = (rc == 0) and has_pass and not has_fail
        if args.require_host_load and not has_host_load:
            passed = False

        host_note = "host_load=ok" if has_host_load else "host_load=missing"
        if not has_host_load:
            missing_host_load += 1

        verilog_ms = parse_verilog_ms(output)
        synth_ms = load_synth_ms(program, args.profile_root) if args.compare_synth_verilog else None
        reason = "" if passed else first_failure_line(output)
        rows.append({
            "program": rel_program,
            "passed": passed,
            "synth_ms": synth_ms,
            "verilog_ms": verilog_ms,
            "wall_s": elapsed,
            "reason": reason,
        })

        if passed:
            if args.compare_synth_verilog:
                print(
                    f"{idx:>3}  {program.stem:<42} {'PASS':<6} "
                    f"{fmt_ms(synth_ms):>10} {fmt_ms(verilog_ms):>11} "
                    f"{fmt_ratio(verilog_ms, synth_ms):>9} {elapsed:>8.2f}"
                )
            else:
                print(
                    f"[run_mdla7_verilog] PASS {rel_program} "
                    f"(verilog_ms={fmt_ms(verilog_ms)}, wall={elapsed:.2f}s, {host_note})"
                )
        else:
            if args.compare_synth_verilog:
                print(
                    f"{idx:>3}  {program.stem:<42} {'FAIL':<6} "
                    f"{fmt_ms(synth_ms):>10} {fmt_ms(verilog_ms):>11} "
                    f"{fmt_ratio(verilog_ms, synth_ms):>9} {elapsed:>8.2f}"
                )
                if reason:
                    print(f"     reason: {reason}")
            else:
                print(
                    f"[run_mdla7_verilog] FAIL {rel_program} "
                    f"(rc={rc}, verilog_ms={fmt_ms(verilog_ms)}, wall={elapsed:.2f}s, {host_note})"
                )
                if reason:
                    print(f"[run_mdla7_verilog] reason: {reason}")
            failures.append(rel_program)
            if args.stop_on_fail:
                break

    run_count = len(rows)
    passed_count = run_count - len(failures)
    print(f"[run_mdla7_verilog] summary: pass={passed_count} fail={len(failures)} total={run_count}")
    if args.compare_synth_verilog and rows:
        synth_total = sum(
            row["synth_ms"] for row in rows
            if isinstance(row.get("synth_ms"), float)
        )
        verilog_total = sum(
            row["verilog_ms"] for row in rows
            if isinstance(row.get("verilog_ms"), float)
        )
        comparable = sum(
            1 for row in rows
            if isinstance(row.get("synth_ms"), float)
            and isinstance(row.get("verilog_ms"), float)
        )
        ratio = (verilog_total / synth_total) if synth_total else None
        print(
            "[run_mdla7_verilog] compare: "
            f"comparable={comparable}/{run_count} "
            f"synth_total_ms={fmt_ms(synth_total if synth_total else None)} "
            f"verilog_total_ms={fmt_ms(verilog_total if verilog_total else None)} "
            f"verilog/synth={fmt_ratio(verilog_total if verilog_total else None, synth_total if synth_total else None)}"
        )
    if missing_host_load:
        print(
            "[run_mdla7_verilog] WARN: some runs did not print HOST: loaded; "
            "that usually means VTestbench was built before host.v loaded .bin files."
        )

    if failures:
        print("[run_mdla7_verilog] failed programs:")
        for failure in failures:
            print(f"  {failure}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
