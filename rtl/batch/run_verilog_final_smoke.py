#!/usr/bin/env python3
"""Build and run the verilog_final block-level smoke tests."""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


TESTS = (
    ("conv", "Testbench_conv_datapath"),
    ("requant", "Testbench_requant_datapath"),
    ("pool", "Testbench_pool_datapath"),
    ("ewe", "Testbench_ewe_datapath"),
    ("tnps", "Testbench_tnps_datapath"),
    ("route", "Testbench_route_timing"),
    ("contention", "Testbench_l1mesh_contention"),
    ("top", "Testbench_top_byte_movers"),
    ("host", "Testbench_host_program"),
)


def repo_paths() -> tuple[Path, Path]:
    script = Path(__file__).resolve()
    rtl_dir = script.parents[1]
    return rtl_dir.parent, rtl_dir


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
    for candidate in candidates:
        if (candidate / "cstdint").exists():
            return candidate
    return None


def run(cmd: list[str], cwd: Path, quiet: bool) -> tuple[int, str]:
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if proc.stdout and not quiet:
        print(proc.stdout, end="" if proc.stdout.endswith("\n") else "\n")
    return proc.returncode, proc.stdout or ""


def find_default_verilator(rtl_dir: Path) -> Path:
    bundled = rtl_dir / "verilator" / "bin" / "verilator"
    bundled_bin = rtl_dir / "verilator" / "bin" / "verilator_bin"
    if bundled.exists() and bundled_bin.exists():
        return bundled
    for name in ("/opt/homebrew/bin/verilator", "/usr/local/bin/verilator", "verilator"):
        found = shutil.which(name) if name == "verilator" else name
        if found and Path(found).exists():
            return Path(found)
    return bundled if bundled.exists() else Path("verilator")


def parse_args(argv: list[str]) -> argparse.Namespace:
    repo_root, rtl_dir = repo_paths()
    default_verilator = find_default_verilator(rtl_dir)

    ap = argparse.ArgumentParser()
    ap.add_argument("--verilator", type=Path, default=default_verilator)
    ap.add_argument("--test", choices=[name for name, _ in TESTS], action="append")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--no-build", action="store_true")
    ap.add_argument("--cxx-stdlib-include", type=Path, default=None)
    ap.add_argument("--program", type=Path, default=None,
                    help="hex descriptor stream passed to the host smoke test as +FINAL_PROGRAM")
    ap.add_argument("--ref-program", type=Path, default=None,
                    help="original MDL7 .bin passed to the host smoke test as +FINAL_REF_PROGRAM")
    ap.set_defaults(repo_root=repo_root, rtl_dir=rtl_dir)
    return ap.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    repo_root: Path = args.repo_root
    rtl_dir: Path = args.rtl_dir
    final_dir = rtl_dir / "verilog_final"
    filelist = final_dir / "filelist_system_tb.f"
    verilator = args.verilator.resolve() if args.verilator != Path("verilator") else args.verilator
    selected = set(args.test or [name for name, _ in TESTS])
    cxx_include = args.cxx_stdlib_include or find_cxx_stdlib_include()
    obj_root = rtl_dir / "obj" / "verilog_final"
    obj_root.mkdir(parents=True, exist_ok=True)

    print(f"[verilog_final_smoke] host: {platform.system()} {platform.machine()}")
    print(f"[verilog_final_smoke] tests: {','.join(name for name, _ in TESTS if name in selected)}")

    failures: list[str] = []
    for name, top in TESTS:
        if name not in selected:
            continue
        obj_dir = obj_root / name
        exe = obj_dir / f"V{top}"
        if not args.no_build:
            cmd = [
                str(verilator),
                "--binary",
                "--timing",
                "-Wall",
                "-Wno-fatal",
            ]
            if cxx_include is not None:
                cmd.extend(["-CFLAGS", f"-isystem {cxx_include}"])
            cmd.extend([
                "-Irtl/synth",
                "-Irtl/verilog_final",
                "-f",
                str(filelist),
                str(final_dir / f"{top}.v"),
                "--top-module",
                top,
                "--Mdir",
                str(obj_dir),
            ])
            rc, output = run(cmd, repo_root, quiet=not args.verbose)
            if rc != 0:
                print(output, end="" if output.endswith("\n") else "\n")
                failures.append(f"{name}: build failed")
                continue
        sim_cmd = [str(exe)]
        if name == "host" and args.program is not None:
            sim_cmd.append(f"+FINAL_PROGRAM={args.program}")
        if name == "host" and args.ref_program is not None:
            sim_cmd.append(f"+FINAL_REF_PROGRAM={args.ref_program}")
        rc, output = run(sim_cmd, repo_root, quiet=False)
        if rc != 0 or "PASS:" not in output or "FAIL:" in output:
            failures.append(f"{name}: simulation failed")

    if failures:
        print("[verilog_final_smoke] FAIL")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print("[verilog_final_smoke] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
