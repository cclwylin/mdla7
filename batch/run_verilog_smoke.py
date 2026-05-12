#!/usr/bin/env python3
"""Build and run the verilog block-level smoke tests."""

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
    ("closed_loop", "Testbench_host_program"),
)

WORDS_PER_COMMAND = 32
OP_DONE = 0
OP_CONV = 1
OP_EWE = 3
OP_POOL = 4
OP_TNPS = 5
OP_UDMA = 6
OP_L1CRC = 7
SMF_LOAD_A = 0x01
SMF_COMPUTE = 0x04
SMF_STORE = 0x08
SMF_FINAL_TILE = 0x10
FLAG_WRITE = 1 << 0
FLAG_TNPS_S2D = 1 << 1
FLAG_FINAL = 1 << 6
FLAG_SRAMCRC = 1 << 10
FLAG_READ_L1 = 1 << 11
FLAG_MICROBLOCK = 1 << 13


def fnv_bytes(data: bytes) -> int:
    value = 0x811C9DC5
    for byte in data:
        value = ((value ^ byte) * 16777619) & 0xFFFFFFFF
    return value


def command(op: int) -> list[int]:
    words = [0] * WORDS_PER_COMMAND
    words[0] = op
    return words


def write_hex(path: Path, commands: list[list[int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as stream:
        for words in commands:
            for word in words:
                stream.write(f"{word & 0xFFFFFFFF:08x}\n")


def make_closed_loop_program(obj_root: Path) -> tuple[Path, Path]:
    """Generate a compact DRAM->UDMA->L1->engine->L1->UDMA->DRAM smoke."""
    program_dir = obj_root / "programs"
    ref_path = program_dir / "closed_loop_all_ref.bin"
    program_path = program_dir / "closed_loop_all_engines.verilog.hex"

    ref_bytes = bytearray(1024)
    ref_bytes[0:16] = bytes([1, 5, 2, 3, 9, 8, 7, 6, 0, 0, 0, 0, 0, 0, 0, 0])
    ref_bytes[32:48] = bytes([10, 11, 12, 13, 14, 15, 16, 17,
                              18, 19, 20, 21, 22, 23, 24, 25])
    ref_bytes[64:80] = bytes([7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    ref_bytes[96:112] = bytes([4, 3, 2, 1, 0, 7, 252, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    ref_bytes[112:128] = bytes([3, 255, 1, 2, 0, 6, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    program_dir.mkdir(parents=True, exist_ok=True)
    ref_path.write_bytes(ref_bytes)

    commands: list[list[int]] = []

    def udma_load(ref_off: int, l1_addr: int, size: int) -> None:
        words = command(OP_UDMA)
        words[1] = size
        words[2] = l1_addr
        words[3] = FLAG_MICROBLOCK | (SMF_LOAD_A << 24)
        words[4] = size
        words[5] = 1
        words[25] = ref_off
        commands.append(words)

    def udma_store(l1_addr: int, ref_off: int, size: int) -> None:
        words = command(OP_UDMA)
        words[1] = size
        words[2] = l1_addr
        words[3] = FLAG_MICROBLOCK | FLAG_WRITE | FLAG_FINAL | (SMF_STORE << 24)
        words[4] = size
        words[5] = 1
        words[25] = ref_off
        commands.append(words)

    def l1crc(l1_addr: int, expected: bytes) -> None:
        words = command(OP_L1CRC)
        words[1] = len(expected)
        words[2] = l1_addr
        words[3] = FLAG_SRAMCRC | FLAG_MICROBLOCK | (SMF_FINAL_TILE << 24)
        words[28] = fnv_bytes(expected)
        words[29] = len(expected)
        commands.append(words)

    # POOL: DRAM -> UDMA -> L1 -> POOL -> L1 -> UDMA -> DRAM -> reload -> L1CRC.
    udma_load(0, 0x1000, 16)
    words = command(OP_POOL)
    words[1] = 4
    words[2] = 0x1000
    words[3] = FLAG_MICROBLOCK | FLAG_READ_L1 | FLAG_FINAL | ((SMF_COMPUTE | SMF_FINAL_TILE) << 24)
    words[12] = 4
    words[18] = 5
    words[27] = 0x1100
    commands.append(words)
    udma_store(0x1100, 0x300, 1)
    udma_load(0x300, 0x1200, 1)
    l1crc(0x1200, bytes([5]))

    # TNPS: read one byte from L1 using the SPACE_TO_DEPTH address mapping and write it back.
    udma_load(32, 0x2000, 16)
    words = command(OP_TNPS)
    words[1] = 1
    words[2] = 0x2000
    words[3] = FLAG_MICROBLOCK | FLAG_FINAL | FLAG_TNPS_S2D | ((SMF_COMPUTE | SMF_FINAL_TILE) << 24)
    words[6] = 4
    words[7] = 4
    words[8] = 1
    words[9] = 2
    words[10] = 2
    words[11] = 4
    words[12] = 2
    words[13] = 1
    words[27] = 0x2100
    commands.append(words)
    udma_store(0x2100, 0x400, 1)
    udma_load(0x400, 0x2200, 1)
    l1crc(0x2200, bytes([10]))

    # EWE: A comes from L1, B from descriptor, then output is written through L1/UDMA.
    udma_load(64, 0x3000, 16)
    words = command(OP_EWE)
    words[1] = 1
    words[2] = 0x3000
    words[3] = FLAG_MICROBLOCK | FLAG_READ_L1 | ((SMF_COMPUTE | SMF_FINAL_TILE) << 24)
    words[8] = 5
    words[12] = 1
    words[18] = 12
    words[27] = 0x3100
    commands.append(words)
    udma_store(0x3100, 0x500, 1)
    udma_load(0x500, 0x3200, 1)
    l1crc(0x3200, bytes([12]))

    # CONV: activation and weight vectors come from L1, then the output is stored through UDMA.
    udma_load(96, 0x4000, 32)
    words = command(OP_CONV)
    words[1] = 16
    words[2] = 0x4000
    words[3] = FLAG_MICROBLOCK | FLAG_READ_L1 | FLAG_FINAL | ((SMF_COMPUTE | SMF_FINAL_TILE) << 24)
    words[12] = 6
    words[13] = 5
    words[14] = 1073741824
    words[15] = 1
    words[16] = -128 & 0xFFFFFFFF
    words[17] = 127
    words[18] = 244
    words[20] = 0x00060001
    words[21] = 0x00010001
    words[22] = 0x01010601
    words[23] = 0x05000101
    words[24] = 0x00030000
    words[27] = 0
    words[30] = 6
    words[31] = (1 << 16) | (4 << 8) | 3
    commands.append(words)
    udma_store(0x0, 0x600, 1)
    udma_load(0x600, 0x4200, 1)
    l1crc(0x4200, bytes([244]))

    commands.append(command(OP_DONE))
    write_hex(program_path, commands)
    return program_path, ref_path


def repo_paths() -> tuple[Path, Path]:
    script = Path(__file__).resolve()
    repo_root = script.parents[1]
    return repo_root, repo_root / "rtl"


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
                    help="hex descriptor stream passed to the host smoke test as +VERILOG_PROGRAM")
    ap.add_argument("--ref-program", type=Path, default=None,
                    help="original MDL7 .bin passed to the host smoke test as +VERILOG_REF_PROGRAM")
    ap.set_defaults(repo_root=repo_root, rtl_dir=rtl_dir)
    return ap.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    repo_root: Path = args.repo_root
    rtl_dir: Path = args.rtl_dir
    verilog_dir = rtl_dir / "verilog"
    filelist = verilog_dir / "filelist_system_tb.f"
    verilator = args.verilator.resolve() if args.verilator != Path("verilator") else args.verilator
    selected = set(args.test or [name for name, _ in TESTS])
    cxx_include = args.cxx_stdlib_include or find_cxx_stdlib_include()
    obj_root = rtl_dir / "obj" / "verilog"
    obj_root.mkdir(parents=True, exist_ok=True)

    print(f"[verilog_smoke] host: {platform.system()} {platform.machine()}")
    print(f"[verilog_smoke] tests: {','.join(name for name, _ in TESTS if name in selected)}")

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
                "-Irtl/verilog",
                "-f",
                str(filelist),
                str(verilog_dir / f"{top}.v"),
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
        program = args.program
        ref_program = args.ref_program
        if name == "closed_loop":
            program, ref_program = make_closed_loop_program(obj_root)
        if top == "Testbench_host_program" and program is not None:
            sim_cmd.append(f"+VERILOG_PROGRAM={program}")
        if top == "Testbench_host_program" and ref_program is not None:
            sim_cmd.append(f"+VERILOG_REF_PROGRAM={ref_program}")
        rc, output = run(sim_cmd, repo_root, quiet=False)
        if rc != 0 or "PASS:" not in output or "FAIL:" in output:
            failures.append(f"{name}: simulation failed")

    if failures:
        print("[verilog_smoke] FAIL")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print("[verilog_smoke] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
