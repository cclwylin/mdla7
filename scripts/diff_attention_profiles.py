#!/usr/bin/env python3
"""
Compare two attention-model profiles to verify Q-tiled chaining keeps score
tiles in L1.

Usage:
    python3 scripts/diff_attention_profiles.py \\
        batch/output/fast/bmm_softmax_bmm_2.5ms_1g_int8.profile.csv \\
        batch/output/fast/bmm_softmax_bmm_tiled_2.5ms_int8.profile.csv

Output (per file): total cycles, total dram_r/w, dram_w broken down by
op-class (fc(bmm) BMM₁ leg vs BMM₂ leg vs softmax sub-rows vs other), and
the headline savings.
"""
import csv, sys
from pathlib import Path


SOFTMAX_OPS = {"sm_pmax", "sm_sub", "sm_exp", "sm_psum", "sm_div", "softmax"}


def load(path):
    rows = []
    with open(path) as f:
        r = csv.DictReader(f)
        for d in r:
            rows.append({
                "id":      int(d["id"]),
                "op":      d["op"].strip(),
                "cycles":  int(d["cycles_layer"]),
                "dram_r":  int(d["dram_r"]),
                "dram_w":  int(d["dram_w"]),
                "sram_r":  int(d["sram_r"]),
                "sram_w":  int(d["sram_w"]),
            })
    return rows


def classify(rows):
    """For each fc(bmm), find the nearest softmax sub-row by index.  If the
    nearest softmax is FORWARD it's BMM₁ (its output feeds softmax); if it's
    BACKWARD it's BMM₂ (its input comes from softmax).  Handles both the
    legacy 'all BMM₁s, then softmax, then all BMM₂s' layout and the new
    per-micro-block alternating layout."""
    n = len(rows)
    sm_positions = [i for i, r in enumerate(rows) if r["op"] in SOFTMAX_OPS]
    tags = ["other"] * n
    import bisect
    for i, r in enumerate(rows):
        if r["op"] == "fc(bmm)":
            if not sm_positions:
                tags[i] = "bmm2"
                continue
            idx = bisect.bisect_left(sm_positions, i)
            fwd = sm_positions[idx] - i if idx < len(sm_positions) else 10**9
            bwd = i - sm_positions[idx - 1] if idx > 0 else 10**9
            tags[i] = "bmm1" if fwd <= bwd else "bmm2"
        elif r["op"] in SOFTMAX_OPS:
            tags[i] = "softmax"
        else:
            tags[i] = r["op"]
    return tags


def summarize(path):
    rows = load(path)
    tags = classify(rows)
    buckets = {}
    for r, t in zip(rows, tags):
        b = buckets.setdefault(t, dict(n=0, cycles=0, dram_r=0, dram_w=0))
        b["n"]      += 1
        b["cycles"] += r["cycles"]
        b["dram_r"] += r["dram_r"]
        b["dram_w"] += r["dram_w"]
    total = dict(
        n=len(rows),
        cycles=sum(r["cycles"] for r in rows),
        dram_r=sum(r["dram_r"] for r in rows),
        dram_w=sum(r["dram_w"] for r in rows),
    )
    return buckets, total


def fmt_mb(b):  return f"{b/1024/1024:>8.2f} MB"


def print_one(label, path):
    buckets, total = summarize(path)
    print(f"\n== {label}: {path} ==")
    print(f"  total layers      : {total['n']}")
    print(f"  total cycles      : {total['cycles']:>14,d}")
    print(f"  total dram_r      : {total['dram_r']:>14,d}  ({fmt_mb(total['dram_r'])})")
    print(f"  total dram_w      : {total['dram_w']:>14,d}  ({fmt_mb(total['dram_w'])})")
    print(f"  per-class breakdown:")
    for tag in ("bmm1", "softmax", "bmm2", "other"):
        b = buckets.get(tag)
        if not b: continue
        print(f"    {tag:<8s} n={b['n']:>4d}  cycles={b['cycles']:>12,d}"
              f"  dram_r={fmt_mb(b['dram_r'])}  dram_w={fmt_mb(b['dram_w'])}")
    return buckets, total


def main():
    if len(sys.argv) not in (2, 3):
        sys.exit(__doc__)

    base = Path(sys.argv[1])
    a_buckets, a_total = print_one("BASELINE", base)

    if len(sys.argv) == 3:
        comp = Path(sys.argv[2])
        b_buckets, b_total = print_one("TILED   ", comp)

        print(f"\n== DELTA (tiled vs baseline) ==")
        def delta(a, b, key):
            d = b[key] - a[key]
            pct = (d / a[key] * 100) if a[key] else 0.0
            return f"{d:>+15,d}  ({pct:>+6.1f}%)"
        print(f"  cycles : {delta(a_total, b_total, 'cycles')}")
        print(f"  dram_r : {delta(a_total, b_total, 'dram_r')}")
        print(f"  dram_w : {delta(a_total, b_total, 'dram_w')}")
        # Headline: BMM₁ dram_w should drop hugely
        bmm1_a = a_buckets.get("bmm1", dict(dram_w=0))["dram_w"]
        bmm1_b = b_buckets.get("bmm1", dict(dram_w=0))["dram_w"]
        print(f"\n  BMM₁ dram_w : {fmt_mb(bmm1_a)} → {fmt_mb(bmm1_b)}"
              f"  (save {fmt_mb(bmm1_a - bmm1_b)})")
        if bmm1_a > 0:
            print(f"  BMM₁ savings: {(bmm1_a - bmm1_b) / bmm1_a * 100:.1f}% of score spill eliminated")


if __name__ == "__main__":
    main()
