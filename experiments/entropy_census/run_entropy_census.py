"""Entropy census (A3ii): how compressible are the pack's byte planes?

Byte-level Shannon entropy of (a) packed u32 sign words, (b) f16 scales,
(c) f16 biases, per tensor class. Bounds any ANS/entropy-coding payoff
(survey item B3) before touching kernels: 8.0 bits/byte == incompressible.
Large tensors are stride-sampled (>=8 MB per tensor still counted).

Usage: uv run python experiments/entropy_census/run_entropy_census.py
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from bonsaifold.stio import PackReader, is_quantized, weight_base

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).parent
SAMPLE_BYTES = 8 << 20


def entropy_bits_per_byte(counts):
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def main():
    pack = PackReader(REPO / "models/Bonsai-27B-mlx-1bit")
    hists = defaultdict(lambda: np.zeros(256, dtype=np.int64))
    totals = defaultdict(int)
    for name in sorted(pack.header):
        if not is_quantized(name, pack.header):
            continue
        base = weight_base(name)
        for cls, tname in [("signs", name), ("scales", base + ".scales"),
                           ("biases", base + ".biases")]:
            begin, end = pack.header[tname]["data_offsets"]
            size = end - begin
            raw = np.frombuffer(pack.read_raw(tname), dtype=np.uint8)
            step = max(1, raw.size // SAMPLE_BYTES)
            hists[cls] += np.bincount(raw[::step], minlength=256)
            totals[cls] += size
    report = {}
    for cls in hists:
        h = entropy_bits_per_byte(hists[cls])
        report[cls] = {
            "total_mb": round(totals[cls] / 1e6, 1),
            "entropy_bits_per_byte": round(h, 4),
            "compressible_fraction": round(1 - h / 8.0, 4),
            "potential_mb": round(totals[cls] / 1e6 * (1 - h / 8.0), 1),
        }
    # scales/biases byte planes interleave exponent+mantissa bytes; also
    # report the two byte lanes separately for the f16 classes
    for cls in ("scales", "biases"):
        name = next(weight_base(n) + f".{cls}" for n in sorted(pack.header)
                    if is_quantized(n, pack.header))
        raw = np.frombuffer(pack.read_raw(name), dtype=np.uint8)
        lo, hi = raw[0::2], raw[1::2]  # f16 little-endian: lo=mantissa, hi=sign+exp+mantissa-top
        report[cls]["lane_entropy_example"] = {
            "low_byte": round(entropy_bits_per_byte(np.bincount(lo, minlength=256)), 3),
            "high_byte": round(entropy_bits_per_byte(np.bincount(hi, minlength=256)), 3),
        }
    (OUT / "entropy_census.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
