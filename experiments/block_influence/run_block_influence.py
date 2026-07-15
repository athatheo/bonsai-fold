"""Phase 1: block-influence (redundancy) map via teacher-forced prefill.

Uses the loader's block tap to stream statistics over the frozen
calibration set — the module tree is never touched, no activations are
stored beyond a 4-block sliding window, and all accumulators stay lazy
until one eval per sequence:
  - BI_i = 1 - E_t[cos(h_in, h_out)]      (ShortGPT-style)
  - angular distance arccos(cos)/pi        (mean)
  - relative update magnitude E[|delta|/|h_in|]
  - pairwise E[cos(delta_i, delta_j)] for same-type pairs with gap <= 3
    (merge-pair shortlist; delta = h_out - h_in at each block's own input)

DO NOT run while the machine is needed elsewhere: loads the 27B pack.
Expected runtime: ~10 min for 64 x 2K tokens (prefill ~240 tok/s).

Usage:
  uv run python experiments/block_influence/run_block_influence.py \
      --pack models/Bonsai-27B-mlx-1bit \
      --calibration experiments/calibration/calibration_set.json \
      [--out experiments/block_influence]
"""
import argparse
import json
import math
from collections import Counter
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from bonsaifold.loader import layer_types_of, load_bonsai
from bonsaifold.probes import load_probe_set

PAIR_MAX_GAP = 3
EPS = 1e-9


class InfluenceRecorder:
    """Accumulates per-block and pairwise statistics from the block tap.

    All sums are lazy mx scalars; call finish_sequence() once per sequence
    to evaluate them. Deltas live in a sliding window of PAIR_MAX_GAP+1
    entries (normalized, f32), so peak extra memory is ~4 tokens x hidden
    vectors instead of one per block.
    """

    def __init__(self, layer_types):
        self.layer_types = layer_types
        z = lambda: mx.array(0.0)
        self.sum_cos = [z() for _ in layer_types]
        self.sum_ang = [z() for _ in layer_types]
        self.sum_rel = [z() for _ in layer_types]
        self.tokens = 0
        self.pairs = [
            (i, j)
            for i, t in enumerate(layer_types)
            for j in range(i + 1, min(i + PAIR_MAX_GAP + 1, len(layer_types)))
            if layer_types[j] == t
        ]
        self.pair_sum = {p: z() for p in self.pairs}
        self._window = {}  # block index -> normalized delta (f32)
        self._seq_tokens = 0

    def tap(self, i, h_in, h_out):
        xf = h_in.astype(mx.float32)
        of = h_out.astype(mx.float32)
        cos = nn.losses.cosine_similarity_loss(xf, of, axis=-1, eps=EPS)
        delta = of - xf
        dnorm = mx.linalg.norm(delta, axis=-1, keepdims=True)
        self.sum_cos[i] = self.sum_cos[i] + cos.sum()
        ang = mx.arccos(mx.clip(cos, -1.0, 1.0)) / math.pi
        self.sum_ang[i] = self.sum_ang[i] + ang.sum()
        rel = dnorm.squeeze(-1) / (mx.linalg.norm(xf, axis=-1) + EPS)
        self.sum_rel[i] = self.sum_rel[i] + rel.sum()
        dhat = delta / mx.maximum(dnorm, EPS)
        for j in list(self._window):
            if j <= i - PAIR_MAX_GAP - 1:
                del self._window[j]
            elif (j, i) in self.pair_sum:
                self.pair_sum[(j, i)] = (
                    self.pair_sum[(j, i)] + (self._window[j] * dhat).sum(-1).sum()
                )
        self._window[i] = dhat
        if i == 0:
            self._seq_tokens = cos.size

    def finish_sequence(self):
        self._window.clear()
        mx.eval(
            *self.sum_cos, *self.sum_ang, *self.sum_rel, *self.pair_sum.values()
        )
        self.tokens += self._seq_tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--calibration", required=True)
    ap.add_argument("--out", default=str(Path(__file__).parent))
    args = ap.parse_args()

    calib = load_probe_set(args.calibration)  # fails fast BEFORE model load
    layer_types = layer_types_of(args.pack)
    model, _ = load_bonsai(args.pack)
    if not hasattr(model, "set_block_tap"):
        raise SystemExit(
            f"{args.pack} did not load through the typed qwen3_5 path "
            "(no block tap); this harness only supports Bonsai packs"
        )
    tm = model.language_model.model

    rec = InfluenceRecorder(layer_types)
    model.set_block_tap(rec.tap)
    tokens_per_category = Counter()
    for item in calib["items"]:
        tm(mx.array([item["token_ids"]]))  # teacher-forced prefill
        rec.finish_sequence()
        cat = item.get("category", "unknown")
        tokens_per_category[cat] += len(item["token_ids"])
        print(f"done {item['id']} ({cat}, {len(item['token_ids'])} tokens)")
    model.set_block_tap(None)

    n = rec.tokens
    blocks = [
        {
            "block": i,
            "attention_type": layer_types[i],
            "tokens": n,
            "mean_cos": float(rec.sum_cos[i]) / n,
            "block_influence": 1.0 - float(rec.sum_cos[i]) / n,
            "mean_angular": float(rec.sum_ang[i]) / n,
            "mean_rel_update": float(rec.sum_rel[i]) / n,
        }
        for i in range(len(layer_types))
    ]
    result = {
        "pack": args.pack,
        "calibration": calib.get("meta", {}),
        "tokens_per_category": dict(tokens_per_category),
        "blocks": blocks,
        "pairwise_delta_cos": [
            {
                "i": i,
                "j": j,
                "type": layer_types[i],
                "mean_cos": float(rec.pair_sum[(i, j)]) / n,
            }
            for i, j in rec.pairs
        ],
    }
    name = Path(args.pack).name
    out = Path(args.out) / f"block_influence_{name}.json"
    out.write_text(json.dumps(result, indent=1))

    ranked = sorted(blocks, key=lambda b: b["block_influence"])
    lines = [
        f"# Block influence: {name}",
        "",
        "Lowest influence first (best drop candidates):",
        "",
        "| rank | block | type | BI = 1-cos | angular | rel. update |",
        "|---|---|---|---|---|---|",
    ]
    for rank, b in enumerate(ranked, 1):
        lines.append(
            f"| {rank} | {b['block']} | {b['attention_type'].split('_')[0]} "
            f"| {b['block_influence']:.5f} | {b['mean_angular']:.5f} "
            f"| {b['mean_rel_update']:.4f} |"
        )
    (Path(args.out) / f"block_influence_{name}.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
