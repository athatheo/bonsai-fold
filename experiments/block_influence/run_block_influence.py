"""Phase 1: block-influence (redundancy) map via teacher-forced prefill.

Wraps every decoder block with a recorder and streams statistics over the
frozen calibration set — no activations stored, no generation:
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
from pathlib import Path

import mlx.core as mx

from bonsaifold.loader import layer_types_of, load_bonsai

PAIR_MAX_GAP = 3


class BlockRecorder:
    """Forwards to the wrapped block, accumulating streaming statistics."""

    def __init__(self, layer):
        self.layer = layer
        self.sum_cos = 0.0
        self.sum_ang = 0.0
        self.sum_relmag = 0.0
        self.tokens = 0
        self.delta = None  # per-sequence, consumed by pairwise stage

    @property
    def is_linear(self):
        return self.layer.is_linear

    def __call__(self, x, mask=None, cache=None):
        out = self.layer(x, mask=mask, cache=cache)
        xf = x.astype(mx.float32)
        of = out.astype(mx.float32)
        cos = (xf * of).sum(-1) / (
            mx.linalg.norm(xf, axis=-1) * mx.linalg.norm(of, axis=-1) + 1e-9
        )
        delta = of - xf
        relmag = mx.linalg.norm(delta, axis=-1) / (mx.linalg.norm(xf, axis=-1) + 1e-9)
        ang = mx.arccos(mx.clip(cos, -1.0, 1.0)) / mx.array(3.141592653589793)
        n = cos.size
        s_cos, s_ang, s_rel = cos.sum(), ang.sum(), relmag.sum()
        mx.eval(s_cos, s_ang, s_rel)
        self.sum_cos += float(s_cos)
        self.sum_ang += float(s_ang)
        self.sum_relmag += float(s_rel)
        self.tokens += int(n)
        self.delta = delta  # kept until pairwise reduction for this sequence
        return out


def pairwise_pairs(layer_types):
    pairs = []
    for i, t in enumerate(layer_types):
        for j in range(i + 1, min(i + PAIR_MAX_GAP + 1, len(layer_types))):
            if layer_types[j] == t:
                pairs.append((i, j))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--calibration", required=True)
    ap.add_argument("--out", default=str(Path(__file__).parent))
    args = ap.parse_args()

    calib = json.loads(Path(args.calibration).read_text())
    layer_types = layer_types_of(args.pack)
    model, _ = load_bonsai(args.pack)
    tm = model.language_model.model

    recorders = [BlockRecorder(l) for l in tm.layers]
    tm.layers = recorders
    pairs = pairwise_pairs(layer_types)
    pair_sum = {p: 0.0 for p in pairs}
    pair_tokens = {p: 0 for p in pairs}
    per_category = {}

    for item in calib["items"]:
        tokens = mx.array([item["token_ids"]])
        tm(tokens)  # teacher-forced prefill; recorders accumulate
        for i, j in pairs:
            di = recorders[i].delta.astype(mx.float32)
            dj = recorders[j].delta.astype(mx.float32)
            cos = (di * dj).sum(-1) / (
                mx.linalg.norm(di, axis=-1) * mx.linalg.norm(dj, axis=-1) + 1e-9
            )
            s = cos.sum()
            mx.eval(s)
            pair_sum[(i, j)] += float(s)
            pair_tokens[(i, j)] += int(cos.size)
        for r in recorders:
            r.delta = None
        cat = item.get("category", "unknown")
        per_category.setdefault(cat, 0)
        per_category[cat] += len(item["token_ids"])
        print(f"done {item['id']} ({cat}, {len(item['token_ids'])} tokens)")

    blocks = []
    for i, r in enumerate(recorders):
        blocks.append(
            {
                "block": i,
                "attention_type": layer_types[i],
                "tokens": r.tokens,
                "mean_cos": r.sum_cos / r.tokens,
                "block_influence": 1.0 - r.sum_cos / r.tokens,
                "mean_angular": r.sum_ang / r.tokens,
                "mean_rel_update": r.sum_relmag / r.tokens,
            }
        )
    result = {
        "pack": args.pack,
        "calibration": calib.get("meta", {}),
        "tokens_per_category": per_category,
        "blocks": blocks,
        "pairwise_delta_cos": [
            {
                "i": i,
                "j": j,
                "type": layer_types[i],
                "mean_cos": pair_sum[(i, j)] / max(pair_tokens[(i, j)], 1),
            }
            for i, j in pairs
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
