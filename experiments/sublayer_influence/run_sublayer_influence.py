"""A1 stage 1: sublayer-granular influence map (128 candidates in one pass).

For every block, measures BOTH sublayers' residual contributions during a
single teacher-forced pass over the calibration set:
  BI_attn(i) = 1 - E_t[cos(x, x + attn_i(x))]
  BI_mlp(i)  = 1 - E_t[cos(h, h + mlp_i(h))]   (h = post-attn stream)
plus relative update magnitudes. ShortGPT-style, at half-block granularity —
the ranking that feeds the A1 KL sweeps. attn sublayer ~16 MB, mlp ~38 MB.

DO NOT run casually: loads the 27B (~15 min).

Usage: uv run python experiments/sublayer_influence/run_sublayer_influence.py
"""
import json
import math
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from bonsaifold.loader import LayerSubsetView, load_bonsai
from bonsaifold.probes import load_probe_set

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).parent
EPS = 1e-9


class SublayerRecorder:
    """Forward-identical to DecoderLayer, accumulating per-sublayer stats."""

    def __init__(self, layer):
        self.layer = layer
        z = lambda: mx.array(0.0)
        self.attn_cos, self.mlp_cos = z(), z()
        self.attn_rel, self.mlp_rel = z(), z()
        self.tokens = 0
        self._seq_tokens = 0

    @property
    def is_linear(self):
        return self.layer.is_linear

    def __call__(self, x, mask=None, cache=None):
        l = self.layer
        attn = l.linear_attn if l.is_linear else l.self_attn
        r = attn(l.input_layernorm(x), mask, cache)
        h = x + r
        m = l.mlp(l.post_attention_layernorm(h))
        out = h + m
        xf, hf = x.astype(mx.float32), h.astype(mx.float32)
        rf, mf = r.astype(mx.float32), m.astype(mx.float32)
        self.attn_cos = self.attn_cos + nn.losses.cosine_similarity_loss(
            xf, xf + rf, axis=-1, eps=EPS
        ).sum()
        self.mlp_cos = self.mlp_cos + nn.losses.cosine_similarity_loss(
            hf, hf + mf, axis=-1, eps=EPS
        ).sum()
        self.attn_rel = self.attn_rel + (
            mx.linalg.norm(rf, axis=-1) / (mx.linalg.norm(xf, axis=-1) + EPS)
        ).sum()
        self.mlp_rel = self.mlp_rel + (
            mx.linalg.norm(mf, axis=-1) / (mx.linalg.norm(hf, axis=-1) + EPS)
        ).sum()
        self._seq_tokens = x.shape[1]
        return out

    def finish(self):
        mx.eval(self.attn_cos, self.mlp_cos, self.attn_rel, self.mlp_rel)
        self.tokens += self._seq_tokens


def main():
    calib = load_probe_set(REPO / "experiments/calibration/calibration_set.json")
    model, _ = load_bonsai(REPO / "models/Bonsai-27B-mlx-1bit")
    tm = model.language_model.model
    layer_types = ["linear" if l.is_linear else "full" for l in tm.layers]

    view = LayerSubsetView(model, list(range(len(tm.layers))))
    recorders = [SublayerRecorder(l) for l in tm.layers]
    view.layers = recorders

    for item in calib["items"]:
        view(mx.array([item["token_ids"]]))
        for r in recorders:
            r.finish()
        print(f"done {item['id']} ({len(item['token_ids'])} tokens)")

    n = recorders[0].tokens
    rows = []
    for i, r in enumerate(recorders):
        for sub, cos, rel in (("attn", r.attn_cos, r.attn_rel), ("mlp", r.mlp_cos, r.mlp_rel)):
            rows.append(
                {
                    "candidate": f"{sub}[{i}]",
                    "block": i,
                    "sublayer": sub,
                    "type": layer_types[i],
                    "influence": 1.0 - float(cos) / n,
                    "mean_rel_update": float(rel) / n,
                }
            )
    rows.sort(key=lambda x: x["influence"])
    result = {"tokens": n, "calibration": calib.get("meta", {}), "candidates": rows}
    (OUT / "sublayer_influence.json").write_text(json.dumps(result, indent=1))
    lines = ["# Sublayer influence (lowest = best drop candidate)", "",
             "| rank | candidate | type | BI | rel update |", "|---|---|---|---|---|"]
    for k, row in enumerate(rows[:40], 1):
        lines.append(
            f"| {k} | {row['candidate']} | {row['type']} "
            f"| {row['influence']:.5f} | {row['mean_rel_update']:.4f} |"
        )
    (OUT / "sublayer_influence.md").write_text("\n".join(lines) + "\n")
    print("--- 12 lowest-influence sublayers:")
    for row in rows[:12]:
        print(f"  {row['candidate']:9s} {row['type']:6s} BI={row['influence']:.5f}")


if __name__ == "__main__":
    main()
