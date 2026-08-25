"""Group R (FLAGGED arm 2): fit per-channel repair gains for folded-709.

Repair sites (one per dropped cluster, the nearest surviving same-type
module downstream):
  - blocks {9,12,13,16} dropped  -> block 17 mlp (linear block) rows
  - attn {37,38} dropped         -> block 40 linear_attn.out_proj rows
    (39 is FULL attention — nearest surviving same-type is 40)
  - attn {58} dropped            -> block 60 linear_attn.out_proj rows

Per channel r the repaired stream is  s_folded + (g_r - 1) * y_r  where y is
the repair module's output; matching the ORIGINAL stream at the same depth
gives the closed-form ridge solution  g_r = 1 + Σ y_r·δ_r / (Σ y_r² + λ),
δ = s_orig - s_folded, accumulated online over the CALIBRATION set only
(probe sets are never touched — leakage guard).

Alignment: whole-block drops renumber; orig block 17/40/60 sit at folded
view positions 13/36/56 (four dropped blocks below 17; attn drops do not
renumber). Writes gains + fit diagnostics to gains_folded709.json.
"""
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from bonsaifold.loader import load_bonsai, sublayer_view

REPO = Path(__file__).resolve().parents[2]
PACK = REPO / "models/Bonsai-27B-mlx-1bit-nobias"
CALIB = REPO / "experiments/calibration/calibration_set.json"
OUT = Path(__file__).parent / "gains_folded709.json"

DROP_BLOCKS = [16, 12, 13, 9]
DROP_ATTN = [37, 38, 58]
# (orig block, folded view position, module attr path)
SITES = [
    (17, 13, "mlp"),
    (40, 36, "linear_attn"),
    (60, 56, "linear_attn"),
]
RIDGE = 1e-4


class ModuleOutputRecorder:
    """Wraps a submodule; stores its latest output (f32 numpy)."""

    def __init__(self, mod):
        self.mod = mod
        self.last = None

    def __call__(self, *a, **k):
        out = self.mod(*a, **k)
        mx.eval(out)
        self.last = np.array(out.astype(mx.float32))[0]
        return out


def main():
    items = json.loads(CALIB.read_text())["items"]
    model, _ = load_bonsai(PACK)
    view = sublayer_view(model, drop_attn=DROP_ATTN, drop_mlp=[], drop_blocks=DROP_BLOCKS)

    tm = model.language_model.model
    # taps on the ORIGINAL model at the site blocks
    orig_streams = {}

    def orig_tap(i, h_in, h_out):
        if i in {s[0] for s in SITES}:
            mx.eval(h_out)
            orig_streams[i] = np.array(h_out.astype(mx.float32))[0]

    # recorders + taps on the folded view (positions renumbered)
    recorders = {}
    for orig_b, pos, attr in SITES:
        layer = view.layers[pos]
        recorders[orig_b] = ModuleOutputRecorder(getattr(layer, attr))
        setattr(layer, attr, recorders[orig_b])
    fold_streams = {}
    pos_to_orig = {pos: b for b, pos, _ in SITES}

    def fold_tap(i, h_in, h_out):
        if i in pos_to_orig:
            mx.eval(h_out)
            fold_streams[pos_to_orig[i]] = np.array(h_out.astype(mx.float32))[0]

    view.block_tap = fold_tap

    acc = {b: {"yy": None, "yd": None, "n": 0} for b, _, _ in SITES}
    for k, item in enumerate(items):
        ids = mx.array([item["token_ids"]])
        tm.block_tap = orig_tap
        mx.eval(model(ids))
        tm.block_tap = None
        mx.eval(view(ids))
        for b, _, _ in SITES:
            y = recorders[b].last            # (T, C) module output, folded run
            delta = orig_streams[b] - fold_streams[b]
            a = acc[b]
            yy, yd = (y * y).sum(0), (y * delta).sum(0)
            a["yy"] = yy if a["yy"] is None else a["yy"] + yy
            a["yd"] = yd if a["yd"] is None else a["yd"] + yd
            a["n"] += y.shape[0]
        if (k + 1) % 8 == 0:
            print(f"{k + 1}/{len(items)} calibration items", flush=True)

    out = {"pack": str(PACK), "config": "folded-709",
           "drop_blocks": DROP_BLOCKS, "drop_attn": DROP_ATTN,
           "ridge": RIDGE, "sites": {}}
    for b, pos, attr in SITES:
        a = acc[b]
        lam = RIDGE * float(a["yy"].mean())
        g = 1.0 + a["yd"] / (a["yy"] + lam)
        out["sites"][str(b)] = {
            "view_pos": pos, "module": attr,
            "gains": [float(x) for x in g],
            "gain_mean": float(g.mean()), "gain_std": float(g.std()),
            "gain_min": float(g.min()), "gain_max": float(g.max()),
            "tokens": a["n"],
        }
        print(f"site b{b}.{attr}: gains mean {g.mean():.4f} std {g.std():.4f} "
              f"range [{g.min():.3f}, {g.max():.3f}]", flush=True)
    OUT.write_text(json.dumps(out))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
