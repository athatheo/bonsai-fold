"""Group R v2: fit each dropped cluster's LINEAR SHADOW and absorb it
in-format into one surviving module (sign bits + scales may change; pack
stays exactly 1-bit g128, same bytes).

Sites (all linear_attn.out_proj — uniform in_dim = value_dim = 6144, adds
directly to the residual stream at the right depth):
  b17.linear_attn.out_proj   <- blocks {9,12,13,16}
  b40.linear_attn.out_proj   <- attn {37,38}
  b60.linear_attn.out_proj   <- attn {58}

Per site: accumulate normal equations over the CALIBRATION set (never
probes) for  min ||dW z - delta||^2 :  A = sum z z^T,  B = sum z delta^T,
dW^T = (A + lam*I)^{-1} B.  Then requantize W_f32 + dW into the format:
per (row, 128-col group)  s'_g = 2*mean|w*|, signs = sign(w*)  (weights are
+-s_g/2 under the derived-bias convention).  Writes the repaired planes +
diagnostics to shadow_folded709.npz / .json.
"""
import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from bonsaifold.loader import load_bonsai, sublayer_view
from bonsaifold.stio import pack_codes, unpack_codes

REPO = Path(__file__).resolve().parents[2]
PACK = REPO / "models/Bonsai-27B-mlx-1bit-nobias"
CALIB = REPO / "experiments/calibration/calibration_set.json"

# defaults = the confirmed folded-709 fit
DROP_BLOCKS = [16, 12, 13, 9]
DROP_ATTN = [37, 38, 58]
SITES = [(17, 13), (40, 36), (60, 56)]  # (orig block, folded view/pack position)
RIDGE_SWEEP = [1e-3, 1e-2, 1e-1, 3e-1, 1.0, 3.0]  # relative to mean diag(A)
MAX_REL_DW = 0.25  # keep requantization noise second-order


class InputOutputRecorder:
    """Wraps linear_attn; stores the out_proj INPUT for the last forward by
    wrapping the inner out_proj, plus passes through unchanged."""

    def __init__(self, attn):
        self.attn = attn
        self.z = None
        inner = attn.out_proj

        class _Tap:
            def __init__(tap, mod, host):
                tap.mod = mod
                tap.host = host

            def __call__(tap, x):
                mx.eval(x)
                tap.host.z = np.array(x.astype(mx.float32))[0]
                return tap.mod(x)

        self._orig_out = inner
        attn.out_proj = _Tap(inner, self)

    def restore(self):
        self.attn.out_proj = self._orig_out

    def __call__(self, *a, **k):
        return self.attn(*a, **k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folded-pack", default=None,
                    help="folded pack to repair (drops persisted in-pack); "
                         "default: build the folded-709 view from nobias")
    ap.add_argument("--drop-blocks", default=None, help="comma ints (view mode)")
    ap.add_argument("--drop-attn", default=None, help="comma ints (view mode)")
    ap.add_argument("--sites", default=None,
                    help="comma 'origblock:pos' pairs, e.g. 5:4,10:7,14:9,17:11")
    ap.add_argument("--out-prefix", default="shadow_folded709")
    args = ap.parse_args()

    global DROP_BLOCKS, DROP_ATTN, SITES
    if args.drop_blocks is not None:
        DROP_BLOCKS = [int(x) for x in args.drop_blocks.split(",")]
    if args.drop_attn is not None:
        DROP_ATTN = [int(x) for x in args.drop_attn.split(",")] if args.drop_attn else []
    if args.sites:
        SITES = [tuple(int(v) for v in pair.split(":")) for pair in args.sites.split(",")]
    out_npz = Path(__file__).parent / f"{args.out_prefix}.npz"
    out_json = Path(__file__).parent / f"{args.out_prefix}.json"

    items = json.loads(CALIB.read_text())["items"]
    model, _ = load_bonsai(PACK)
    if args.folded_pack:
        folded, _ = load_bonsai(args.folded_pack)
        view = folded.language_model.model  # pack positions == site positions
        view_layers = view.layers
    else:
        view = sublayer_view(model, drop_attn=DROP_ATTN, drop_mlp=[],
                             drop_blocks=DROP_BLOCKS)
        view_layers = view.layers
    tm = model.language_model.model

    site_blocks = {b for b, _ in SITES}
    orig_streams, fold_streams = {}, {}
    pos_to_orig = {pos: b for b, pos in SITES}

    def orig_tap(i, h_in, h_out):
        if i in site_blocks:
            mx.eval(h_out)
            orig_streams[i] = np.array(h_out.astype(mx.float32))[0]

    def fold_tap(i, h_in, h_out):
        if i in pos_to_orig:
            mx.eval(h_out)
            fold_streams[pos_to_orig[i]] = np.array(h_out.astype(mx.float32))[0]

    recorders = {}
    for b, pos in SITES:
        layer = view_layers[pos]
        recorders[b] = InputOutputRecorder(layer.linear_attn)
    view.block_tap = fold_tap

    acc = {b: {"A": None, "B": None, "n": 0} for b, _ in SITES}
    for k, item in enumerate(items):
        ids = mx.array([item["token_ids"]])
        tm.block_tap = orig_tap
        mx.eval(model(ids))
        tm.block_tap = None
        mx.eval(view(ids) if not args.folded_pack else folded(ids))
        for b, _ in SITES:
            z = recorders[b].z                    # (T, in_dim) folded run
            delta = orig_streams[b] - fold_streams[b]  # (T, 5120)
            a = acc[b]
            A_i = z.T @ z
            B_i = z.T @ delta
            a["A"] = A_i if a["A"] is None else a["A"] + A_i
            a["B"] = B_i if a["B"] is None else a["B"] + B_i
            a["n"] += z.shape[0]
        if (k + 1) % 8 == 0:
            print(f"{k + 1}/{len(items)} calibration items", flush=True)

    diag = {"sites": {}}
    planes = {}
    for b, pos in SITES:
        a = acc[b]
        mod = recorders[b]._orig_out              # the real QuantizedLinear
        w_packed = np.array(mod.weight)
        scales = np.array(mod.scales, dtype=np.float32)
        codes = unpack_codes(w_packed, 1)         # (rows, K) in {0,1}
        signs = codes * 2.0 - 1.0
        K = signs.shape[1]
        w_f32 = signs * np.repeat(scales, 128, axis=1)[:, :K] / 2.0
        # ridge sweep: smallest lambda whose correction stays small vs W
        base_lam = float(np.trace(a["A"]) / a["A"].shape[0])
        dW, lam = None, None
        for rel in RIDGE_SWEEP:
            lam = rel * base_lam
            dWT = np.linalg.solve(a["A"] + lam * np.eye(a["A"].shape[0]), a["B"])
            dW = dWT.T
            if np.abs(dW).mean() / np.abs(w_f32).mean() <= MAX_REL_DW:
                break
        w_star = w_f32 + dW

        g = w_star.reshape(w_star.shape[0], -1, 128)
        s_new = 2.0 * np.abs(g).mean(axis=2)      # w = +-s/2
        codes_new = (g > 0).reshape(w_star.shape[0], -1).astype(np.uint32)
        flips = int((codes_new != codes).sum())
        planes[f"b{b}.weight"] = pack_codes(codes_new, 1)
        planes[f"b{b}.scales"] = s_new.astype(np.float16)
        rel_dw = float(np.abs(dW).mean() / np.abs(w_f32).mean())
        s_drift = float(np.abs(s_new - scales).mean() / np.abs(scales).mean())
        diag["sites"][str(b)] = {
            "view_pos": pos, "tokens": a["n"], "ridge_lambda": lam,
            "flips": flips, "flip_frac": flips / codes.size,
            "mean_abs_dW_over_w": rel_dw, "scale_drift": s_drift,
        }
        print(f"site b{b}: flips {flips} ({flips/codes.size:.3%}), "
              f"|dW|/|w| {rel_dw:.3f}, scale drift {s_drift:.3%}", flush=True)

    np.savez(out_npz, **planes)
    np.savez(Path(__file__).parent / f"{args.out_prefix}_normal_eqs.npz",
             **{f"A{b}": acc[b]["A"] for b, _ in SITES},
             **{f"B{b}": acc[b]["B"] for b, _ in SITES})
    diag["drop_blocks"] = DROP_BLOCKS
    diag["drop_attn"] = DROP_ATTN
    diag["folded_pack"] = args.folded_pack
    out_json.write_text(json.dumps(diag))
    print(f"wrote {out_npz} and {out_json}")


if __name__ == "__main__":
    main()
