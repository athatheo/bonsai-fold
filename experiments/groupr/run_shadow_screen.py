"""Group R v2 paired screen: shadow-repaired folded-709 vs unrepaired,
identical probes, both regimes. Repaired planes are swapped in via
non-destructive shims (npz weight/scales, derived-kernel matmul); originals
restored and canary-checked after. Bench remains the decision gate.
"""
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from bonsaifold.loader import load_bonsai, sublayer_view
from bonsaifold.probes import load_probe_set

REPO = Path(__file__).resolve().parents[2]
PACK = REPO / "models/Bonsai-27B-mlx-1bit-nobias"
NPZ = Path(__file__).parent / "shadow_folded709.npz"
META = Path(__file__).parent / "shadow_folded709.json"
OUT = Path(__file__).parent / "screen_shadow.json"
CHUNK = 256
REGIMES = {
    "onpolicy": REPO / "experiments/calibration/probes_onpolicy.json",
    "offpolicy": REPO / "experiments/calibration/probes_offpolicy.json",
}


class PlaneSwap:
    """out_proj stand-in over repaired (weight, scales) planes."""

    def __init__(self, src, w, s):
        self.weight = mx.array(w)
        self.scales = mx.array(s)
        self.group_size, self.bits, self.mode = src.group_size, src.bits, src.mode

    def __call__(self, x):
        return mx.quantized_matmul(
            x, self.weight, scales=self.scales, biases=None, transpose=True,
            group_size=self.group_size, bits=self.bits, mode=self.mode,
        )


def item_kl(ref_logits, cand_logits):
    total, n = 0.0, cand_logits.shape[1]
    for i in range(0, n, CHUNK):
        c = slice(i, i + CHUNK)
        lp_r = nn.log_softmax(ref_logits[:, c].astype(mx.float32), axis=-1)
        lp_c = nn.log_softmax(cand_logits[:, c].astype(mx.float32), axis=-1)
        total += float(mx.where(lp_r == -mx.inf, 0.0,
                                mx.exp(lp_r) * (lp_r - lp_c)).sum())
    return total, n


def main():
    meta = json.loads(META.read_text())
    planes = np.load(NPZ)
    done = json.loads(OUT.read_text()) if OUT.exists() else {"items": {}}
    model, _ = load_bonsai(PACK)
    view = sublayer_view(model, drop_attn=meta["drop_attn"], drop_mlp=[],
                         drop_blocks=meta["drop_blocks"])

    sites = {int(b): s for b, s in meta["sites"].items()}
    originals = {}

    def set_repair(on):
        for b, s in sites.items():
            attn = view.layers[s["view_pos"]].linear_attn
            if on:
                originals[b] = attn.out_proj
                attn.out_proj = PlaneSwap(
                    attn.out_proj, planes[f"b{b}.weight"], planes[f"b{b}.scales"]
                )
            else:
                attn.out_proj = originals[b]

    for regime, path in REGIMES.items():
        items = load_probe_set(path)["items"][:100]
        for item in items:
            key = f"{regime}/{item['id']}"
            if key in done["items"]:
                continue
            ids = mx.array([item["token_ids"]])
            ref = model(ids)
            mx.eval(ref)
            base = view(ids)
            mx.eval(base)
            kl_base, n = item_kl(ref, base)
            set_repair(True)
            rep = view(ids)
            mx.eval(rep)
            kl_rep, _ = item_kl(ref, rep)
            set_repair(False)
            done["items"][key] = {"kl_base": kl_base, "kl_rep": kl_rep, "ntok": n}
            OUT.write_text(json.dumps(done))
            print(f"{key}: base={kl_base/n:.5f} repaired={kl_rep/n:.5f}", flush=True)
        rows = [v for k, v in done["items"].items() if k.startswith(regime)]
        nt = sum(r["ntok"] for r in rows)
        done[regime] = {"kl_base": sum(r["kl_base"] for r in rows) / nt,
                        "kl_repaired": sum(r["kl_rep"] for r in rows) / nt}
        OUT.write_text(json.dumps(done))
        s = done[regime]
        print(f"== {regime}: base {s['kl_base']:.5f} -> repaired "
              f"{s['kl_repaired']:.5f} ({s['kl_repaired']/s['kl_base']:.1%}) ==",
              flush=True)
    print("SCREEN COMPLETE")


if __name__ == "__main__":
    main()
