"""Group R screen: repaired folded-709 view vs reference, both regimes,
with the UNREPAIRED view screened on identical probes for paired deltas.

Gain application is a per-channel output multiply (GainWrap) — exactly
equivalent to scaling the module's f16 scale rows (row-linear), and
non-destructive to the shared weights. The bench remains the decision gate.

Usage:
  uv run python experiments/groupr/run_repair_screen.py [--items 100]
"""
import argparse
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from bonsaifold.loader import load_bonsai, sublayer_view
from bonsaifold.probes import load_probe_set

REPO = Path(__file__).resolve().parents[2]
PACK = REPO / "models/Bonsai-27B-mlx-1bit-nobias"
GAINS = Path(__file__).parent / "gains_folded709.json"
OUT = Path(__file__).parent / "screen_repair.json"
CHUNK = 256
REGIMES = {
    "onpolicy": REPO / "experiments/calibration/probes_onpolicy.json",
    "offpolicy": REPO / "experiments/calibration/probes_offpolicy.json",
}


class GainWrap:
    def __init__(self, mod, gains):
        self.mod = mod
        self.g = mx.array(gains).astype(mx.float16)

    def __call__(self, *a, **k):
        return self.mod(*a, **k) * self.g


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", type=int, default=100)
    args = ap.parse_args()

    spec = json.loads(GAINS.read_text())
    done = json.loads(OUT.read_text()) if OUT.exists() else {"items": {}}
    model, _ = load_bonsai(PACK)
    view = sublayer_view(model, drop_attn=spec["drop_attn"], drop_mlp=[],
                         drop_blocks=spec["drop_blocks"])

    def set_repair(on):
        for b, site in spec["sites"].items():
            layer = view.layers[site["view_pos"]]
            if on:
                setattr(layer, site["module"],
                        GainWrap(getattr(layer, site["module"]), site["gains"]))
            else:
                setattr(layer, site["module"],
                        getattr(layer, site["module"]).mod)

    for regime, path in REGIMES.items():
        items = load_probe_set(path)["items"][: args.items]
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
        done[f"{regime}"] = {
            "kl_base": sum(r["kl_base"] for r in rows) / nt,
            "kl_repaired": sum(r["kl_rep"] for r in rows) / nt,
        }
        OUT.write_text(json.dumps(done))
        s = done[regime]
        print(f"== {regime}: base {s['kl_base']:.5f} -> repaired "
              f"{s['kl_repaired']:.5f} ({s['kl_repaired']/s['kl_base']:.1%}) ==",
              flush=True)
    print("SCREEN COMPLETE")


if __name__ == "__main__":
    main()
