"""Group C KL screen (FLAGGED value-modifying arm): 8-bit scale round-trip
vs untouched reference, both regimes, streaming.

Two derived-kernel models held at once (~7 GB): the candidate gets
apply_roundtrip() (in-memory; identical values to what the future pack
stores), the reference stays exact. Full-vocab forward KL per item, chunked
log-softmax in f32, per-item checkpointing keyed by probe id — resume-safe
across battery deaths.

Usage:
  uv run python experiments/groupc/run_scalequant_screen.py [--items 100]
"""
import argparse
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from bonsaifold.loader import load_bonsai
from bonsaifold.probes import load_probe_set
from bonsaifold.scalequant import apply_roundtrip

REPO = Path(__file__).resolve().parents[2]
PACK = REPO / "models/Bonsai-27B-mlx-1bit-nobias"
OUT = Path(__file__).parent / "screen_results.json"
CHUNK = 256

REGIMES = {
    "onpolicy": REPO / "experiments/calibration/probes_onpolicy.json",
    "offpolicy": REPO / "experiments/calibration/probes_offpolicy.json",
}


def item_kl(ref_model, cand_model, token_ids):
    ids = mx.array([token_ids])
    a = ref_model(ids)
    b = cand_model(ids)
    mx.eval(a, b)
    total, n = 0.0, a.shape[1]
    for s in range(0, n, CHUNK):
        r = nn.log_softmax(a[:, s : s + CHUNK].astype(mx.float32), axis=-1)
        c = nn.log_softmax(b[:, s : s + CHUNK].astype(mx.float32), axis=-1)
        total += float(nn.losses.kl_div_loss(c, r, axis=-1).sum())
    return total, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", type=int, default=100)
    args = ap.parse_args()

    done = json.loads(OUT.read_text()) if OUT.exists() else {"items": {}}

    ref, _ = load_bonsai(PACK)  # stamped -> derived kernels, exact scales
    cand, _ = load_bonsai(PACK)
    n_mod, max_rel = apply_roundtrip(cand)
    mx.eval(cand.parameters())
    print(f"round-tripped {n_mod} modules, max relative scale error {max_rel:.5f}")
    done["max_rel_scale_err"] = max_rel

    for regime, path in REGIMES.items():
        items = load_probe_set(path)["items"][: args.items]
        for item in items:
            key = f"{regime}/{item['id']}"
            if key in done["items"]:
                continue
            total, n = item_kl(ref, cand, item["token_ids"])
            done["items"][key] = {"kl_sum": total, "ntok": n}
            OUT.write_text(json.dumps(done))
            print(f"{key}: n={n} kl={total / n:.5f}", flush=True)
        rows = [v for k, v in done["items"].items() if k.startswith(regime)]
        kl = sum(r["kl_sum"] for r in rows) / sum(r["ntok"] for r in rows)
        done[f"kl_{regime}"] = kl
        OUT.write_text(json.dumps(done))
        print(f"== {regime}: mean KL {kl:.5f} over {len(rows)} items ==", flush=True)

    print("SCREEN COMPLETE")
    print(json.dumps({k: v for k, v in done.items() if k != "items"}, indent=1))


if __name__ == "__main__":
    main()
