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
    """(summed forward KL over positions, n_positions) for one probe.

    -inf-safe: lm_head-trimmed packs emit -inf on dropped vocab rows; where
    the reference is -inf its probability is 0, so the contribution is 0
    (kl_div_loss would produce 0*inf = NaN there)."""
    ids = mx.array([token_ids])
    ref_logits = ref_model(ids)
    cand_logits = cand_model(ids)
    mx.eval(ref_logits, cand_logits)
    total, n = 0.0, ref_logits.shape[1]
    for i in range(0, n, CHUNK):
        chunk = slice(i, i + CHUNK)
        lp_ref = nn.log_softmax(ref_logits[:, chunk].astype(mx.float32), axis=-1)
        lp_cand = nn.log_softmax(cand_logits[:, chunk].astype(mx.float32), axis=-1)
        terms = mx.where(
            lp_ref == -mx.inf, 0.0, mx.exp(lp_ref) * (lp_ref - lp_cand)
        )
        total += float(terms.sum())
    return total, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", type=int, default=100)
    args = ap.parse_args()

    fingerprint = {"pack": str(PACK), "transform": "scaleq8-row"}
    done = json.loads(OUT.read_text()) if OUT.exists() else {"items": {}}
    if done.get("fingerprint", fingerprint) != fingerprint:
        raise SystemExit(
            f"{OUT} holds results for {done['fingerprint']}, not {fingerprint}; "
            "move it aside"
        )
    done["fingerprint"] = fingerprint

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
