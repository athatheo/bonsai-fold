"""Q5: KV-group singles screen over all full-attention blocks (27B).

64 candidates (16 full-attn blocks x 4 GQA groups), zero-copy adapter
swapped per candidate with restoration verified by construction (screen
aborts if a restore breaks bit-identity on a canary probe). Reference
logits cached in RAM (24 on-policy probes ~ 13 GB, evosearch-proven).

Usage:
  uv run python experiments/headmap/run_head_screen.py \
      [--probes onpolicy|offpolicy] [--items 24] [--candidates b3.g0 ...]
"""
import argparse
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from bonsaifold.heads import install_head_drop, restore_head_drop
from bonsaifold.loader import load_bonsai
from bonsaifold.probes import load_probe_set

REPO = Path(__file__).resolve().parents[2]
PACK = REPO / "models/Bonsai-27B-mlx-1bit-nobias"
PROBES = {
    "onpolicy": REPO / "experiments/calibration/probes_onpolicy.json",
    "offpolicy": REPO / "experiments/calibration/probes_offpolicy.json",
}
CHUNK = 256
FULL_ATTN_BLOCKS = [i for i in range(64) if (i + 1) % 4 == 0]


def kl_vs_ref(ref_logits, cand_logits):
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
    ap.add_argument("--probes", choices=list(PROBES), default="onpolicy")
    ap.add_argument("--items", type=int, default=24)
    ap.add_argument("--candidates", nargs="*", default=None,
                    help="e.g. b3.g0; default = all 64 group singles")
    args = ap.parse_args()

    cands = args.candidates or [
        f"b{b}.g{g}" for b in FULL_ATTN_BLOCKS for g in range(4)
    ]
    out_path = Path(__file__).parent / f"results/singles_{args.probes}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fingerprint = {"pack": str(PACK), "probes": args.probes, "items": args.items}
    done = json.loads(out_path.read_text()) if out_path.exists() else {
        "fingerprint": fingerprint, "candidates": []}
    if done["fingerprint"] != fingerprint:
        raise SystemExit(f"{out_path} fingerprint mismatch; move it aside")
    have = {c["name"] for c in done["candidates"]}

    items = load_probe_set(PROBES[args.probes])["items"][: args.items]
    model, _ = load_bonsai(PACK)
    ref = {}
    for item in items:
        logits = model(mx.array([item["token_ids"]]))
        mx.eval(logits)
        ref[item["id"]] = logits
    canary_ids = mx.array([items[0]["token_ids"][:256]])
    canary = model(canary_ids)
    mx.eval(canary)
    print(f"reference cached: {len(ref)} probes", flush=True)

    for spec in cands:
        name = f"head[{spec}]"
        if name in have:
            continue
        b, g = spec.split(".")
        block, group = int(b[1:]), int(g[1:])
        orig = install_head_drop(model, block, [group])
        total, ntok = 0.0, 0
        for item in items:
            cand = model(mx.array([item["token_ids"]]))
            mx.eval(cand)
            t, n = kl_vs_ref(ref[item["id"]], cand)
            total += t
            ntok += n
        restore_head_drop(model, block, orig)
        check = model(canary_ids)
        mx.eval(check)
        if not bool(mx.array_equal(check, canary)):
            raise SystemExit(f"restore after {spec} broke the canary — aborting")
        rec = {"name": name, "spec": spec, "mean_kl_nats": total / ntok, "ntok": ntok}
        done["candidates"].append(rec)
        out_path.write_text(json.dumps(done))
        print(f"{name}: kl={rec['mean_kl_nats']:.6f}", flush=True)
    print("SCREEN COMPLETE")


if __name__ == "__main__":
    main()
