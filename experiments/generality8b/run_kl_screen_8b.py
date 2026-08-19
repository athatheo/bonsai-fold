"""Q3: KL screens for the DENSE Bonsai-8B (mirrors kl_screen/run_kl_screen).

Reference = full 8B (stock loader, dense qwen3); candidates are zero-copy
dense_drop_view configurations. Full-vocab forward KL, chunked f32
log-softmax, per-item checkpointing keyed by (candidate, probe id).

GPU JOB — do not run while a 27B bench is running.

Usage:
  uv run python experiments/generality8b/run_kl_screen_8b.py \
      --probes probes_offpolicy_8b.json --out results/singles_offpolicy_8b.json \
      --candidates b0..b35            # every single block
  ... --candidates "b3,b17" "a5+m9"  # explicit specs (comma=blocks, +=mixed)
"""
import argparse
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.utils import load

from bonsaifold.dense import dense_drop_view

HERE = Path(__file__).parent
PACK = HERE.parents[1] / "models/Bonsai-8B-mlx-1bit"
CHUNK = 256


def parse_spec(spec):
    """'b3,b17' or 'b3+a5+m9' -> (blocks, attn, mlp)."""
    kinds = {"b": [], "a": [], "m": []}
    for tokn in spec.replace(",", "+").split("+"):
        kinds[tokn[0]].append(int(tokn[1:]))
    return kinds["b"], kinds["a"], kinds["m"]


def expand_candidates(cands, n_blocks):
    if len(cands) == 1 and ".." in cands[0]:
        kind = cands[0][0]
        lo, hi = cands[0][1:].split("..")
        return [f"{kind}{i}" for i in range(int(lo), int(hi) + 1)]
    return list(cands)


def item_kl(ref_memmap, cand_logits):
    """KL against a disk-cached f16 reference (np.memmap [n, vocab]).

    RAM note: 8B probes are ~4K tokens x 151K vocab, so an in-RAM reference
    cache would be ~29 GB for 24 probes; the memmap keeps it on SSD and
    streams chunks."""
    total = 0.0
    n = cand_logits.shape[1]
    for i in range(0, n, CHUNK):
        c = slice(i, i + CHUNK)
        lp_r = nn.log_softmax(mx.array(ref_memmap[c]).astype(mx.float32), axis=-1)
        lp_c = nn.log_softmax(cand_logits[:, c][0].astype(mx.float32), axis=-1)
        terms = mx.where(lp_r == -mx.inf, 0.0, mx.exp(lp_r) * (lp_r - lp_c))
        total += float(terms.sum())
    return total, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--candidates", nargs="+", required=True)
    ap.add_argument("--items", type=int, default=24)
    args = ap.parse_args()

    probes = json.loads((HERE / args.probes).read_text())
    items = probes["items"][: args.items]
    out_path = HERE / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fingerprint = {"pack": str(PACK), "probes": args.probes, "items": args.items}
    done = json.loads(out_path.read_text()) if out_path.exists() else {
        "fingerprint": fingerprint, "candidates": []}
    if done["fingerprint"] != fingerprint:
        raise SystemExit(f"{out_path} fingerprint mismatch; move it aside")
    have = {c["name"] for c in done["candidates"]}

    import os

    import numpy as np

    model, _ = load(str(PACK))
    n_blocks = len(model.model.layers)
    cache_dir = Path(os.environ.get("TMPDIR", "/tmp")) / "bonsai8b_ref_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    ref_cache = {}
    for item in items:
        f = cache_dir / f"{item['id']}_{len(item['token_ids'])}.f16"
        n = len(item["token_ids"])
        if not f.exists():
            logits = model(mx.array([item["token_ids"]]))
            mx.eval(logits)
            arr = np.array(logits[0].astype(mx.float16))
            np.memmap(f, dtype=np.float16, mode="w+", shape=arr.shape)[:] = arr
        vocab = model.model.embed_tokens.weight.shape[0]
        ref_cache[item["id"]] = np.memmap(f, dtype=np.float16, mode="r",
                                          shape=(n, vocab))
    print(f"reference disk-cached: {len(ref_cache)} probes", flush=True)

    for spec in expand_candidates(args.candidates, n_blocks):
        name = f"cand[{spec}]"
        if name in have:
            continue
        blocks, attn, mlp = parse_spec(spec)
        view = dense_drop_view(model, drop_blocks=blocks, drop_attn=attn, drop_mlp=mlp)
        total, ntok = 0.0, 0
        for item in items:
            cand = view(mx.array([item["token_ids"]]))
            mx.eval(cand)
            t, n = item_kl(ref_cache[item["id"]], cand)
            total += t
            ntok += n
        rec = {"name": name, "spec": spec, "mean_kl_nats": total / ntok, "ntok": ntok}
        done["candidates"].append(rec)
        out_path.write_text(json.dumps(done))
        print(f"{name}: kl={rec['mean_kl_nats']:.5f}", flush=True)
    print("SCREEN COMPLETE")


if __name__ == "__main__":
    main()
