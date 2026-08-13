"""A6: (1+lambda) evolutionary search over per-block operator genomes.

Genome: 64 ints, one per block — 0=keep, 1=drop_block, 2=drop_attn,
3=drop_mlp. Validity: block 0 always keep; at least one surviving
full-attention and one surviving linear attention sublayer (mask anchors).
Fitness (stage 1): mean full-vocab forward KL on a FIXED 24-probe on-policy
subset, reference logits cached in RAM once. Selection is lexicographic
within a structural-byte tier: candidates below the tier's byte floor are
invalid; among valid, lower KL wins. Elites (new best per generation) are
flagged for 100-probe two-regime confirmation (separate script pass).

Checkpointing: every evaluated genome is appended to the ledger JSON
(genome-hash -> bytes, kl); generations resume mid-stream. Population is
seeded from measured-good configs, mutation flips 1-3 blocks weighted
toward operators that were cheap in the singles screens.

DO NOT run casually: holds the 27B + ~13 GB of cached reference logits.
~2.5 min per candidate; a 16-candidate generation ~40 min; run overnight
under the standing continuous-run approval.

Usage:
  uv run python experiments/evosearch/run_evosearch.py \
      --tier-mb 350 --generations 12 [--lam 16] [--seed 20260812]
"""
import argparse
import hashlib
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from bonsaifold.loader import LayerSubsetView, SublayerAdapter, load_bonsai
from bonsaifold.probes import load_probe_set

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).parent
PACK = REPO / "models/Bonsai-27B-mlx-1bit-nobias"
KEEP, DROP_BLOCK, DROP_ATTN, DROP_MLP = 0, 1, 2, 3
N_BLOCKS = 64
STAGE1_PROBES = 24
CHUNK = 256

# per-op structural bytes (MB), census-derived
LINEAR_BLOCK, FULL_BLOCK = 60.0, 58.2
LINEAR_ATTN, FULL_ATTN, MLP = 16.2, 20.2, 38.0

# measured-good seeds (folded-709 config and screened relatives)
SEEDS = [
    {16: 1, 12: 1, 13: 1, 9: 1, 38: 2, 37: 2, 58: 2},
    {16: 1, 12: 1, 13: 1, 9: 1},
    {38: 2, 37: 2, 58: 2, 57: 2, 61: 2, 8: 2},
]
# singles-screen-cheap blocks get higher mutation weight
CHEAP_ATTN = [38, 37, 58, 57, 61, 8, 5, 4, 12, 9]
CHEAP_MLP = [4, 12]
CHEAP_BLOCK = [16, 12, 13, 9, 8, 4, 5]


def genome_bytes(g, layer_types):
    mb = 0.0
    for i, op in enumerate(g):
        lin = layer_types[i] == "linear_attention"
        if op == DROP_BLOCK:
            mb += LINEAR_BLOCK if lin else FULL_BLOCK
        elif op == DROP_ATTN:
            mb += LINEAR_ATTN if lin else FULL_ATTN
        elif op == DROP_MLP:
            mb += MLP
    return mb


def valid(g, layer_types):
    if g[0] != KEEP:
        return False
    fa = any(
        layer_types[i] == "full_attention" and g[i] in (KEEP, DROP_MLP)
        for i in range(len(g))
    )
    ssm = any(
        layer_types[i] == "linear_attention" and g[i] in (KEEP, DROP_MLP)
        for i in range(len(g))
    )
    return fa and ssm


def build_view(model, g):
    keep_blocks = [i for i in range(len(g)) if g[i] != DROP_BLOCK]
    view = LayerSubsetView(model, keep_blocks)
    tm_layers = model.language_model.model.layers
    new_layers = []
    for pos, i in enumerate(keep_blocks):
        if g[i] == DROP_ATTN:
            new_layers.append(SublayerAdapter(tm_layers[i], True, False))
        elif g[i] == DROP_MLP:
            new_layers.append(SublayerAdapter(tm_layers[i], False, True))
        else:
            new_layers.append(tm_layers[i])
    view.layers = new_layers
    view.fa_idx = next(
        (p for p, i in enumerate(keep_blocks)
         if not tm_layers[i].is_linear and g[i] != DROP_ATTN), None)
    view.ssm_idx = next(
        (p for p, i in enumerate(keep_blocks)
         if tm_layers[i].is_linear and g[i] != DROP_ATTN), None)
    return view


def gh(g):
    return hashlib.sha256(bytes(g)).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier-mb", type=float, required=True)
    ap.add_argument("--generations", type=int, default=12)
    ap.add_argument("--lam", type=int, default=16)
    ap.add_argument("--seed", type=int, default=20260812)
    args = ap.parse_args()

    ledger_path = OUT / f"ledger_t{int(args.tier_mb)}.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {}
    rng = np.random.default_rng(args.seed)

    probes = load_probe_set(REPO / "experiments/calibration/probes_onpolicy.json")
    items = probes["items"][:STAGE1_PROBES]
    model, _ = load_bonsai(PACK)
    layer_types = json.loads((PACK / "config.json").read_text())["text_config"]["layer_types"]

    print("caching reference logits (f16) for stage-1 probes ...")
    ref_cache = []
    for item in items:
        logits = model(mx.array([item["token_ids"]]))
        mx.eval(logits)
        ref_cache.append(logits)

    def kl_of(view):
        import mlx.nn as nn
        total, ntok = 0.0, 0
        for item, ref_logits in zip(items, ref_cache):
            cand = view(mx.array([item["token_ids"]]))
            n = ref_logits.shape[1]
            for s in range(0, n, CHUNK):
                r = nn.log_softmax(ref_logits[:, s:s+CHUNK].astype(mx.float32), axis=-1)
                c = nn.log_softmax(cand[:, s:s+CHUNK].astype(mx.float32), axis=-1)
                total += float(nn.losses.kl_div_loss(c, r, axis=-1).sum())
            ntok += n
        return total / ntok

    def evaluate(g):
        h = gh(g)
        if h in ledger:
            return ledger[h]
        mb = genome_bytes(g, layer_types)
        rec = {"genome": list(g), "bytes_mb": round(mb, 1), "kl": None}
        if valid(g, layer_types) and mb >= args.tier_mb:
            rec["kl"] = kl_of(build_view(model, g))
        ledger[h] = rec
        ledger_path.write_text(json.dumps(ledger))
        tag = f"kl={rec['kl']:.5f}" if rec["kl"] is not None else "invalid/under-tier"
        print(f"eval {h}: {mb:.0f} MB {tag}")
        return rec

    def mutate(g):
        g = list(g)
        for _ in range(rng.integers(1, 4)):
            r = rng.random()
            if r < 0.4:
                i = int(rng.choice(CHEAP_ATTN)); g[i] = DROP_ATTN
            elif r < 0.55:
                i = int(rng.choice(CHEAP_BLOCK)); g[i] = DROP_BLOCK
            elif r < 0.65:
                i = int(rng.choice(CHEAP_MLP)); g[i] = DROP_MLP
            elif r < 0.85:
                i = int(rng.integers(1, N_BLOCKS)); g[i] = int(rng.integers(0, 4))
            else:
                i = int(rng.integers(1, N_BLOCKS)); g[i] = KEEP
        return g

    # seed population: grow measured-good configs to the tier
    def seed_genome(ops):
        g = [KEEP] * N_BLOCKS
        for i, op in ops.items():
            g[i] = op
        while genome_bytes(g, layer_types) < args.tier_mb:
            g = mutate(g)
        return g

    best = None
    for s in SEEDS:
        rec = evaluate(seed_genome(dict(s)))
        if rec["kl"] is not None and (best is None or rec["kl"] < best["kl"]):
            best = rec
    print(f"seed best: {best['bytes_mb']} MB kl={best['kl']:.5f}")

    for gen in range(args.generations):
        children = [mutate(best["genome"]) for _ in range(args.lam)]
        for ch in children:
            rec = evaluate(ch)
            if rec["kl"] is not None and rec["kl"] < best["kl"]:
                best = rec
                print(f"GEN {gen}: NEW BEST {rec['bytes_mb']} MB kl={rec['kl']:.5f}")
        (OUT / f"best_t{int(args.tier_mb)}.json").write_text(json.dumps(best, indent=1))
        print(f"gen {gen} done; best {best['bytes_mb']} MB kl={best['kl']:.5f}")
    print("search complete")


if __name__ == "__main__":
    main()
