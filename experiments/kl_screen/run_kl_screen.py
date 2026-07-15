"""Phase 1/2: KL screen — teacher-forced forward KL between two packs.

For each frozen probe sequence, both models are teacher-forced over the same
tokens and the mean per-token forward KL(reference || candidate) over the
full vocabulary is reported (nats). Mirrors the whitepaper's KV-tolerance
methodology, applied to structural perturbation. Logits are compared in
position chunks so the [seq, 248320] float32 tensors never materialize
whole. Checkpoints per item — safe to interrupt and resume.

DO NOT run casually: loads both packs (~10 GB together for two 1-bit packs).
Expected runtime: ~15 min per candidate for the standard 100-item probe set.

Usage:
  uv run python experiments/kl_screen/run_kl_screen.py \
      --reference models/Bonsai-27B-mlx-1bit \
      --candidate models/folded/drop_b17 \
      --probes experiments/calibration/probes_onpolicy.json \
      --out experiments/kl_screen/results/drop_b17_onpolicy.json
"""
import argparse
import json
from pathlib import Path

import mlx.core as mx

from bonsaifold.loader import load_bonsai

CHUNK = 256  # positions per logit chunk


def sequence_kl(ref_model, cand_model, token_ids):
    """Mean per-token forward KL(ref || cand) in nats, full vocabulary."""
    tokens = mx.array([token_ids])
    ref_logits = ref_model(tokens)
    cand_logits = cand_model(tokens)
    n = ref_logits.shape[1]
    total = mx.array(0.0)
    for s in range(0, n, CHUNK):
        r = ref_logits[:, s : s + CHUNK].astype(mx.float32)
        c = cand_logits[:, s : s + CHUNK].astype(mx.float32)
        r_logp = r - mx.logsumexp(r, axis=-1, keepdims=True)
        c_logp = c - mx.logsumexp(c, axis=-1, keepdims=True)
        kl = (mx.exp(r_logp) * (r_logp - c_logp)).sum(-1)
        total = total + kl.sum()
        mx.eval(total)
    return float(total) / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--probes", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if out.exists():  # resume
        done = {r["id"]: r for r in json.loads(out.read_text())["items"]}

    probes = json.loads(Path(args.probes).read_text())
    ref_model, _ = load_bonsai(args.reference)
    cand_model, _ = load_bonsai(args.candidate)

    items = []
    for item in probes["items"]:
        if item["id"] in done:
            items.append(done[item["id"]])
            continue
        kl = sequence_kl(ref_model, cand_model, item["token_ids"])
        rec = {"id": item["id"], "tokens": len(item["token_ids"]), "kl_nats": kl}
        items.append(rec)
        n_tok = sum(r["tokens"] for r in items)
        mean = sum(r["kl_nats"] * r["tokens"] for r in items) / n_tok
        out.write_text(
            json.dumps(
                {
                    "reference": args.reference,
                    "candidate": args.candidate,
                    "probes": probes.get("meta", {}),
                    "mean_kl_nats": mean,
                    "items": items,
                },
                indent=1,
            )
        )
        print(f"{item['id']}: {kl:.5f} nats (running mean {mean:.5f})")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
