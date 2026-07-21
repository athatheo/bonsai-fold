"""Phase 1/2: KL screen — teacher-forced forward KL vs the reference model.

Sweep-native: candidates are zero-copy layer-subset VIEWS of the loaded
reference (bonsaifold.loader.drop_view), so screening N drop candidates
loads one model, writes no packs, and computes each probe's reference
log-probs once, amortized across all candidates. A written pack can also
be screened (--candidate-pack) for final verification of shipped folds.

Per probe sequence, both stacks are teacher-forced over the same frozen
tokens; the metric is mean per-token forward KL(reference || candidate)
over the full vocabulary (nats) — mlx's kl_div_loss on log-probs, which
mirrors the whitepaper's KV-tolerance methodology. Chunked in positions so
float32 softmax temporaries stay bounded. Checkpoints per item with a
probe-set fingerprint; resume refuses a checkpoint from different probes.

DO NOT run casually: loads the 27B reference (~5 GB; +5 GB per
--candidate-pack). ~15 min per candidate for the 100-item probe set.

Usage (candidate sweep, no packs written):
  uv run python experiments/kl_screen/run_kl_screen.py \
      --reference models/Bonsai-27B-mlx-1bit \
      --probes experiments/calibration/probes_onpolicy.json \
      --drop 17 --drop 17,21 --drop 9,17,21,33 \
      --out experiments/kl_screen/results/sweep1.json
"""
import argparse
import json
import os
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from bonsaifold.loader import drop_view, load_bonsai
from bonsaifold.probes import load_probe_set, probe_fingerprint

CHUNK = 256  # positions per float32 softmax chunk


def chunked_logprobs(logits):
    for s in range(0, logits.shape[1], CHUNK):
        x = logits[:, s : s + CHUNK].astype(mx.float32)
        yield nn.log_softmax(x, axis=-1)


def sequence_kl(ref_logits, cand_logits):
    """Mean per-token forward KL(ref || cand) in nats, full vocabulary."""
    total = 0.0
    for r_logp, c_logp in zip(chunked_logprobs(ref_logits), chunked_logprobs(cand_logits)):
        kl = nn.losses.kl_div_loss(c_logp, r_logp, axis=-1)
        total += float(kl.sum())
    return total / ref_logits.shape[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True)
    ap.add_argument("--probes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--drop",
        action="append",
        default=[],
        help="comma-separated block indices; repeat per candidate",
    )
    ap.add_argument("--candidate-pack", action="append", default=[])
    args = ap.parse_args()
    if not args.drop and not args.candidate_pack:
        ap.error("give at least one --drop or --candidate-pack")

    probes = load_probe_set(args.probes)  # fails fast BEFORE model load
    fingerprint = probe_fingerprint(probes)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {}  # (candidate, item id) -> record
    prev_results = {}  # candidates from earlier runs, preserved in the output
    if out.exists():
        prev = json.loads(out.read_text())
        if "candidates" not in prev:
            raise SystemExit(
                f"{out} is in the old single-candidate format; move it aside"
            )
        if prev.get("probe_fingerprint") != fingerprint:
            raise SystemExit(
                f"{out} was produced with different probes "
                f"({prev.get('probe_fingerprint')} != {fingerprint}); "
                "move it aside or pick a new --out"
            )
        for c in prev["candidates"]:
            prev_results[c["name"]] = c["items"]
            for r in c["items"]:
                done[(c["name"], r["id"])] = r

    # candidate names are derivable without the model: skip the ~5GB load
    # (and mlx startup) when every (candidate, item) pair is checkpointed
    names = [f"drop[{spec}]" for spec in args.drop] + list(args.candidate_pack)
    if all((n, item["id"]) in done for n in names for item in probes["items"]):
        print(f"{out} already complete")
        return

    ref_model, _ = load_bonsai(args.reference)
    candidates = [
        (f"drop[{spec}]", drop_view(ref_model, [int(i) for i in spec.split(",")]))
        for spec in args.drop
    ]
    for pack in args.candidate_pack:
        candidates.append((pack, load_bonsai(pack)[0]))

    # earlier runs' candidates ride along untouched, so extending a sweep
    # into the same --out never destroys prior results
    results = {**prev_results, **{name: [] for name, _ in candidates}}

    def write():
        # atomic replace: a battery death mid-write must never leave a
        # truncated checkpoint that aborts the resume it exists to enable
        tmp = out.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(
                {
                    "reference": args.reference,
                    "probes": probes.get("meta", {}),
                    "probe_fingerprint": fingerprint,
                    "candidates": [
                        {
                            "name": name,
                            "mean_kl_nats": (
                                sum(r["kl_nats"] * r["tokens"] for r in items)
                                / max(sum(r["tokens"] for r in items), 1)
                            ),
                            "items": items,
                        }
                        for name, items in results.items()
                    ],
                },
                indent=1,
            )
        )
        os.replace(tmp, out)

    for item in probes["items"]:
        pending = [
            (name, cand)
            for name, cand in candidates
            if (name, item["id"]) not in done
        ]
        for name, _ in candidates:
            if (name, item["id"]) in done:
                results[name].append(done[(name, item["id"])])
        if not pending:
            continue
        tokens = mx.array([item["token_ids"]])
        ref_logits = ref_model(tokens)
        mx.eval(ref_logits)  # computed once, reused by every candidate
        for name, cand in pending:
            kl = sequence_kl(ref_logits, cand(tokens))
            rec = {"id": item["id"], "tokens": len(item["token_ids"]), "kl_nats": kl}
            results[name].append(rec)
            print(f"{name}  {item['id']}: {kl:.5f} nats")
        write()
    write()
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
