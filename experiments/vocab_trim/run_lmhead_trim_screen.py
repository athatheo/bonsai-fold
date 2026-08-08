"""A4 screen: output-side lm_head row trimming, measured analytically.

Row deletion does not change surviving logits, so the trimmed model's
distribution is exactly the reference distribution renormalized over the
keep-set. One reference forward per probe therefore yields every metric:
  - dropped probability mass per position, E[P_ref(dropped)] and max
  - teacher-forced NLL inflation: -log(kept_mass) at each frozen token
    (all frozen tokens are in the keep-set by construction)
  - greedy-decode divergence: P(argmax in dropped set)

Keep-set = every token observed in the frozen corpus (vocab census) union
chat-template/thinking specials, padded with lowest-index unused rows to a
multiple of 128 (g128-clean row count).

DO NOT run casually: loads the 27B (~10 min for 100 on-policy probes).

Usage: uv run python experiments/vocab_trim/run_lmhead_trim_screen.py \
    [--probes experiments/calibration/probes_onpolicy.json]
"""
import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from bonsaifold import bonsai_prompt
from bonsaifold.loader import load_bonsai
from bonsaifold.probes import load_probe_set, probe_fingerprint

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).parent
VOCAB = 248320
CHUNK = 256


def build_keep_list(tokenizer):
    counts = json.loads((REPO / "experiments/vocab_census/vocab_census.json").read_text())
    # census stores summary only; recount used ids from the frozen sets
    used = set()
    for f in ["calibration_set", "probes_onpolicy", "probes_offpolicy"]:
        data = json.loads((REPO / f"experiments/calibration/{f}.json").read_text())
        for it in data["items"]:
            used.update(it.get("token_ids") or [])
    bench = json.loads((REPO / "experiments/minibench/minibench_items.json").read_text())
    for it in bench["items"]:
        used.update(tokenizer.encode(it["prompt"]))
    used.update(bonsai_prompt(tokenizer, [{"role": "user", "content": "x"}]))
    for s in ("</think>", "<|im_end|>", "<|endoftext|>"):
        used.update(tokenizer.encode(s))
    keep = sorted(u for u in used if u < VOCAB)
    pad = (-len(keep)) % 128
    unused_iter = (i for i in range(VOCAB) if i not in used)
    keep.extend(next(unused_iter) for _ in range(pad))
    return sorted(keep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probes", default=str(REPO / "experiments/calibration/probes_onpolicy.json"))
    args = ap.parse_args()
    probes = load_probe_set(args.probes)

    model, tokenizer = load_bonsai(REPO / "models/Bonsai-27B-mlx-1bit")
    keep = build_keep_list(tokenizer)
    keep_mask = np.zeros(VOCAB, dtype=bool)
    keep_mask[keep] = True
    keep_mx = mx.array(keep_mask)

    rows_deleted = VOCAB - len(keep)
    saved_mb = rows_deleted * (5120 * 1.125 / 8 + 2 * 2 * 40) / 1e6  # packed row + scales/biases

    stats = []
    for item in probes["items"]:
        ids = item["token_ids"]
        logits = model(mx.array([ids]))
        n = logits.shape[1]
        dm_sum = 0.0; dm_max = 0.0; nll_sum = 0.0; argmax_dropped = 0
        for s in range(0, n, CHUNK):
            x = logits[:, s : s + CHUNK].astype(mx.float32)
            p = mx.softmax(x, axis=-1)
            dropped = (p * (1 - keep_mx)).sum(-1)  # [1, T]
            am = mx.argmax(x, axis=-1)
            mx.eval(dropped, am)
            d = np.array(dropped, copy=False)[0]
            dm_sum += float(d.sum()); dm_max = max(dm_max, float(d.max()))
            nll_sum += float(-np.log1p(-np.clip(d, 0, 1 - 1e-9)).sum())
            argmax_dropped += int((~keep_mask[np.array(am, copy=False)[0]]).sum())
        stats.append({
            "id": item["id"], "tokens": n,
            "mean_dropped_mass": dm_sum / n,
            "max_dropped_mass": dm_max,
            "mean_nll_inflation_nats": nll_sum / n,
            "argmax_dropped_frac": argmax_dropped / n,
        })
        print(f"{item['id']}: dropped-mass mean {dm_sum/n:.2e} max {dm_max:.2e} "
              f"nll+{nll_sum/n:.2e} argmax-div {argmax_dropped/n:.2e}")

    agg = lambda k: float(np.mean([s[k] for s in stats]))
    report = {
        "probes": probes.get("meta", {}), "probe_fingerprint": probe_fingerprint(probes),
        "keep_rows": len(keep), "rows_deleted": rows_deleted,
        "saved_mb": round(saved_mb, 1),
        "mean_dropped_mass": agg("mean_dropped_mass"),
        "max_dropped_mass": max(s["max_dropped_mass"] for s in stats),
        "mean_nll_inflation_nats": agg("mean_nll_inflation_nats"),
        "mean_argmax_dropped_frac": agg("argmax_dropped_frac"),
        "items": stats,
    }
    name = Path(args.probes).stem
    (OUT / f"lmhead_trim_screen_{name}.json").write_text(json.dumps(report, indent=1))
    (OUT / "keep_list.json").write_text(json.dumps({"keep": keep}))
    print(json.dumps({k: v for k, v in report.items() if k != "items"}, indent=1))


if __name__ == "__main__":
    main()
