"""Freeze the mini-bench item sets (no model, network downloads only).

Tasks (plan §4): GSM8K-100, MATH500-100 (disjoint from the KL calibration
and probe draws), IFEval-100 (only items whose instruction types our
verifier subset implements), MMLU-Redux-200 (error_type == "ok" items).
Writes minibench_items.json: {"meta", "items": [{id, task, prompt, gold,
...task fields}]}. Prompts are TEXT; the runner applies the chat template.

Usage: uv run python experiments/minibench/build_minibench.py
"""
import json
from pathlib import Path

import numpy as np

from bonsaifold.minibench.ifeval import item_supported

OUTDIR = Path(__file__).parent
SEED = 20260715  # same master seed as the calibration builder
CALIB_PROBE_DRAW = 124  # MATH-500 indices consumed by calibration+probes


def main():
    from datasets import get_dataset_config_names, load_dataset

    rng = np.random.default_rng(SEED)
    items = []

    # ---- GSM8K-100 ----
    gsm = load_dataset("openai/gsm8k", "main", split="test")
    for k, i in enumerate(sorted(rng.choice(len(gsm), size=100, replace=False))):
        row = gsm[int(i)]
        items.append(
            {
                "id": f"gsm8k-{k:03d}",
                "task": "gsm8k",
                "prompt": row["question"]
                + "\n\nPlease reason step by step, and put your final numeric "
                "answer within \\boxed{}.",
                "gold": row["answer"],  # runner uses gsm8k_gold()
            }
        )

    # ---- MATH500-100, disjoint from the KL calibration/probe draws ----
    math500 = load_dataset("HuggingFaceH4/MATH-500", split="test")
    # reproduce the calibration builder's permutation to know what it used
    used = set(np.random.default_rng(SEED).permutation(len(math500))[:CALIB_PROBE_DRAW])
    remaining = [i for i in range(len(math500)) if i not in used]
    for k, i in enumerate(sorted(rng.choice(remaining, size=100, replace=False))):
        row = math500[int(i)]
        items.append(
            {
                "id": f"math500-{k:03d}",
                "task": "math500",
                "prompt": row["problem"]
                + "\n\nPut your final answer within \\boxed{}.",
                "gold": row["answer"],
            }
        )

    # ---- IFEval-100, verifiable-subset items only ----
    ifeval = load_dataset("google/IFEval", split="train")
    supported = [
        i for i in range(len(ifeval)) if item_supported(ifeval[i]["instruction_id_list"])
    ]
    picks = sorted(rng.choice(supported, size=min(100, len(supported)), replace=False))
    for k, i in enumerate(picks):
        row = ifeval[int(i)]
        items.append(
            {
                "id": f"ifeval-{k:03d}",
                "task": "ifeval",
                "prompt": row["prompt"],
                "instruction_id_list": row["instruction_id_list"],
                "kwargs": row["kwargs"],
            }
        )

    # ---- MMLU-Redux-200 across subjects, clean items only ----
    pool = []
    for subject in get_dataset_config_names("edinburgh-dawg/mmlu-redux-2.0"):
        ds = load_dataset("edinburgh-dawg/mmlu-redux-2.0", subject, split="test")
        for i in range(len(ds)):
            if ds[i].get("error_type") == "ok":
                pool.append((subject, i, ds[i]))
    letters = "ABCD"
    for k, pi in enumerate(sorted(rng.choice(len(pool), size=200, replace=False))):
        subject, i, row = pool[int(pi)]
        choices = "\n".join(f"({letters[j]}) {c}" for j, c in enumerate(row["choices"]))
        items.append(
            {
                "id": f"mmlu-{k:03d}",
                "task": "mmlu",
                "prompt": f"{row['question']}\n\n{choices}\n\n"
                "Answer with the letter of the correct choice.",
                "gold": letters[int(row["answer"])],
                "source": f"{subject}[{i}]",
            }
        )

    meta = {
        "seed": SEED,
        "counts": {t: sum(1 for it in items if it["task"] == t) for t in
                   ("gsm8k", "math500", "ifeval", "mmlu")},
        "ifeval_supported_pool": len(supported),
    }
    out = OUTDIR / "minibench_items.json"
    out.write_text(json.dumps({"meta": meta, "items": items}))
    print(f"wrote {out}: {meta['counts']} (ifeval pool {len(supported)})")


if __name__ == "__main__":
    main()
