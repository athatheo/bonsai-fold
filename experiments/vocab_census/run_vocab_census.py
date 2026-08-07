"""Vocabulary-usage census: how much of the 248,320-row embedding/lm_head
would trimming touch, measured on every frozen artifact we have?

Motivation: the tail (embed + lm_head) is ~0.36 GB of 1-bit rows; rows for
tokens that never occur in our text distribution are dead weight that can be
DELETED (format-native, no value changes) — the width analogue of block
dropping, applied to the vocab axis. This census only measures usage; the
trim operator comes later if the numbers justify it.

Counts token-id occurrences across: calibration set, on/off-policy probes
(frozen token ids), mini-bench prompts + all recorded model generations we
have (mini-bench result files store only correctness, so generations are
approximated by the probe completions — noted in output), plus every token
the chat template itself emits. Reports coverage curves and projected GB
savings at several keep-K cuts.

Usage: uv run python experiments/vocab_census/run_vocab_census.py
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).parent
PACK = REPO / "models/Bonsai-27B-mlx-1bit"
VOCAB = 248320
HIDDEN = 5120
BPW = 1.125  # 1-bit g128 effective bits/weight


def main():
    from mlx_lm.utils import load_tokenizer

    tok = load_tokenizer(PACK)
    counts = Counter()
    sources = {}

    def add(ids, source):
        counts.update(ids)
        sources[source] = sources.get(source, 0) + len(ids)

    for f in ["calibration_set", "probes_onpolicy", "probes_offpolicy"]:
        data = json.loads((REPO / f"experiments/calibration/{f}.json").read_text())
        for it in data["items"]:
            if it.get("token_ids"):
                add(it["token_ids"], f)

    bench = json.loads((REPO / "experiments/minibench/minibench_items.json").read_text())
    for it in bench["items"]:
        add(tok.encode(it["prompt"]), "minibench_prompts")

    # chat-template scaffolding tokens (roles, think tags, eos) must survive
    # any trim regardless of corpus counts
    from bonsaifold import bonsai_prompt

    template_ids = set(bonsai_prompt(tok, [{"role": "user", "content": "x"}]))
    template_ids.update(
        tok.encode("</think>") + tok.encode("<|im_end|>") + tok.encode("<|endoftext|>")
    )

    used = len(counts)
    total = sum(counts.values())
    freq = np.array(sorted(counts.values(), reverse=True), dtype=np.int64)
    cum = np.cumsum(freq) / total
    row_bytes = 2 * HIDDEN * BPW / 8  # embed + lm_head row pair, packed+scales

    def keep_stats(k):
        cov = float(cum[min(k, used) - 1]) if used else 0.0
        return {
            "keep_rows": k,
            "corpus_coverage": round(cov, 6),
            "deleted_rows": VOCAB - k,
            "saved_mb": round((VOCAB - k) * row_bytes / 1e6, 1),
        }

    report = {
        "vocab_size": VOCAB,
        "distinct_tokens_used": used,
        "total_tokens_counted": int(total),
        "tokens_by_source": sources,
        "template_special_tokens": len(template_ids),
        "unused_rows": VOCAB - used,
        "unused_rows_saved_mb": round((VOCAB - used) * row_bytes / 1e6, 1),
        "keep_cuts": [keep_stats(k) for k in (200_000, 150_000, 100_000, 64_000, used)],
        "caveats": [
            "usage measured on our frozen English/math/code distribution only",
            "model GENERATIONS beyond the frozen completions are not counted",
            "input-side unseen tokens break hard on trimmed embed; output-side "
            "trim only restricts emissions — trim design must treat the two "
            "sides separately",
        ],
    }
    (OUT / "vocab_census.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "keep_cuts"}, indent=1))
    for c in report["keep_cuts"]:
        print(c)


if __name__ == "__main__":
    main()
