"""A5 input-side risk census: how often would FRESH text hit a deleted
embedding row?

The lm_head trim (A4) is output-side and graceful; trimming the EMBED side
breaks hard on any input token outside the keep-set. This census tokenizes
held-out corpora the keep-set was NOT built from (fresh wikitext slices,
fresh dolly rows, vendor code files never sampled, GSM8K train split) and
measures keep-set coverage per corpus. CPU only.

Usage: uv run python experiments/vocab_trim/run_embed_coverage.py
"""
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).parent


def main():
    from datasets import load_dataset
    from mlx_lm.utils import load_tokenizer

    tok = load_tokenizer(REPO / "models/Bonsai-27B-mlx-1bit")
    keep = set(json.loads((OUT / "keep_list_v2.json").read_text())["keep"])
    print(f"keep-set: {len(keep)} rows")

    corpora = {}
    wiki = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
    corpora["wikitext_heldout"] = "\n".join(wiki[60_000 + i]["text"] for i in range(8_000))
    dolly = load_dataset("databricks/databricks-dolly-15k", split="train")
    corpora["dolly_heldout"] = "\n".join(
        dolly[i]["instruction"] + "\n" + dolly[i]["response"] for i in range(2000, 3500)
    )
    gsm = load_dataset("openai/gsm8k", "main", split="train")
    corpora["gsm8k_train"] = "\n".join(gsm[i]["question"] + gsm[i]["answer"] for i in range(1500))
    code_files = sorted((REPO / "vendor/mlx/python").rglob("*.py"))[:60]
    corpora["code_heldout"] = "\n".join(f.read_text(errors="replace") for f in code_files)

    report = {"keep_rows": len(keep), "corpora": {}}
    for name, text in corpora.items():
        ids = tok.encode(text)
        oov = [t for t in ids if t not in keep]
        distinct_oov = sorted(set(oov))
        report["corpora"][name] = {
            "tokens": len(ids),
            "oov_tokens": len(oov),
            "oov_rate": round(len(oov) / len(ids), 8),
            "distinct_oov": len(distinct_oov),
            "example_oov": [tok.decode([t]) for t in distinct_oov[:10]],
        }
        print(f"{name:18s} {len(ids):>9d} tokens, OOV {len(oov)} "
              f"({len(oov)/len(ids):.2e}), distinct {len(distinct_oov)}")
    (OUT / "embed_coverage.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
