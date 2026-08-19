"""Q3: probe sets for the DENSE Bonsai-8B (mirrors calibration/build_calibration).

  --stage offpolicy  (CPU: tokenize wikitext-103 chunks with the 8B
                      tokenizer; disjoint doc selection, seeded)
  --stage onpolicy   (LOADS THE 8B AND GENERATES its own MATH-500 thinking
                      traces — GPU; run only when the 27B benches are idle)

Same regime definitions as the 27B program so cross-model comparisons are
regime-matched; token ids are 8B-tokenizer-specific (vocab 151669).
"""
import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).parent
PACK = REPO / "models/Bonsai-8B-mlx-1bit"
SEED = 20260819
N_PROBES = 100
TARGET_TOKENS = (2000, 4000)


def build_offpolicy():
    from datasets import load_dataset
    from mlx_lm.utils import load_tokenizer

    tok = load_tokenizer(PACK)
    wiki = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
    import numpy as np

    rng = np.random.default_rng(SEED)
    # assemble contiguous doc chunks until the token window fits the target
    items, used, i = [], set(), 0
    order = rng.permutation(len(wiki))
    for start in order:
        if len(items) >= N_PROBES:
            break
        start = int(start)
        if start in used:
            continue
        text, j = "", start
        while j < len(wiki) and len(text) < TARGET_TOKENS[1] * 6:
            text += wiki[j]["text"]
            j += 1
        used.update(range(start, j))
        ids = tok.encode(text)[: TARGET_TOKENS[1]]
        if len(ids) < TARGET_TOKENS[0]:
            continue
        items.append({"id": f"wiki8b-{len(items):03d}", "source_row": start,
                      "token_ids": ids})
        i += 1
    out = OUT / "probes_offpolicy_8b.json"
    out.write_text(json.dumps({
        "meta": {"seed": SEED, "tokenizer_pack": str(PACK),
                 "target_tokens": TARGET_TOKENS, "n": len(items)},
        "items": items,
    }))
    print(f"wrote {out}: {len(items)} probes, "
          f"{sum(len(x['token_ids']) for x in items)} tokens")


def build_onpolicy(max_tokens):
    import mlx.core as mx
    from datasets import load_dataset
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler
    from mlx_lm.utils import load

    model, tok = load(str(PACK))  # dense qwen3: stock loader is correct
    math = load_dataset("HuggingFaceH4/MATH-500", split="test")
    import numpy as np

    rng = np.random.default_rng(SEED)
    picks = [int(i) for i in rng.permutation(len(math))[:N_PROBES]]
    out = OUT / "probes_onpolicy_8b.json"
    done = json.loads(out.read_text()) if out.exists() else {
        "meta": {"seed": SEED, "pack": str(PACK), "max_tokens": max_tokens,
                 "sampling": {"temp": 0.7, "top_p": 0.95, "top_k": 20}},
        "items": [],
    }
    have = {x["id"] for x in done["items"]}
    sampler = make_sampler(temp=0.7, top_p=0.95, top_k=20)
    for k, row in enumerate(picks):
        pid = f"math8b-{k:03d}"
        if pid in have:
            continue
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": math[row]["problem"]}],
            add_generation_prompt=True,
        )
        mx.random.seed(SEED + k)
        text = ""
        for r in stream_generate(model, tok, prompt, max_tokens=max_tokens,
                                 sampler=sampler):
            text += r.text
        ids = list(prompt) + tok.encode(text, add_special_tokens=False)
        done["items"].append({"id": pid, "source_row": row, "token_ids": ids})
        out.write_text(json.dumps(done))
        print(f"{pid}: {len(ids)} tokens", flush=True)
    print(f"onpolicy complete: {len(done['items'])} probes")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["offpolicy", "onpolicy"], required=True)
    ap.add_argument("--max-tokens", type=int, default=2048)
    args = ap.parse_args()
    if args.stage == "offpolicy":
        build_offpolicy()
    else:
        build_onpolicy(args.max_tokens)


if __name__ == "__main__":
    main()
