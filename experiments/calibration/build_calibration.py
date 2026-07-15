"""Build the frozen calibration and KL-probe sets.

Two stages, because only one needs the model:

  --stage prompts  (no inference; run anytime)
    - calibration_set.json: 64 sequences for block-influence mapping —
      24 reasoning (MATH-500 prompts; completions filled by stage 2),
      16 long-context prose (wikitext-103, ~2K tokens each),
      12 code (files from vendor/mlx, ~2K tokens each),
      12 chat (dolly-15k, chat-templated).
    - probes_offpolicy.json: 100 wikitext-103 chunks (~2-4K tokens),
      disjoint from the calibration prose.
    - probes_onpolicy_prompts.json: 100 MATH-500 prompts (disjoint from the
      24 calibration ones), awaiting stage 2 completions.

  --stage completions  (LOADS THE 27B AND GENERATES — ~25 min; do not run
    while the machine is needed elsewhere)
    - generates thinking-mode completions (mandated sampling, budget
      --max-tokens, default 1024) from the unfolded pack for the reasoning
      calibration items and the on-policy probes, freezing the token ids.
      Deterministic per --seed via mx.random.seed.

Everything is committed: token ids of public data plus, after stage 2,
model-generated tokens. Selection is seeded and recorded in meta.
"""
import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUTDIR = Path(__file__).parent
DEFAULT_PACK = REPO / "models/Bonsai-27B-mlx-1bit"
SEED = 20260715
TARGET_TOKENS = 2048
N_CALIB_REASONING, N_CALIB_PROSE, N_CALIB_CODE, N_CALIB_CHAT = 24, 16, 12, 12
N_PROBE_ONPOLICY, N_PROBE_OFFPOLICY = 100, 100


def get_tokenizer(pack):
    from mlx_lm.utils import load_tokenizer

    return load_tokenizer(Path(pack))


def build_prompts(pack):
    from datasets import load_dataset

    from bonsaifold import bonsai_prompt

    tok = get_tokenizer(pack)
    rng = np.random.default_rng(SEED)

    # ---- MATH-500: disjoint calibration/probe prompt draws ----
    math500 = load_dataset("HuggingFaceH4/MATH-500", split="test")
    idx = rng.permutation(len(math500))
    calib_idx = idx[:N_CALIB_REASONING]
    probe_idx = idx[N_CALIB_REASONING : N_CALIB_REASONING + N_PROBE_ONPOLICY]

    def math_item(i, tag):
        problem = math500[int(i)]["problem"]
        return {
            "id": f"{tag}{int(i):03d}",
            "category": "reasoning",
            "source": f"MATH-500[{int(i)}]",
            "prompt_token_ids": bonsai_prompt(tok, [{"role": "user", "content": problem}]),
            "token_ids": None,  # filled by --stage completions
        }

    calib_items = [math_item(i, "math-calib-") for i in calib_idx]
    onpolicy_items = [math_item(i, "math-probe-") for i in probe_idx]

    # ---- wikitext-103: prose calibration + off-policy probes, disjoint ----
    wiki = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
    # ~40K rows ≈ 4M tokens: two orders of magnitude more than the 116
    # disjoint chunks need, cheap to tokenize once
    text = "\n".join(wiki[i]["text"] for i in range(40_000))
    all_ids = tok.encode(text)
    n_chunks = N_CALIB_PROSE + N_PROBE_OFFPOLICY
    lengths = rng.integers(2048, 4097, size=n_chunks)
    starts = rng.choice(len(all_ids) - 4200, size=n_chunks, replace=False)
    chunks = [
        all_ids[int(s) : int(s) + int(l)] for s, l in zip(sorted(starts), lengths)
    ]
    rng.shuffle(chunks)
    for k in range(N_CALIB_PROSE):
        calib_items.append(
            {
                "id": f"prose-{k:03d}",
                "category": "prose",
                "source": "wikitext-103-raw-v1",
                "token_ids": chunks[k][:TARGET_TOKENS],
            }
        )
    offpolicy_items = [
        {
            "id": f"wiki-probe-{k:03d}",
            "category": "prose",
            "source": "wikitext-103-raw-v1",
            "token_ids": chunks[N_CALIB_PROSE + k],
        }
        for k in range(N_PROBE_OFFPOLICY)
    ]

    # ---- code: local vendor/mlx sources (MIT), no download ----
    exts = ("*.py", "*.cpp", "*.h", "*.metal")
    files = sorted(
        f
        for pat in exts
        for f in (REPO / "vendor/mlx/mlx").rglob(pat)
        if f.stat().st_size > 8000
    )
    picks = rng.choice(len(files), size=N_CALIB_CODE, replace=False)
    for k, fi in enumerate(sorted(picks)):
        f = files[int(fi)]
        ids = tok.encode(f.read_text(errors="replace"))[:TARGET_TOKENS]
        calib_items.append(
            {
                "id": f"code-{k:03d}",
                "category": "code",
                "source": str(f.relative_to(REPO)),
                "token_ids": ids,
            }
        )

    # ---- chat: dolly-15k, longest-response items, chat-templated ----
    dolly = load_dataset("databricks/databricks-dolly-15k", split="train")
    order = sorted(
        range(len(dolly)), key=lambda i: -len(dolly[i]["response"])
    )[:400]
    picks = rng.choice(order, size=N_CALIB_CHAT, replace=False)
    for k, di in enumerate(sorted(picks)):
        row = dolly[int(di)]
        content = row["instruction"]
        if row["context"]:
            content = row["context"] + "\n\n" + content
        ids = bonsai_prompt(tok, [{"role": "user", "content": content}])
        ids = list(ids) + tok.encode(row["response"])
        calib_items.append(
            {
                "id": f"chat-{k:03d}",
                "category": "chat",
                "source": f"dolly-15k[{int(di)}]",
                "token_ids": ids[: TARGET_TOKENS],
            }
        )

    meta = {"seed": SEED, "tokenizer_pack": str(pack), "target_tokens": TARGET_TOKENS}
    (OUTDIR / "calibration_set.json").write_text(
        json.dumps({"meta": {**meta, "note": "reasoning token_ids pending --stage completions"}, "items": calib_items})
    )
    (OUTDIR / "probes_offpolicy.json").write_text(
        json.dumps({"meta": meta, "items": offpolicy_items})
    )
    (OUTDIR / "probes_onpolicy_prompts.json").write_text(
        json.dumps({"meta": meta, "items": onpolicy_items})
    )
    counts = {}
    for it in calib_items:
        counts[it["category"]] = counts.get(it["category"], 0) + 1
    print(f"calibration: {counts}; on-policy prompts: {len(onpolicy_items)}; "
          f"off-policy probes: {len(offpolicy_items)}")


def build_completions(pack, max_tokens, seed):
    import mlx.core as mx
    from mlx_lm import stream_generate

    from bonsaifold import make_bonsai_sampler
    from bonsaifold.loader import load_bonsai

    model, tok = load_bonsai(pack)

    def complete(prompt_ids, gen_seed):
        mx.random.seed(gen_seed)
        out_ids = []
        for resp in stream_generate(
            model, tok, prompt_ids, max_tokens=max_tokens, sampler=make_bonsai_sampler()
        ):
            out_ids.append(resp.token)
        return out_ids

    for fname, out_name in [
        ("calibration_set.json", "calibration_set.json"),
        ("probes_onpolicy_prompts.json", "probes_onpolicy.json"),
    ]:
        data = json.loads((OUTDIR / fname).read_text())
        for k, item in enumerate(data["items"]):
            if item.get("token_ids") is not None or "prompt_token_ids" not in item:
                continue
            gen = complete(item["prompt_token_ids"], seed + k)
            item["token_ids"] = list(item["prompt_token_ids"]) + gen
            (OUTDIR / out_name).write_text(json.dumps(data))  # checkpoint
            print(f"{item['id']}: +{len(gen)} tokens")
        data["meta"]["completion_pack"] = str(pack)
        data["meta"]["completion_max_tokens"] = max_tokens
        data["meta"]["completion_seed"] = seed
        (OUTDIR / out_name).write_text(json.dumps(data))
        print(f"wrote {OUTDIR / out_name}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["prompts", "completions"], required=True)
    ap.add_argument("--pack", default=str(DEFAULT_PACK))
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    if args.stage == "prompts":
        build_prompts(args.pack)
    else:
        build_completions(args.pack, args.max_tokens, args.seed)
