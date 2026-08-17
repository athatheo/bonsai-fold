"""Run the frozen mini-bench on a pack or on a drop-candidate view.

Thinking mode with the mandated sampling; rule-first scoring via
bonsaifold.minibench (answers after the thinking block only). Checkpoints
per item; resume by id. One JSON per config under results/.

DO NOT run casually: loads the 27B and generates up to --max-tokens per
item (~500 items x ~1.5K tokens avg ~= 3h per config at ~66 tok/s).

Usage:
  uv run python experiments/minibench/run_minibench.py \
      --pack models/Bonsai-27B-mlx-1bit [--drop 12,8] \
      --out experiments/minibench/results/reference.json

External (non-Bonsai) comparator packs load via stock mlx-lm instead of the
layer_types shim (--stock-loader); sampling, thinking mode, budget, items,
and scoring stay identical so the comparison is protocol-matched:
  uv run python experiments/minibench/run_minibench.py \
      --pack models/<external-mlx-pack> --stock-loader \
      --out experiments/minibench/results/external_<name>.json
"""
import argparse
import json
import os
from pathlib import Path

import mlx.core as mx
from mlx_lm import stream_generate

from bonsaifold import bonsai_prompt, make_bonsai_sampler
from bonsaifold.loader import drop_view, load_bonsai
from bonsaifold.minibench import answers, ifeval

ITEMS = Path(__file__).parent / "minibench_items.json"


def score(item, response):
    # a response that never closes its thinking block gave no answer —
    # scoring the raw chain-of-thought would fish stray numbers out of it
    if "</think>" not in response:
        return False
    body = answers.strip_thinking(response)
    try:
        if item["task"] == "gsm8k":
            pred = answers.extract_boxed(body)  # falls back to last number
            return answers.math_equal(pred, answers.gsm8k_gold(item["gold"])) if pred else False
        if item["task"] == "math500":
            pred = answers.extract_boxed(body)
            return answers.math_equal(pred, item["gold"]) if pred else False
        if item["task"] == "mmlu":
            return answers.extract_mmlu_choice(body) == item["gold"]
        if item["task"] == "ifeval":
            return all(
                ifeval.verify(iid, kw or {}, body)
                for iid, kw in zip(item["instruction_id_list"], item["kwargs"])
            )
        raise ValueError(f"unknown task {item['task']}")
    except ValueError:
        # malformed item/gold must not wedge the resume loop; scored wrong
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--drop", default=None, help="comma-separated block indices")
    ap.add_argument(
        "--drop-sub",
        default=None,
        help="mixed drop spec, tokens b/a/m<idx> joined by '+' (screen grammar)",
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=16384)  # short tier
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--stock-loader",
        action="store_true",
        help="load via stock mlx_lm.load (external comparator packs)",
    )
    args = ap.parse_args()
    if args.stock_loader and args.drop:
        ap.error("--drop requires the Bonsai layer_types loader")

    data = json.loads(ITEMS.read_text())
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if out.exists():
        prev = json.loads(out.read_text())
        prev_config = {k: prev.get(k) for k in ("pack", "drop", "drop_sub", "max_tokens", "seed")}
        prev_config["stock_loader"] = bool(prev.get("stock_loader"))
        now_config = {
            "pack": args.pack,
            "drop": args.drop,
            "drop_sub": args.drop_sub,
            "max_tokens": args.max_tokens,
            "seed": args.seed,
            "stock_loader": args.stock_loader,
        }
        if prev_config != now_config:
            raise SystemExit(
                f"{out} was produced with different config "
                f"({prev_config} != {now_config}); move it aside or pick a new --out"
            )
        done = {r["id"]: r for r in prev["items"]}
    if all(item["id"] in done for item in data["items"]):
        print(f"{out} already complete")  # skip the multi-minute model load
        return

    if args.stock_loader:
        from mlx_lm import load

        model, tokenizer = load(args.pack)
    else:
        model, tokenizer = load_bonsai(args.pack)
    target = model
    if args.drop and args.drop_sub:
        raise SystemExit("use either --drop or --drop-sub, not both")
    if args.drop:
        target = drop_view(model, [int(i) for i in args.drop.split(",")])
    if args.drop_sub:
        from bonsaifold.loader import sublayer_view

        parts = {"a": [], "m": [], "b": []}
        for tokn in args.drop_sub.split("+"):
            parts[tokn[0]].append(int(tokn[1:]))
        target = sublayer_view(
            model, drop_attn=parts["a"], drop_mlp=parts["m"], drop_blocks=parts["b"]
        )

    results = []

    def write():
        by_task = {}
        for r in results:
            by_task.setdefault(r["task"], []).append(r["correct"])
        scores = {t: sum(v) / len(v) for t, v in by_task.items()}
        # atomic replace: crash-safe against battery death mid-write
        tmp = out.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(
                {
                    "pack": args.pack,
                    "drop": args.drop,
                    "drop_sub": args.drop_sub,
                    "stock_loader": args.stock_loader,
                    "max_tokens": args.max_tokens,
                    "seed": args.seed,
                    "task_accuracy": scores,
                    "macro_avg": sum(scores.values()) / len(scores),
                    "items": results,
                },
                indent=1,
            )
        )
        os.replace(tmp, out)

    for k, item in enumerate(data["items"]):
        if item["id"] in done:
            results.append(done[item["id"]])
            continue
        mx.random.seed(args.seed * 1_000_003 + k)  # per-item reproducibility
        prompt = bonsai_prompt(tokenizer, [{"role": "user", "content": item["prompt"]}])
        text = []
        last = None
        for resp in stream_generate(
            target, tokenizer, prompt, max_tokens=args.max_tokens,
            sampler=make_bonsai_sampler(),
        ):
            text.append(resp.text)
            last = resp
        response = "".join(text)
        rec = {
            "id": item["id"],
            "task": item["task"],
            "correct": bool(score(item, response)),
            "gen_tokens": last.generation_tokens,
            "finish_reason": last.finish_reason,
        }
        results.append(rec)
        write()
        acc = sum(r["correct"] for r in results) / len(results)
        print(f"{item['id']}: {'ok' if rec['correct'] else 'MISS'} "
              f"({rec['gen_tokens']} tok) running acc {acc:.3f}")
    write()
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
