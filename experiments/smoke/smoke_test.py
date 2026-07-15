"""Phase 0 smoke test: ~200 tokens in thinking mode, report tok/s + peak memory.

Usage: uv run python experiments/smoke/smoke_test.py models/Bonsai-27B-mlx-1bit
"""
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load, stream_generate

from bonsaifold import bonsai_prompt, make_bonsai_sampler

PROMPT = "A tree is pruned so that exactly one branch is removed each day. If it starts with 64 branches and pruning stops when 48 remain, how many days does pruning take? Briefly explain."


def main(pack_dir, max_tokens=200):
    model, tokenizer = load(pack_dir)
    prompt = bonsai_prompt(tokenizer, [{"role": "user", "content": PROMPT}])
    sampler = make_bonsai_sampler()

    mx.reset_peak_memory()
    text = []
    last = None
    t0 = time.perf_counter()
    for resp in stream_generate(model, tokenizer, prompt, max_tokens=max_tokens, sampler=sampler):
        text.append(resp.text)
        last = resp
    wall = time.perf_counter() - t0

    result = {
        "pack": str(pack_dir),
        "prompt_tokens": last.prompt_tokens,
        "prompt_tps": round(last.prompt_tps, 2),
        "generation_tokens": last.generation_tokens,
        "generation_tps": round(last.generation_tps, 2),
        "wall_seconds": round(wall, 2),
        "peak_memory_gb": round(last.peak_memory, 3),
        "mlx_version": mx.__version__,
        "output": "".join(text),
    }
    out = Path(__file__).parent / f"smoke_{Path(pack_dir).name}.json"
    out.write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items() if k != "output"}, indent=1))
    print("--- first 400 chars of output ---")
    print(result["output"][:400])


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 200)
