"""Phase 3 prerequisite: build real merged packs for the top shortlist pair
and prove the (possibly heterogeneous-bits) stack loads and generates.

Writes models/merged_{i}-{j}_{operator}/ (gitignored), runs verify_merge
(exact recompute of every merged tensor), then loads via load_bonsai and
generates a short completion with the mandated sampling. The promotion pack
is the first heterogeneous 1-bit/2-bit stack through this runtime — the
per-tensor bits=2 overrides must dispatch correctly or generation will be
garbage/crash (plan §7 risk item).

Usage: uv run python experiments/merge/build_and_smoke.py [--pair 4,5]
"""
import argparse
import json
import shutil
import time
from pathlib import Path

SRC = "models/Bonsai-27B-mlx-1bit"
PROMPT = "Explain in two sentences why the sky is blue."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", default="4,5")
    ap.add_argument("--max-tokens", type=int, default=200)
    args = ap.parse_args()
    i, j = (int(x) for x in args.pair.split(","))

    from mlx_lm import stream_generate

    from bonsaifold import bonsai_prompt, make_bonsai_sampler
    from bonsaifold.loader import load_bonsai
    from bonsaifold.merge import merge_blocks, verify_merge

    report = {"pair": [i, j], "operators": {}}
    for operator in ("sign_election", "promotion"):
        dst = Path(f"models/merged_{i}-{j}_{operator}")
        if dst.exists():
            shutil.rmtree(dst)  # rebuild from scratch; packs are derived artifacts
        t0 = time.time()
        block_map = merge_blocks(SRC, dst, i, j, operator)
        t_build = time.time() - t0
        t0 = time.time()
        ver = verify_merge(SRC, dst, block_map, i, j, operator)
        t_verify = time.time() - t0
        assert ver["ok"], f"{operator}: verify_merge failed: " + json.dumps(
            {k: v for k, v in ver.items() if k != "merged_checks"}
        )
        model, tokenizer = load_bonsai(dst)
        prompt = bonsai_prompt(tokenizer, [{"role": "user", "content": PROMPT}])
        text, last = [], None
        t0 = time.time()
        for resp in stream_generate(
            model, tokenizer, prompt, max_tokens=args.max_tokens,
            sampler=make_bonsai_sampler(),
        ):
            text.append(resp.text)
            last = resp
        gen = "".join(text)
        report["operators"][operator] = {
            "build_s": round(t_build, 1),
            "verify_s": round(t_verify, 1),
            "verify_ok": ver["ok"],
            "survivors_checked": ver["survivors_checked"],
            "gen_tokens": last.generation_tokens,
            "decode_tps": round(last.generation_tps, 1),
            "peak_gb": round(last.peak_memory, 2),
            "pack_gb": round(
                sum(f.stat().st_size for f in dst.glob("*.safetensors")) / 1e9, 3
            ),
            "sample_tail": gen[-300:],
        }
        print(f"=== {operator}: build {t_build:.0f}s verify {t_verify:.0f}s "
              f"{last.generation_tokens} tok @ {last.generation_tps:.1f} tps")
        print(gen[:400])
        del model
    out = Path("experiments/merge/build_and_smoke_report.json")
    out.write_text(json.dumps(report, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
