"""Phase 0 model census of an MLX Bonsai pack.

Builds the per-block inventory from config.json and the safetensors header(s)
alone; the packing verification additionally reads scales+biases for every
quantized tensor and the packed words of 2-bit and histogram-sampled tensors.
Writes JSON + markdown to this directory.

Usage: uv run python experiments/census/run_census.py models/Bonsai-27B-mlx-1bit
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from bonsaifold.stio import PackReader, block_owner, is_quantized, unpack_codes, weight_base

HIST_SAMPLE_TARGET = 2_000_000  # codes per sampled histogram
F16_SMALLEST_NORMAL = 6.104e-05


def main(pack_dir):
    pack = PackReader(pack_dir)
    outdir = Path(__file__).parent
    tcfg = pack.config["text_config"]
    header = pack.header
    quant_names = sorted(n for n in header if is_quantized(n, header))

    layer_types = tcfg["layer_types"]
    full_attn_idx = [i for i, t in enumerate(layer_types) if t == "full_attention"]

    blocks = defaultdict(list)
    for name in header:
        blocks[block_owner(name)].append(name)

    def tensor_record(name):
        info = header[name]
        rec = {"name": name, "dtype": info["dtype"], "shape": info["shape"]}
        rec["bytes"] = info["data_offsets"][1] - info["data_offsets"][0]
        if name in quant_names:
            gs, bits = pack.quant_meta(name)
            scales = header[weight_base(name) + ".scales"]
            rec["quant"] = {"bits": bits, "group_size": gs}
            rec["logical_params"] = info["shape"][0] * scales["shape"][-1] * gs
        else:
            rec["logical_params"] = int(np.prod(info["shape"]))
        return rec

    inventory = []
    for i, ltype in enumerate(layer_types):
        tensors = [tensor_record(n) for n in sorted(blocks[i])]
        inventory.append(
            {
                "block": i,
                "attention_type": ltype,
                "n_tensors": len(tensors),
                "params": sum(t["logical_params"] for t in tensors),
                "bytes": sum(t["bytes"] for t in tensors),
                "tensors": tensors,
            }
        )

    tail = [tensor_record(n) for n in sorted(blocks["tail"])]
    vision = [tensor_record(n) for n in sorted(blocks["vision_tower"])]

    # ---- packing verification ----
    # Paper mapping (1-bit): w = s_mlx*q + b_mlx, q in {0,1}, s_mlx = 2*s_g,
    # b_mlx = -s_g  =>  biases == f16(-scales/2) elementwise. Exact in f16
    # except where f16 subnormal rounding loses the last bit, so 1-bit (and
    # ONLY 1-bit: negating an f16 is always exact, so the 2-bit relation
    # biases == -scales admits no rounding escape) tolerates subnormal-scale
    # deviations up to one ulp. Ternary (2-bit): q in {0,1,2}, code 3 unused,
    # checked on the packed words of every 2-bit tensor (full coverage, no
    # unpacking). Scale/bias relations are checked on EVERY quantized tensor;
    # relations are only defined for f16 storage — other dtypes and other bit
    # widths are reported as unverified. Verdicts are three-state: True
    # (verified), False (violated), None (no known relation); all_ok means
    # "nothing violated".
    packing = {"tensors": []}
    rng = np.random.default_rng(0)
    hist_sample = set(rng.choice(quant_names, size=min(24, len(quant_names)), replace=False))
    hist_sample.update(
        n
        for n in [
            "language_model.model.embed_tokens.weight",
            "language_model.lm_head.weight",
            "language_model.model.layers.0.linear_attn.in_proj_qkv.weight",
            "language_model.model.layers.3.self_attn.q_proj.weight",
        ]
        if n in header
    )
    for wname in quant_names:
        base = weight_base(wname)
        gs, bits = pack.quant_meta(wname)
        s = pack.read(base + ".scales")
        b = pack.read(base + ".biases")
        rec = {"tensor": wname, "bits": int(bits), "group_size": gs}

        pred = None
        if s.dtype == np.float16:
            if bits == 1:
                pred = (-(s.astype(np.float32) / 2)).astype(np.float16)
            elif bits == 2:
                pred = -s
        if pred is None:
            rec["scale_bias_ok"] = None  # no known relation for this width/dtype
        else:
            dev = pred != b
            n_dev = int(dev.sum())
            rec["scale_bias_ok"] = n_dev == 0
            if n_dev:
                rec["deviating_groups"] = n_dev
                rec["deviating_scale_absmax"] = float(np.abs(s[dev].astype(np.float32)).max())
                rec["deviation_absmax"] = float(
                    np.abs(b[dev].astype(np.float32) - pred[dev].astype(np.float32)).max()
                )
                if bits == 1:
                    # benign only if every deviation is subnormal f16 rounding
                    sub_ok = (
                        rec["deviating_scale_absmax"] < F16_SMALLEST_NORMAL
                        and rec["deviation_absmax"] <= 6e-08
                    )
                    rec["deviations_subnormal_only"] = sub_ok
                    rec["scale_bias_ok"] = sub_ok

        w = pack.read(wname) if (bits == 2 or wname in hist_sample) else None
        if bits == 2:
            # code 3 = both bits of a 2-bit slot set; check on packed words.
            odd_mask = np.uint32(0x55555555)
            rec["code3_unused"] = not bool((w & (w >> np.uint32(1)) & odd_mask).any())
        if wname in hist_sample:
            # Sample packed words BEFORE unpacking (a full unpack of
            # embed/lm_head would materialize ~5 GB per tensor).
            per_word = 32 // bits
            words = w.reshape(-1)
            step = max(1, words.size // (HIST_SAMPLE_TARGET // per_word))
            codes = unpack_codes(words[::step], bits)
            rec["code_histogram"] = np.bincount(codes, minlength=1 << bits).tolist()
            rec["code_histogram_word_step"] = step  # sampled, not full coverage

        rec["violated"] = rec["scale_bias_ok"] is False or rec.get("code3_unused") is False
        packing["tensors"].append(rec)

    packing["all_ok"] = not any(t["violated"] for t in packing["tensors"])
    packing["unverified"] = sum(1 for t in packing["tensors"] if t["scale_bias_ok"] is None)

    # ---- summary ----
    lang_bytes = sum(bl["bytes"] for bl in inventory) + sum(t["bytes"] for t in tail)
    vis_bytes = sum(t["bytes"] for t in vision)
    census = {
        "pack": str(pack.pack_dir),
        "model_type": pack.config["model_type"],
        "num_language_blocks": len(layer_types),
        "full_attention_indices": full_attn_idx,
        "interleave_pattern": "".join("F" if t == "full_attention" else "L" for t in layer_types),
        "quantization": pack.config.get("quantization"),
        "language_blocks_bytes": lang_bytes,
        "vision_tower_bytes": vis_bytes,
        "blocks": inventory,
        "tail_tensors": tail,
        "vision_tensors_count": len(vision),
        "packing_verification": packing,
    }
    name = pack.pack_dir.name
    (outdir / f"census_{name}.json").write_text(json.dumps(census, indent=1))

    ltype_short = {"linear_attention": "linear", "full_attention": "full"}
    lines = [
        f"# Census: {name}",
        "",
        f"- model_type `{pack.config['model_type']}`, {len(layer_types)} language blocks, "
        f"quant {pack.config.get('quantization')}",
        f"- full-attention blocks ({len(full_attn_idx)}): {full_attn_idx}",
        f"- pattern: `{census['interleave_pattern']}`",
        "- NOTE: mlx-lm's qwen3_5 runtime derives block types POSITIONALLY from "
        "`full_attention_interval` (is_linear = (i+1) % 4 != 0); it never reads "
        "`layer_types`. Any block-drop harness must re-establish types itself.",
        f"- language bytes {lang_bytes/1e9:.3f} GB, vision tower {vis_bytes/1e9:.3f} GB "
        f"({len(vision)} tensors, prefix vision_tower.*)",
        f"- packing verification: all_ok={packing['all_ok']} over {len(packing['tensors'])} "
        f"quantized tensors ({packing['unverified']} with no known relation for their bit-width)",
        "",
        "| block | type | tensors | params (M) | MB |",
        "|---|---|---|---|---|",
    ]
    for bl in inventory:
        lines.append(
            f"| {bl['block']} | {ltype_short[bl['attention_type']]} | {bl['n_tensors']} "
            f"| {bl['params']/1e6:.1f} | {bl['bytes']/1e6:.1f} |"
        )
    tail_p = sum(t["logical_params"] for t in tail)
    lines += [
        "",
        f"Tail (embed/lm_head/final norm): {tail_p/1e6:.1f} M logical params, "
        f"{sum(t['bytes'] for t in tail)/1e6:.1f} MB",
    ]
    (outdir / f"census_{name}.md").write_text("\n".join(lines) + "\n")
    print(f"wrote census_{name}.json / .md; packing all_ok={packing['all_ok']}")
    print(f"full attention at: {full_attn_idx}")


if __name__ == "__main__":
    main(sys.argv[1])
