import json

import numpy as np
import pytest

from bonsaifold.stio import DTYPE_NP, write_safetensors

_NP_TO_ST = {np.dtype(v): k for k, v in DTYPE_NP.items() if k != "BF16"}


def st_entry(arr):
    """safetensors entry dict for a numpy array (raw-bytes form)."""
    return {
        "dtype": _NP_TO_ST[arr.dtype],
        "shape": arr.shape,
        "raw": arr.tobytes(),
    }


N_BLOCKS = 8
LAYER_TYPES = [
    "linear_attention",
    "linear_attention",
    "linear_attention",
    "full_attention",
] * 2


def write_merge_pack(dst, two_bit_blocks=()):
    """Merge-testable synthetic pack: valid 1-bit triples (biases ==
    f16(-scales/2) by construction) + one FP tensor per block, tail, vision
    tensor. two_bit_blocks: store those blocks' triples at 2-bit shapes
    (pre-promoted sources, must be refused by merge_blocks)."""
    rng = np.random.default_rng(20260728)
    dst.mkdir()
    entries = {}
    quant_cfg = {"group_size": 128, "bits": 1}
    for i in range(N_BLOCKS):
        base = f"language_model.model.layers.{i}.mlp.up_proj"
        bits = 2 if i in two_bit_blocks else 1
        words = 256 * bits // 32  # the 2-bit variant differs only in SHAPE:
        # merge_blocks must refuse it via quant_meta before reading any codes
        entries[f"{base}.weight"] = st_entry(
            rng.integers(0, 2**32, size=(8, words), dtype=np.uint32)
        )
        scales = rng.uniform(0.5, 2.0, size=(8, 2)).astype(np.float16)
        biases = (-(scales.astype(np.float32)) / 2).astype(np.float16)
        entries[f"{base}.scales"] = st_entry(scales)
        entries[f"{base}.biases"] = st_entry(biases)
        entries[f"language_model.model.layers.{i}.input_layernorm.weight"] = st_entry(
            rng.standard_normal(16).astype(np.float16)
        )
    entries["language_model.lm_head.weight"] = st_entry(
        rng.standard_normal((4, 4)).astype(np.float16)
    )
    entries["vision_tower.blocks.0.attn.qkv.weight"] = st_entry(
        rng.standard_normal((2, 2)).astype(np.float16)
    )
    write_safetensors(dst / "model.safetensors", entries)
    config = {
        "model_type": "qwen3_5",
        "quantization": dict(quant_cfg),
        "text_config": {
            "model_type": "qwen3_5_text",
            "num_hidden_layers": N_BLOCKS,
            "layer_types": LAYER_TYPES,
            "full_attention_interval": 4,
        },
    }
    (dst / "config.json").write_text(json.dumps(config))
    (dst / "tokenizer_config.json").write_text("{}")
    return dst


@pytest.fixture
def merge_pack(tmp_path):
    return write_merge_pack(tmp_path / "merge_src")


@pytest.fixture
def mini_pack(tmp_path):
    """8-block synthetic pack with quantized-looking tensors, a tail, a
    vision tensor, and one per-tensor quantization override."""
    rng = np.random.default_rng(7)
    d = tmp_path / "src"
    d.mkdir()
    entries = {}
    for i in range(N_BLOCKS):
        base = f"language_model.model.layers.{i}.mlp.up_proj"
        entries[f"{base}.weight"] = st_entry(
            rng.integers(0, 2**32, size=(8, 4), dtype=np.uint32)
        )
        entries[f"{base}.scales"] = st_entry(rng.standard_normal((8, 1)).astype(np.float16))
    entries["language_model.lm_head.weight"] = st_entry(
        rng.standard_normal((4, 4)).astype(np.float16)
    )
    entries["vision_tower.blocks.0.attn.qkv.weight"] = st_entry(
        rng.standard_normal((2, 2)).astype(np.float16)
    )
    write_safetensors(d / "model.safetensors", entries)
    config = {
        "model_type": "qwen3_5",
        "quantization": {
            "group_size": 128,
            "bits": 1,
            "language_model.model.layers.5.mlp.up_proj": {"bits": 2},
        },
        "text_config": {
            "model_type": "qwen3_5_text",
            "num_hidden_layers": N_BLOCKS,
            "layer_types": LAYER_TYPES,
            "full_attention_interval": 4,
        },
    }
    (d / "config.json").write_text(json.dumps(config))
    (d / "tokenizer_config.json").write_text("{}")
    return d
