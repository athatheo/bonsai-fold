import json

import numpy as np
import pytest

from bonsaifold.fold import drop_blocks, verify_byte_identity
from bonsaifold.stio import PackReader, write_safetensors

N_BLOCKS = 8
LAYER_TYPES = [
    "linear_attention",
    "linear_attention",
    "linear_attention",
    "full_attention",
] * 2


def make_pack(tmp_path, name="src"):
    rng = np.random.default_rng(7)
    d = tmp_path / name
    d.mkdir()
    entries = {}
    for i in range(N_BLOCKS):
        w = rng.integers(0, 2**32, size=(8, 4), dtype=np.uint32)
        s = rng.standard_normal((8, 1)).astype(np.float16)
        base = f"language_model.model.layers.{i}.mlp.up_proj"
        entries[f"{base}.weight"] = {"dtype": "U32", "shape": w.shape, "raw": w.tobytes()}
        entries[f"{base}.scales"] = {"dtype": "F16", "shape": s.shape, "raw": s.tobytes()}
    tail = rng.standard_normal((4, 4)).astype(np.float16)
    entries["language_model.lm_head.weight"] = {
        "dtype": "F16",
        "shape": tail.shape,
        "raw": tail.tobytes(),
    }
    vis = rng.standard_normal((2, 2)).astype(np.float16)
    entries["vision_tower.blocks.0.attn.qkv.weight"] = {
        "dtype": "F16",
        "shape": vis.shape,
        "raw": vis.tobytes(),
    }
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


def test_drop_blocks_basic(tmp_path):
    src = make_pack(tmp_path)
    dst = tmp_path / "dst"
    block_map = drop_blocks(src, dst, drop=[1, 3])
    assert block_map == {0: 0, 2: 1, 4: 2, 5: 3, 6: 4, 7: 5}

    pack = PackReader(dst)
    tcfg = pack.config["text_config"]
    assert tcfg["num_hidden_layers"] == 6
    assert tcfg["layer_types"] == [LAYER_TYPES[i] for i in [0, 2, 4, 5, 6, 7]]
    assert pack.config["bonsai_fold"]["dropped_blocks"] == [1, 3]
    # per-tensor quant override for old block 5 renamed to new block 3
    assert "language_model.model.layers.3.mlp.up_proj" in pack.config["quantization"]
    assert "language_model.model.layers.5.mlp.up_proj" not in pack.config["quantization"]
    # sidecar copied, index written
    assert (dst / "tokenizer_config.json").exists()
    assert (dst / "model.safetensors.index.json").exists()

    report = verify_byte_identity(src, dst, block_map)
    assert report["ok"], report

    # kept tensor bytes are identical to the source under the new name
    src_pack = PackReader(src)
    assert pack.read_raw(
        "language_model.model.layers.1.mlp.up_proj.weight"
    ) == src_pack.read_raw("language_model.model.layers.2.mlp.up_proj.weight")
    # non-layer tensors pass through unchanged
    assert pack.read_raw("language_model.lm_head.weight") == src_pack.read_raw(
        "language_model.lm_head.weight"
    )
    assert "vision_tower.blocks.0.attn.qkv.weight" in pack.header


def test_drop_blocks_validation(tmp_path):
    src = make_pack(tmp_path)
    with pytest.raises(ValueError):
        drop_blocks(src, tmp_path / "bad1", drop=[99])
    with pytest.raises(ValueError):
        drop_blocks(src, tmp_path / "bad2", drop=list(range(N_BLOCKS)))
    drop_blocks(src, tmp_path / "once", drop=[0])
    with pytest.raises(FileExistsError):
        drop_blocks(src, tmp_path / "once", drop=[0])


def test_verify_catches_corruption(tmp_path):
    src = make_pack(tmp_path)
    dst = tmp_path / "dst"
    block_map = drop_blocks(src, dst, drop=[6])
    st = dst / "model.safetensors"
    data = bytearray(st.read_bytes())
    data[-1] ^= 0xFF  # flip a byte in the last tensor's payload
    st.write_bytes(bytes(data))
    report = verify_byte_identity(src, dst, block_map)
    assert not report["ok"]
    assert report["mismatched"]
