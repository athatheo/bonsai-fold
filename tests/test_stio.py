import json

import numpy as np
import pytest

from bonsaifold.stio import (
    PackReader,
    block_owner,
    infer_bits,
    is_quantized,
    manual_dequant,
    read_st_header,
    reindex_name,
    unpack_codes,
    weight_base,
    write_safetensors,
)


def test_block_owner_and_weight_base():
    assert block_owner("language_model.model.layers.17.mlp.up_proj.weight") == 17
    assert block_owner("vision_tower.blocks.3.attn.qkv.weight") == "vision_tower"
    assert block_owner("language_model.lm_head.weight") == "tail"
    assert weight_base("a.b.weight") == "a.b"
    assert weight_base("a.b.scales") is None


def test_reindex_name():
    m = {0: 0, 2: 1, 3: 2}  # block 1 dropped
    assert (
        reindex_name("language_model.model.layers.2.mlp.up_proj.weight", m)
        == "language_model.model.layers.1.mlp.up_proj.weight"
    )
    assert reindex_name("language_model.model.layers.1.mlp.up_proj.weight", m) is None
    assert reindex_name("language_model.lm_head.weight", m) == "language_model.lm_head.weight"
    assert (
        reindex_name("vision_tower.blocks.1.attn.qkv.weight", m)
        == "vision_tower.blocks.1.attn.qkv.weight"
    )


def test_unpack_and_dequant_1bit():
    # word 0b...0101 -> codes 1,0,1,0,... LSB-first
    packed = np.array([[0x55555555, 0x00000001]], dtype=np.uint32)
    codes = unpack_codes(packed, 1)
    assert codes.shape == (1, 64)
    assert codes[0, 0] == 1 and codes[0, 1] == 0 and codes[0, 32] == 1 and codes[0, 33] == 0
    s = np.array([[2.0]], dtype=np.float16)  # one group of 64? group_size=64
    b = np.array([[-1.0]], dtype=np.float16)
    w = manual_dequant(packed, s, b, bits=1, group_size=64)
    assert set(np.unique(w)) == {-1.0, 1.0}


def test_unpack_2bit_codes():
    # 2-bit codes: word 0b...100100 -> 0,1,2,0,... LSB-first
    word = sum(((i % 3) << (2 * i)) for i in range(16))
    codes = unpack_codes(np.array([[word]], dtype=np.uint32), 2)
    assert list(codes[0][:6]) == [0, 1, 2, 0, 1, 2]


def test_infer_bits():
    assert infer_bits([256, 544], [256, 136], 128) == 1  # 17408 cols
    assert infer_bits([256, 1088], [256, 136], 128) == 2


def test_write_read_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    a = rng.standard_normal((4, 8)).astype(np.float16)
    b = rng.integers(0, 2**32, size=(2, 3), dtype=np.uint32)
    entries = {
        "t.a": {"dtype": "F16", "shape": a.shape, "raw": a.tobytes()},
        "t.b.weight": {"dtype": "U32", "shape": b.shape, "raw": b.tobytes()},
    }
    p = tmp_path / "model.safetensors"
    write_safetensors(p, entries)
    header, ds = read_st_header(p)
    assert set(header) == {"t.a", "t.b.weight"}
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "x"}))
    pack = PackReader(tmp_path)
    assert np.array_equal(pack.read("t.a"), a)
    assert np.array_equal(pack.read("t.b.weight"), b)
    assert pack.read_raw("t.a") == a.tobytes()
    assert not is_quantized("t.b.weight", pack.header)  # no scales sibling


def test_quant_params_overrides(tmp_path):
    cfg = {
        "model_type": "x",
        "quantization": {"group_size": 128, "bits": 1, "some.block": {"bits": 2}},
    }
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    write_safetensors(tmp_path / "model.safetensors", {})
    pack = PackReader(tmp_path)
    assert pack.quant_params() == (128, 1)
    assert pack.quant_params("some.block") == (128, 2)
    assert pack.quant_params("other.block") == (128, 1)
