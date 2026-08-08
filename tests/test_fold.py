import pytest
from conftest import LAYER_TYPES, N_BLOCKS

from bonsaifold.fold import drop_blocks, verify_byte_identity
from bonsaifold.stio import PackReader


def test_drop_blocks_basic(tmp_path, mini_pack):
    dst = tmp_path / "dst"
    block_map = drop_blocks(mini_pack, dst, drop=[1, 3])
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

    report = verify_byte_identity(mini_pack, dst, block_map)
    assert report["ok"], report

    # kept tensor bytes are identical to the source under the new name
    src_pack = PackReader(mini_pack)
    assert pack.read_raw(
        "language_model.model.layers.1.mlp.up_proj.weight"
    ) == src_pack.read_raw("language_model.model.layers.2.mlp.up_proj.weight")
    # non-layer tensors pass through unchanged
    assert pack.read_raw("language_model.lm_head.weight") == src_pack.read_raw(
        "language_model.lm_head.weight"
    )
    assert "vision_tower.blocks.0.attn.qkv.weight" in pack.header


def test_drop_blocks_validation(tmp_path, mini_pack):
    with pytest.raises(ValueError):
        drop_blocks(mini_pack, tmp_path / "bad1", drop=[99])
    with pytest.raises(ValueError):
        drop_blocks(mini_pack, tmp_path / "bad2", drop=list(range(N_BLOCKS)))
    drop_blocks(mini_pack, tmp_path / "once", drop=[0])
    with pytest.raises(FileExistsError):
        drop_blocks(mini_pack, tmp_path / "once", drop=[0])


def test_verify_catches_corruption(tmp_path, mini_pack):
    dst = tmp_path / "dst"
    block_map = drop_blocks(mini_pack, dst, drop=[6])
    st = dst / "model.safetensors"
    data = bytearray(st.read_bytes())
    data[-1] ^= 0xFF  # flip a byte in the last tensor's payload
    st.write_bytes(bytes(data))
    report = verify_byte_identity(mini_pack, dst, block_map)
    assert not report["ok"]
    assert report["mismatched"]


def test_verify_catches_missing_and_extra(tmp_path, mini_pack):
    dst = tmp_path / "dst"
    block_map = drop_blocks(mini_pack, dst, drop=[0])
    # claim block 7 also survived: its dst name is absent -> missing
    bad_map = dict(block_map)
    report = verify_byte_identity(mini_pack, dst, {**bad_map, 7: 99})
    assert not report["ok"] and report["missing"]
    # claim block 2 was dropped: its dst tensors become extra
    del bad_map[2]
    report = verify_byte_identity(mini_pack, dst, bad_map)
    assert not report["ok"] and report["extra"]


def test_strip_bias_plane(tmp_path, mini_pack):
    import numpy as np

    from bonsaifold.fold import strip_bias_plane
    from bonsaifold.stio import PackReader

    dst = tmp_path / "nobias"
    n = strip_bias_plane(mini_pack, dst)
    src, out = PackReader(mini_pack), PackReader(dst)
    # mini_pack has weight+scales pairs (quantized bases counted) but no
    # .biases tensors, so nothing is dropped — pure pass-through copy
    assert n == 8
    # every kept tensor is byte-identical
    for name in out.header:
        assert out.read_raw(name) == src.read_raw(name)
    assert out.config["text_config"]["bonsai_bias_plane"] == "derived"


def test_strip_bias_plane_removes_bias_tensors(tmp_path):
    import json

    import numpy as np
    from conftest import st_entry

    from bonsaifold.fold import strip_bias_plane
    from bonsaifold.stio import PackReader, write_safetensors

    rng = np.random.default_rng(3)
    d = tmp_path / "src"
    d.mkdir()
    entries = {}
    scales = rng.uniform(0.5, 2.0, size=(8, 2)).astype(np.float16)
    biases = (-(scales.astype(np.float32)) / 2).astype(np.float16)
    base = "language_model.model.layers.0.mlp.up_proj"
    entries[f"{base}.weight"] = st_entry(
        rng.integers(0, 2**32, size=(8, 8), dtype=np.uint32)
    )
    entries[f"{base}.scales"] = st_entry(scales)
    entries[f"{base}.biases"] = st_entry(biases)
    entries["language_model.model.norm.weight"] = st_entry(
        rng.standard_normal(16).astype(np.float16)
    )
    write_safetensors(d / "model.safetensors", entries)
    (d / "config.json").write_text(json.dumps({
        "model_type": "qwen3_5",
        "quantization": {"group_size": 128, "bits": 1},
        "text_config": {"model_type": "qwen3_5_text", "num_hidden_layers": 1,
                        "layer_types": ["linear_attention"],
                        "full_attention_interval": 4},
    }))
    dst = tmp_path / "nobias"
    n = strip_bias_plane(d, dst)
    assert n == 1
    out = PackReader(dst)
    assert f"{base}.biases" not in out.header
    assert out.read_raw(f"{base}.weight") == entries[f"{base}.weight"]["raw"]
    assert out.read_raw(f"{base}.scales") == scales.tobytes()
    # loader-side derivation reproduces the stored biases bit-exactly
    import mlx.core as mx
    derived = (-(mx.array(scales).astype(mx.float32) / 2)).astype(mx.float16)
    assert np.array_equal(np.array(derived, copy=False), biases)
