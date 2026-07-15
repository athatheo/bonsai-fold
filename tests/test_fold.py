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
