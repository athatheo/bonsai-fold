"""Group C operator unit tests (synthetic; no model load)."""
import json

import mlx.core as mx
import numpy as np
import pytest
from conftest import LAYER_TYPES, N_BLOCKS, st_entry

from bonsaifold.fold import strip_bias_plane
from bonsaifold.scalequant import (
    QMAX,
    quantize_scale_plane,
    quantize_scales,
    reconstruct_scales,
    roundtrip_scales,
)
from bonsaifold.stio import PackReader, write_safetensors


def test_shapes_and_dtypes():
    s = mx.random.uniform(0.001, 0.1, (64, 40)).astype(mx.float16)
    q, lo, step = quantize_scales(s)
    assert q.dtype == mx.uint8 and q.shape == s.shape
    assert lo.dtype == mx.float16 and lo.shape == (64, 1)
    assert step.dtype == mx.float16 and step.shape == (64, 1)
    assert reconstruct_scales(q, lo, step).dtype == mx.float16


def test_relative_error_bound():
    # 8-bit affine over a row: |s' - s| <= step/2 + f16 rounding
    s = mx.random.uniform(0.001, 0.1, (256, 40)).astype(mx.float16)
    s2 = roundtrip_scales(s)
    f, f2 = np.array(s, dtype=np.float32), np.array(s2, dtype=np.float32)
    step = (f.max(-1) - f.min(-1)) / QMAX
    err = np.abs(f2 - f).max(-1)
    assert (err <= step * 0.5 + np.abs(f).max(-1) * 2e-3 + 1e-6).all()


def test_constant_row_is_exact():
    s = mx.full((4, 32), 0.03125, dtype=mx.float16)  # exact in f16
    assert bool(mx.array_equal(roundtrip_scales(s), s))


def test_endpoints_are_preserved():
    # row min and max quantize to codes 0 and QMAX -> reconstruct to the
    # f16-rounded endpoints
    s = mx.array([[0.01, 0.02, 0.05, 0.09]]).astype(mx.float16)
    q, lo, step = quantize_scales(s)
    qn = np.array(q)
    assert qn.min() == 0 and qn.max() == QMAX
    s2 = np.array(reconstruct_scales(q, lo, step), dtype=np.float32)
    assert abs(s2[0, 0] - 0.01) < 2e-4 and abs(s2[0, -1] - 0.09) < 2e-3


def test_negative_and_mixed_rows():
    s = mx.array([[-0.05, -0.01, 0.0, 0.02]]).astype(mx.float16)
    s2 = roundtrip_scales(s)
    err = np.abs(np.array(s2 - s, dtype=np.float32))
    assert err.max() < (0.07 / QMAX) * 0.5 + 1e-3


def test_determinism():
    s = mx.random.uniform(0.001, 0.1, (32, 32)).astype(mx.float16)
    a, b = roundtrip_scales(s), roundtrip_scales(s)
    assert bool(mx.array_equal(a, b))


def write_wide_pack(tmp_path):
    """Synthetic pack with real-geometry scales (K=4096 -> 32 cols at g128).
    Returns the pack dir."""
    rng = np.random.default_rng(11)
    d = tmp_path / "wide_src"
    d.mkdir()
    entries = {}
    for i in range(N_BLOCKS):
        base = f"language_model.model.layers.{i}.mlp.up_proj"
        entries[f"{base}.weight"] = st_entry(
            rng.integers(0, 2**32, size=(8, 128), dtype=np.uint32)
        )
        entries[f"{base}.scales"] = st_entry(
            rng.uniform(0.001, 0.1, (8, 32)).astype(np.float16)
        )
    write_safetensors(d / "model.safetensors", entries)
    (d / "config.json").write_text(json.dumps({
        "model_type": "qwen3_5",
        "quantization": {"group_size": 128, "bits": 1},
        "text_config": {
            "model_type": "qwen3_5_text",
            "num_hidden_layers": N_BLOCKS,
            "layer_types": LAYER_TYPES,
            "full_attention_interval": 4,
        },
    }))
    (d / "tokenizer_config.json").write_text("{}")
    return d


def test_quantize_scale_plane_pack(tmp_path):
    """strip biases -> Group C pack: q8 triplet replaces 1-bit scales, other
    tensors byte-identical, reconstruction == in-memory roundtrip."""
    nobias = tmp_path / "nobias"
    strip_bias_plane(write_wide_pack(tmp_path), nobias)
    dst = tmp_path / "groupc"
    n, saved = quantize_scale_plane(nobias, dst)
    # 8 blocks x (8x32) scales: 512B f16 -> 256B q8 + 32B meta each
    assert n == 8 and saved == 8 * (512 - 256 - 32)

    src, out = PackReader(nobias), PackReader(dst)
    base = "language_model.model.layers.0.mlp.up_proj"
    assert base + ".scales" not in out.header
    for suf in (".scales_q8", ".scales_lo", ".scales_step"):
        assert base + suf in out.header
    got = reconstruct_scales(
        out.read(base + ".scales_q8"),
        out.read(base + ".scales_lo"),
        out.read(base + ".scales_step"),
    )
    assert bool(mx.array_equal(got, roundtrip_scales(src.read(base + ".scales"))))
    # untouched planes are byte copies
    assert out.read_raw(base + ".weight") == src.read_raw(base + ".weight")
    tc = out.config["text_config"]
    assert tc["bonsai_scale_plane"] == {"bits": 8, "grouping": "row"}
    assert out.config["bonsai_fold"]["flagged_arm"] == "group_c_value_modifying"


def test_quantize_scale_plane_requires_derived(tmp_path, mini_pack):
    with pytest.raises(ValueError, match="bias-derived"):
        quantize_scale_plane(mini_pack, tmp_path / "out")


def test_unprofitable_geometry_falls_back_to_copy(tmp_path, mini_pack):
    """mini_pack scales are (8,1): the q8 triplet would be larger, so the
    operator must keep the original plane byte-identical."""
    nobias = tmp_path / "nobias"
    strip_bias_plane(mini_pack, nobias)
    n, saved = quantize_scale_plane(nobias, tmp_path / "groupc")
    assert n == 0 and saved == 0
    src, out = PackReader(nobias), PackReader(tmp_path / "groupc")
    base = "language_model.model.layers.0.mlp.up_proj"
    assert base + ".scales_q8" not in out.header
    assert out.read_raw(base + ".scales") == src.read_raw(base + ".scales")


# ---- Group C x downstream-operator interactions (review findings 2026-08-17)


def test_reconstruct_scale_plane_dict():
    from bonsaifold.loader import reconstruct_scale_plane

    s = mx.random.uniform(0.001, 0.1, (8, 32)).astype(mx.float16)
    q, lo, step = quantize_scales(s)
    w = {"x.scales_q8": q, "x.scales_lo": lo, "x.scales_step": step}
    reconstruct_scale_plane(w)
    assert set(w) == {"x.scales"}
    assert bool(mx.array_equal(w["x.scales"], roundtrip_scales(s)))


def test_reconstruct_scale_plane_refuses_corrupt():
    from bonsaifold.loader import reconstruct_scale_plane

    s = mx.random.uniform(0.001, 0.1, (8, 32)).astype(mx.float16)
    q, lo, step = quantize_scales(s)
    with pytest.raises(ValueError, match="both"):
        reconstruct_scale_plane(
            {"x.scales": s, "x.scales_q8": q, "x.scales_lo": lo, "x.scales_step": step}
        )
    with pytest.raises(ValueError, match="_lo/_step"):
        reconstruct_scale_plane({"x.scales_q8": q, "x.scales_lo": lo})


def test_materialize_bias_plane_only_one_bit():
    from bonsaifold.loader import materialize_bias_plane

    s1 = mx.random.uniform(0.001, 0.1, (8, 2)).astype(mx.float16)
    w1 = mx.zeros((8, 8), dtype=mx.uint32)  # ratio 4 -> 1-bit g128
    s2 = mx.random.uniform(0.001, 0.1, (8, 2)).astype(mx.float16)
    w2 = mx.zeros((8, 16), dtype=mx.uint32)  # ratio 8 -> 2-bit (promoted)
    w = {"a.scales": s1, "a.weight": w1, "b.scales": s2, "b.weight": w2}
    materialize_bias_plane(w)
    assert "a.biases" in w and "b.biases" not in w
    expected = (-(s1.astype(mx.float32) / 2)).astype(mx.float16)
    assert bool(mx.array_equal(w["a.biases"], expected))


def test_apply_roundtrip_filters_one_bit():
    import mlx.nn as nn

    from bonsaifold.scalequant import apply_roundtrip

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.one = nn.QuantizedLinear(256, 8, bias=False, group_size=128, bits=1)
            self.four = nn.QuantizedLinear(256, 8, bias=False, group_size=64, bits=4)

    m = M()
    s_one, s_four = m.one.scales, m.four.scales
    n, max_rel = apply_roundtrip(m)
    assert n == 1 and max_rel >= 0
    assert bool(mx.array_equal(m.one.scales, roundtrip_scales(s_one)))
    assert bool(mx.array_equal(m.four.scales, s_four))  # untouched control


def _groupc_pack(tmp_path):
    nobias = tmp_path / "nobias"
    strip_bias_plane(write_wide_pack(tmp_path), nobias)
    gc = tmp_path / "gc"
    quantize_scale_plane(nobias, gc)
    return gc


def test_trim_lm_head_refuses_groupc(tmp_path):
    from bonsaifold.fold import trim_lm_head

    with pytest.raises(ValueError, match="scale-quantized"):
        trim_lm_head(_groupc_pack(tmp_path), tmp_path / "out", [0, 1])


def test_merge_blocks_refuses_groupc(tmp_path):
    from bonsaifold.merge import merge_blocks

    with pytest.raises(ValueError, match="scale-quantized"):
        merge_blocks(_groupc_pack(tmp_path), tmp_path / "out", 0, 1, "sign_election")


def test_block_tensor_kinds_refuses_orphan_u32(tmp_path):
    from bonsaifold.merge import _block_tensor_kinds

    d = tmp_path / "orphan"
    d.mkdir()
    write_safetensors(d / "model.safetensors", {
        "language_model.model.layers.0.mlp.up_proj.weight": st_entry(
            np.zeros((4, 8), dtype=np.uint32)
        ),
    })
    (d / "config.json").write_text(json.dumps({
        "model_type": "qwen3_5", "quantization": {"group_size": 128, "bits": 1},
        "text_config": {"num_hidden_layers": 1, "layer_types": ["linear_attention"],
                        "full_attention_interval": 4},
    }))
    with pytest.raises(ValueError, match="without a .scales partner"):
        _block_tensor_kinds(PackReader(d), 0)


def test_strip_bias_plane_refuses_wrong_invariant(tmp_path):
    src = write_wide_pack(tmp_path)
    # rewrite one tensor's biases to the 2-bit promotion convention (b == -s)
    r = PackReader(src)
    entries = {}
    for name in r.header:
        arr = r.read(name)
        entries[name] = st_entry(arr)
        if name.endswith(".scales"):
            entries[name[: -len(".scales")] + ".biases"] = st_entry(-arr)
    write_safetensors(src / "model.safetensors", entries)
    with pytest.raises(ValueError, match="refusing to strip"):
        strip_bias_plane(src, tmp_path / "out")


def test_flagged_arm_provenance_survives_drop(tmp_path):
    from bonsaifold.fold import drop_blocks

    gc = _groupc_pack(tmp_path)
    dst = tmp_path / "dropped"
    drop_blocks(gc, dst, [1])
    cfg = PackReader(dst).config
    assert cfg["bonsai_fold"]["operation"] == "drop"
    assert cfg["bonsai_fold"]["flagged_arm"] == "group_c_value_modifying"
    ops = [h["operation"] for h in cfg["bonsai_fold"]["history"]]
    assert ops == ["strip_bias_plane", "quantize_scale_plane"]
    assert cfg["text_config"]["bonsai_scale_plane"]["bits"] == 8


def test_mixed_bits_pack_keeps_two_bit_scales_f16(tmp_path):
    d = tmp_path / "mixed"
    d.mkdir()
    rng = np.random.default_rng(3)
    entries = {}
    for i, wcols in enumerate((128, 256)):  # block 0: 1-bit ratio, block 1: 2-bit ratio
        base = f"language_model.model.layers.{i}.mlp.up_proj"
        entries[f"{base}.weight"] = st_entry(
            rng.integers(0, 2**32, size=(8, wcols), dtype=np.uint32))
        entries[f"{base}.scales"] = st_entry(
            rng.uniform(0.001, 0.1, (8, 32)).astype(np.float16))
    write_safetensors(d / "model.safetensors", entries)
    (d / "config.json").write_text(json.dumps({
        "model_type": "qwen3_5",
        "quantization": {"group_size": 128, "bits": 1,
                         "language_model.model.layers.1.mlp.up_proj": {"bits": 2}},
        "text_config": {"num_hidden_layers": 2,
                        "layer_types": ["linear_attention", "linear_attention"],
                        "full_attention_interval": 4,
                        "bonsai_bias_plane": "derived"},
    }))
    (d / "tokenizer_config.json").write_text("{}")
    out = tmp_path / "gc_mixed"
    n, saved = quantize_scale_plane(d, out)
    assert n == 1
    r = PackReader(out)
    assert "language_model.model.layers.0.mlp.up_proj.scales_q8" in r.header
    assert "language_model.model.layers.1.mlp.up_proj.scales" in r.header  # 2-bit stays f16
