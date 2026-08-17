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
