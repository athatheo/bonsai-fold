"""Structural tests of the layer_types-aware loader on a tiny random-weight
qwen3_5 model — no pack files, no 27B load. Verifies block-type assignment,
cache classes, forward passes, the zero-copy layer-subset view, and the
block tap for aligned, non-aligned, all-linear, and all-full patterns
(stock mlx-lm would mis-type all but the first).
"""
import mlx.core as mx
import numpy as np
import pytest
from mlx_lm.models.cache import ArraysCache, KVCache

from bonsaifold.loader import _typed_classes, drop_view, layer_subset_view

TINY_TEXT_CONFIG = {
    "model_type": "qwen3_5_text",
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "full_attention_interval": 4,
    "linear_conv_kernel_dim": 4,
    "linear_key_head_dim": 8,
    "linear_num_key_heads": 2,
    "linear_num_value_heads": 4,
    "linear_value_head_dim": 8,
    "attn_output_gate": True,
    "output_gate_type": "swish",
    "mamba_ssm_dtype": "float32",
    "partial_rotary_factor": 0.25,
    "rope_parameters": {
        "rope_theta": 10000,
        "rope_type": "default",
        "partial_rotary_factor": 0.25,
    },
    "rms_norm_eps": 1e-6,
    "vocab_size": 256,
    "tie_word_embeddings": False,
    "max_position_embeddings": 4096,
}

L, F = "linear_attention", "full_attention"


def build(layer_types):
    config = {
        "model_type": "qwen3_5",
        "text_config": {
            **TINY_TEXT_CONFIG,
            "num_hidden_layers": len(layer_types),
            "layer_types": layer_types,
        },
    }
    Model, ModelArgs = _typed_classes(config)
    return Model(ModelArgs.from_dict(config))


@pytest.mark.parametrize(
    "layer_types",
    [
        [L, L, L, F],  # aligned (stock rule agrees)
        [L, L, F, L, F],  # non-aligned: what a single-block drop produces
        [F, L, L],  # full attention first
        [L, L, L, L],  # all linear
        [F, F],  # all full
    ],
)
def test_types_cache_and_forward(layer_types):
    model = build(layer_types)
    layers = model.language_model.model.layers
    assert [l.is_linear for l in layers] == [t == L for t in layer_types]
    # the type decides which submodule exists
    for l, t in zip(layers, layer_types):
        assert hasattr(l, "linear_attn") == (t == L)
        assert hasattr(l, "self_attn") == (t == F)
    cache = model.make_cache()
    for c, t in zip(cache, layer_types):
        assert isinstance(c, ArraysCache if t == L else KVCache)
    tokens = mx.array([[1, 2, 3, 4, 5]])
    logits = model(tokens, cache=cache)
    mx.eval(logits)
    assert logits.shape == (1, 5, TINY_TEXT_CONFIG["vocab_size"])
    # decode step with populated cache
    step = model(mx.array([[6]]), cache=cache)
    mx.eval(step)
    assert step.shape == (1, 1, TINY_TEXT_CONFIG["vocab_size"])


def test_mismatched_layer_types_rejected():
    config = {
        "model_type": "qwen3_5",
        "text_config": {
            **TINY_TEXT_CONFIG,
            "num_hidden_layers": 3,
            "layer_types": [L, F],  # wrong length
        },
    }
    Model, ModelArgs = _typed_classes(config)
    args = ModelArgs.from_dict(config)
    with pytest.raises(ValueError):
        Model(args)


def test_non_qwen_config_passthrough():
    config = {"model_type": "llama", "hidden_size": 8}
    from mlx_lm.utils import _get_classes

    assert _typed_classes(config) == _get_classes(config=config)


def test_layer_subset_view_matches_full_model():
    """keep=all view must reproduce the full model's logits exactly, and a
    proper-subset view must match a directly-built model sharing the kept
    layers — the zero-copy guarantee behind the Phase 2 sweep."""
    layer_types = [L, L, F, L]
    model = build(layer_types)
    tokens = mx.array([[3, 1, 4, 1, 5]])

    full = model(tokens)
    view_all = layer_subset_view(model, [0, 1, 2, 3])(tokens)
    assert np.array_equal(np.array(full, copy=False), np.array(view_all, copy=False))

    keep = [0, 2, 3]
    view = layer_subset_view(model, keep)
    assert [l.is_linear for l in view.layers] == [True, False, True]
    assert [type(c) for c in view.make_cache()] == [ArraysCache, KVCache, ArraysCache]
    # reference: same forward through the same shared layer objects
    tm = model.language_model.model
    kept_layers = [tm.layers[i] for i in keep]
    assert all(a is b for a, b in zip(view.layers, kept_layers))  # zero-copy
    out = view(tokens)
    mx.eval(out)
    assert out.shape == (1, 5, TINY_TEXT_CONFIG["vocab_size"])
    # the source model is untouched
    assert len(tm.layers) == 4
    assert np.array_equal(np.array(model(tokens), copy=False), np.array(full, copy=False))

    dview = drop_view(model, [1])
    assert [l is k for l, k in zip(dview.layers, kept_layers)] == [True, True, True]
    assert np.array_equal(
        np.array(dview(tokens), copy=False), np.array(out, copy=False)
    )

    with pytest.raises(ValueError):
        layer_subset_view(model, [])
    with pytest.raises(ValueError):
        layer_subset_view(model, [0, 0, 1])
    with pytest.raises(ValueError):
        layer_subset_view(model, [99])


def test_block_tap_fires_per_block():
    layer_types = [L, F, L]
    model = build(layer_types)
    calls = []
    model.set_block_tap(lambda i, h_in, h_out: calls.append((i, h_in.shape, h_out.shape)))
    tokens = mx.array([[7, 8, 9]])
    mx.eval(model(tokens))
    assert [c[0] for c in calls] == [0, 1, 2]
    assert all(c[1] == c[2] == (1, 3, 64) for c in calls)
    model.set_block_tap(None)
    calls.clear()
    mx.eval(model(tokens))
    assert calls == []


def test_sublayer_view_drop_attn_and_mlp():
    from bonsaifold.loader import sublayer_view

    layer_types = [L, F, L, L]
    model = build(layer_types)
    tm = model.language_model.model
    tokens = mx.array([[3, 1, 4, 1, 5]])
    full = np.array(model(tokens), copy=False)

    # drop-attn on block 2: manual forward must match exactly
    v = sublayer_view(model, drop_attn=[2])
    got = np.array(v(tokens), copy=False)
    h = tm.embed_tokens(tokens)
    import mlx_lm.models.qwen3_5 as q35

    ssm_mask = q35.create_ssm_mask(h, None)
    fa_mask = q35.create_attention_mask(h, None)
    for i, l in enumerate(tm.layers):
        mask = ssm_mask if l.is_linear else fa_mask
        if i == 2:
            h = h + l.mlp(l.post_attention_layernorm(h))  # attn skipped
        else:
            h = l(h, mask=mask, cache=None)
    manual = model.language_model.lm_head(tm.norm(h))
    assert np.array_equal(got, np.array(manual, copy=False))
    assert not np.array_equal(got, full)

    # dropping BOTH sublayers of a block == dropping the block
    from bonsaifold.loader import drop_view

    both = sublayer_view(model, drop_attn=[1], drop_mlp=[1])
    assert np.array_equal(
        np.array(both(tokens), copy=False),
        np.array(drop_view(model, [1])(tokens), copy=False),
    )

    # cache classes follow the underlying block types; decode step works
    from mlx_lm.models.cache import ArraysCache, KVCache

    v2 = sublayer_view(model, drop_mlp=[0])
    cache = v2.make_cache()
    assert [type(c) for c in cache] == [ArraysCache, KVCache, ArraysCache, ArraysCache]
    mx.eval(v2(tokens, cache=cache))
    mx.eval(v2(mx.array([[6]]), cache=cache))

    with pytest.raises(ValueError):
        sublayer_view(model)
    with pytest.raises(ValueError):
        sublayer_view(model, drop_attn=[99])
