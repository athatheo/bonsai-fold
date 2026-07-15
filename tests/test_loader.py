"""Structural tests of the layer_types-aware loader on a tiny random-weight
qwen3_5 model — no pack files, no 27B load. Verifies block-type assignment,
cache classes, and a forward pass for aligned, non-aligned, all-linear, and
all-full patterns (stock mlx-lm would mis-type all but the first).
"""
import mlx.core as mx
import pytest
from mlx_lm.models.cache import ArraysCache, KVCache

from bonsaifold.loader import _typed_classes

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


L, F = "linear_attention", "full_attention"


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
    with pytest.raises(ValueError):
        build_bad = {
            "model_type": "qwen3_5",
            "text_config": {
                **TINY_TEXT_CONFIG,
                "num_hidden_layers": 3,
                "layer_types": [L, F],  # wrong length
            },
        }
        Model, ModelArgs = _typed_classes(build_bad)
        Model(ModelArgs.from_dict(build_bad))


def test_non_qwen_config_passthrough():
    config = {"model_type": "llama", "hidden_size": 8}
    from mlx_lm.utils import _get_classes

    assert _typed_classes(config) == _get_classes(config=config)
