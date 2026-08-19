"""Q3 dense-view structure tests (no model load; fake layers)."""
from types import SimpleNamespace

import pytest

from bonsaifold.dense import DenseSublayerAdapter, DenseSubsetView, dense_drop_view


class FakeLayer:
    """Records calls; attn adds 1, mlp adds 10 (distinguishable passthroughs)."""

    def __init__(self):
        self.input_layernorm = lambda x: x
        self.post_attention_layernorm = lambda x: x
        self.self_attn = lambda x, mask, cache: 1
        self.mlp = lambda x: 10

    def __call__(self, x, mask=None, cache=None):
        return x + 1 + 10


def fake_model(n, tie=True):
    inner = SimpleNamespace(
        layers=[FakeLayer() for _ in range(n)],
        embed_tokens=SimpleNamespace(
            __call__=lambda self, x: x, as_linear=lambda h: ("logits", h)
        ),
        norm=lambda h: h,
    )
    return SimpleNamespace(model=inner, args=SimpleNamespace(tie_word_embeddings=tie))


def test_adapter_passthroughs():
    l = FakeLayer()
    assert DenseSublayerAdapter(l)(0) == 11
    assert DenseSublayerAdapter(l, drop_attn=True)(0) == 10
    assert DenseSublayerAdapter(l, drop_mlp=True)(0) == 1
    assert DenseSublayerAdapter(l, drop_attn=True, drop_mlp=True)(0) == 0


def test_subset_view_validation():
    m = fake_model(8)
    with pytest.raises(ValueError, match="within 0..7"):
        DenseSubsetView(m, [0, 8])
    with pytest.raises(ValueError, match="non-empty"):
        DenseSubsetView(m, [])
    with pytest.raises(ValueError, match="must be kept"):
        DenseSubsetView(m, [0, 1], drop_attn=[5])
    with pytest.raises(ValueError, match="whole block instead"):
        DenseSubsetView(m, [0, 1], drop_attn=[1], drop_mlp=[1])


def test_drop_view_composition():
    m = fake_model(8)
    v = dense_drop_view(m, drop_blocks=[3, 5], drop_attn=[0], drop_mlp=[7])
    assert v.keep == [0, 1, 2, 4, 6, 7]
    assert len(v.layers) == 6
    assert isinstance(v.layers[0], DenseSublayerAdapter) and v.layers[0].drop_attn
    assert isinstance(v.layers[-1], DenseSublayerAdapter) and v.layers[-1].drop_mlp
    assert all(not isinstance(l, DenseSublayerAdapter) for l in v.layers[1:-1])
    with pytest.raises(ValueError, match="both dropped"):
        dense_drop_view(m, drop_blocks=[3], drop_attn=[3])
