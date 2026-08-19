"""Q3: drop-candidate views for DENSE qwen3 packs (Bonsai-8B family).

The 8B sibling is stock qwen3 (all full attention), so none of the hybrid
loader's positional-typing machinery applies: the stock loader is safe, and
views only need the plain qwen3 forward (single mask from cache[0]). Mirrors
loader.LayerSubsetView / SublayerAdapter semantics: zero-copy, source model
never mutated, screening only — ship candidates by writing a real pack with
the fold operators (which are architecture-agnostic on tensor names).
"""
import mlx_lm.models.qwen3 as q3


class DenseSublayerAdapter:
    """TransformerBlock stand-in with attn and/or mlp deleted (residual
    passthrough). Mirrors the block's exact call structure."""

    def __init__(self, layer, drop_attn=False, drop_mlp=False):
        self.layer = layer
        self.drop_attn = drop_attn
        self.drop_mlp = drop_mlp

    def __call__(self, x, mask=None, cache=None):
        l = self.layer
        h = x if self.drop_attn else x + l.self_attn(l.input_layernorm(x), mask, cache)
        if self.drop_mlp:
            return h
        return h + l.mlp(l.post_attention_layernorm(h))


class DenseSubsetView:
    """Zero-copy drop-candidate view over a loaded dense qwen3 model.

    keep: surviving block indices (base numbering). drop_attn/drop_mlp:
    indices (base numbering, must be in keep) whose sublayer is deleted.
    Call signature matches the outer Model (returns logits).
    """

    def __init__(self, model, keep, drop_attn=(), drop_mlp=()):
        inner = model.model
        n = len(inner.layers)
        keep = sorted(set(keep))
        if not keep or not all(0 <= i < n for i in keep):
            raise ValueError(f"keep indices must be within 0..{n-1} and non-empty")
        da, dm = set(drop_attn), set(drop_mlp)
        if not (da <= set(keep) and dm <= set(keep)):
            raise ValueError("drop_attn/drop_mlp indices must be kept blocks")
        overfull = da & dm
        if overfull:
            raise ValueError(
                f"blocks {sorted(overfull)} drop both sublayers; drop the "
                "whole block instead"
            )
        self._model = model
        self.layers = [
            DenseSublayerAdapter(inner.layers[i], i in da, i in dm)
            if (i in da or i in dm) else inner.layers[i]
            for i in keep
        ]
        self.keep = keep

    def __call__(self, inputs, cache=None, input_embeddings=None):
        m = self._model
        inner = m.model
        h = input_embeddings if input_embeddings is not None else inner.embed_tokens(inputs)
        if cache is None:
            cache = [None] * len(self.layers)
        mask = q3.create_attention_mask(h, cache[0])
        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, c)
        h = inner.norm(h)
        if m.args.tie_word_embeddings:
            return inner.embed_tokens.as_linear(h)
        return m.lm_head(h)


def dense_drop_view(model, drop_blocks=(), drop_attn=(), drop_mlp=()):
    """Convenience mirror of loader.sublayer_view for dense packs."""
    n = len(model.model.layers)
    db = set(drop_blocks)
    bad = db & (set(drop_attn) | set(drop_mlp))
    if bad:
        raise ValueError(f"blocks {sorted(bad)} both dropped whole and sublayer-dropped")
    keep = [i for i in range(n) if i not in db]
    return DenseSubsetView(model, keep, drop_attn=drop_attn, drop_mlp=drop_mlp)
