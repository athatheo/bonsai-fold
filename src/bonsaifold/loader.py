"""Layer-types-aware loading for qwen3_5 Bonsai packs, folded or not.

mlx-lm's qwen3_5 assigns block types POSITIONALLY:
`DecoderLayer.is_linear = (layer_idx + 1) % full_attention_interval != 0`,
and never reads `config.layer_types`. Folded packs (blocks dropped or
merged) break that rule for any pattern not aligned to whole groups of 4,
so this loader rebuilds the layer stack from the explicit `layer_types`
list in the pack's config. For unfolded packs the two derivations agree —
loading through here must be a logit-identical no-op (verified in
experiments/runtime_validation once per environment; see LAB_NOTEBOOK).

Three public entry points:
  load_bonsai(pack_dir)          — load a pack (model, tokenizer)
  layer_subset_view(model, keep) — zero-copy drop-candidate view for sweeps
                                   (weights shared; no pack written)
  block taps                     — model.set_block_tap(fn) for per-block
                                   residual-stream recording without
                                   touching the module tree

Only model construction is changed; weight loading, quantization wiring,
sanitize, and cache creation are stock mlx-lm.
"""
from pathlib import Path

import mlx_lm.models.qwen3_5 as q35
from mlx_lm.models.cache import ArraysCache, KVCache
from mlx_lm.utils import _get_classes, load_config, load_model, load_tokenizer


def _first_index(layers, linear):
    return next((i for i, l in enumerate(layers) if l.is_linear == linear), None)


def _hybrid_forward(tm_or_view, inputs, cache, input_embeddings=None):
    """Shared forward over a hybrid layer stack with explicit types.

    `tm_or_view` provides embed_tokens, norm, layers, fa_idx, ssm_idx, and
    optionally block_tap(i, h_in, h_out). Unlike stock qwen3_5, mask
    creation guards on type presence (folded stacks may lack one type).
    """
    s = tm_or_view
    if input_embeddings is not None:
        h = input_embeddings
    else:
        h = s.embed_tokens(inputs)
    if cache is None:
        cache = [None] * len(s.layers)
    fa_mask = (
        q35.create_attention_mask(h, cache[s.fa_idx]) if s.fa_idx is not None else None
    )
    ssm_mask = (
        q35.create_ssm_mask(h, cache[s.ssm_idx]) if s.ssm_idx is not None else None
    )
    tap = getattr(s, "block_tap", None)
    for i, (layer, c) in enumerate(zip(s.layers, cache)):
        h_out = layer(h, mask=ssm_mask if layer.is_linear else fa_mask, cache=c)
        if tap is not None:
            tap(i, h, h_out)
        h = h_out
    return s.norm(h)


class TypedDecoderLayer(q35.DecoderLayer):
    """DecoderLayer that tolerates deleted sublayers (pack-persisted A1
    drops): a missing attn/mlp submodule is a residual passthrough. Stays a
    real nn.Module so surviving parameters remain in the tree for loading."""

    def __call__(self, x, mask=None, cache=None):
        if hasattr(self, "linear_attn") or hasattr(self, "self_attn"):
            attn = self.linear_attn if self.is_linear else self.self_attn
            h = x + attn(self.input_layernorm(x), mask, cache)
        else:
            h = x
        if hasattr(self, "mlp"):
            return h + self.mlp(self.post_attention_layernorm(h))
        return h


class TypedQwen3_5TextModel(q35.Qwen3_5TextModel):
    """Stock text model with block types taken from an explicit list."""

    block_tap = None  # optional fn(i, h_in, h_out); see set_block_tap

    def __init__(self, targs, layer_types):
        if len(layer_types) != targs.num_hidden_layers:
            raise ValueError(
                f"layer_types has {len(layer_types)} entries, "
                f"num_hidden_layers is {targs.num_hidden_layers}"
            )
        super().__init__(targs)
        # layer_idx is only consumed by DecoderLayer's positional type rule,
        # so forcing an index of known parity forces the type.
        force_idx = {
            "linear_attention": 0,
            "full_attention": targs.full_attention_interval - 1,
        }
        self.layers = [
            TypedDecoderLayer(args=targs, layer_idx=force_idx[t]) for t in layer_types
        ]
        self.fa_idx = _first_index(self.layers, linear=False)
        self.ssm_idx = _first_index(self.layers, linear=True)

    def __call__(self, inputs, cache=None, input_embeddings=None):
        return _hybrid_forward(self, inputs, cache, input_embeddings)


class TypedModel(q35.Model):
    def __init__(self, args):
        super().__init__(args)
        tcfg = args.text_config
        targs = q35.TextModelArgs.from_dict(tcfg)
        self._derive_bias_plane = tcfg.get("bonsai_bias_plane") == "derived"
        self._scale_plane = tcfg.get("bonsai_scale_plane")
        self.language_model.model = TypedQwen3_5TextModel(
            targs, tcfg["layer_types"]
        )
        # pack-persisted sublayer drops: build those blocks WITHOUT the
        # deleted submodule (strict weight loading then expects nothing for it)
        sub = tcfg.get("bonsai_sublayer_drops", {"attn": [], "mlp": []})
        tm = self.language_model.model
        for i in sub["attn"]:
            l = tm.layers[i]
            delattr(l, "linear_attn" if l.is_linear else "self_attn")
            delattr(l, "input_layernorm")
        for i in sub["mlp"]:
            delattr(tm.layers[i], "mlp")
            delattr(tm.layers[i], "post_attention_layernorm")
        if sub["attn"] or sub["mlp"]:
            # TypedDecoderLayer skips deleted submodules in forward; only the
            # mask anchors need recomputing over blocks whose attn survives
            da = set(sub["attn"])
            tm.fa_idx = next(
                (i for i, l in enumerate(tm.layers) if not l.is_linear and i not in da), None
            )
            tm.ssm_idx = next(
                (i for i, l in enumerate(tm.layers) if l.is_linear and i not in da), None
            )
        # pack-persisted lm_head trim: narrow head + logit scatter-back
        self._lm_head_keep = tcfg.get("bonsai_lm_head_keep")
        if self._lm_head_keep is not None:
            import mlx.core as mx
            import mlx.nn as nn

            self.language_model.lm_head = nn.Linear(
                targs.hidden_size, len(self._lm_head_keep), bias=False
            )
            self._keep_idx = mx.array(self._lm_head_keep)
            self._full_vocab = targs.vocab_size

    def __call__(self, inputs, cache=None, input_embeddings=None):
        out = super().__call__(inputs, cache=cache, input_embeddings=input_embeddings)
        if getattr(self, "_lm_head_keep", None) is not None:
            import mlx.core as mx

            full = mx.full((*out.shape[:-1], self._full_vocab), -mx.inf, dtype=out.dtype)
            out = mx.put_along_axis(
                full,
                mx.broadcast_to(self._keep_idx, (*out.shape[:-1], len(self._lm_head_keep))),
                out,
                axis=-1,
            )
        return out

    def sanitize(self, weights):
        weights = super().sanitize(weights)
        if self._scale_plane:
            # Group C (FLAGGED value-modifying): reconstruct f16 scales from
            # the 8-bit plane. Must precede bias materialization — derived
            # biases follow the reconstructed scales.
            from .scalequant import reconstruct_scales

            for k in list(weights):
                if k.endswith(".scales_q8"):
                    base = k[: -len("_q8")]
                    weights[base] = reconstruct_scales(
                        weights[k],
                        weights.pop(base + "_lo"),
                        weights.pop(base + "_step"),
                    )
                    del weights[k]
        if self._derive_bias_plane:
            # B1 pack-v2: the biases plane was stripped (redundant — Phase 0
            # proved biases == f16(-scales/2)); re-materialize it at load
            import mlx.core as mx

            for k in list(weights):
                if k.endswith(".scales") and k[: -len(".scales")] + ".biases" not in weights:
                    s = weights[k]
                    weights[k[: -len(".scales")] + ".biases"] = (
                        -(s.astype(mx.float32) / 2)
                    ).astype(s.dtype)
        return weights

    def set_block_tap(self, fn):
        """Install fn(i, h_in, h_out), called on every block's residual
        stream during forward. Pass None to remove."""
        self.language_model.model.block_tap = fn


def _typed_classes(config):
    Model, ModelArgs = _get_classes(config=config)
    tcfg = config.get("text_config", config)
    if config.get("model_type") != "qwen3_5" or not tcfg.get("layer_types"):
        return Model, ModelArgs
    return TypedModel, ModelArgs


def _apply_derived_kernels(model):
    """B1 kernel stage: route every 1-bit affine module through the
    in-register bias-derivation kernels (mode="affine-derived", bit-identical
    to affine — parity battery 2026-08-17). The stock quantized modules fetch
    biases via self.get("biases") and pass self.mode through, so deleting the
    biases parameter and flipping mode is the whole integration; under a lazy
    load the sanitize-built bias graphs are dropped before ever evaluating."""
    import mlx.nn as nn

    for m in model.modules():
        if (
            isinstance(m, (nn.QuantizedLinear, nn.QuantizedEmbedding))
            and m.mode == "affine"
            and m.bits == 1
        ):
            if "biases" in m:
                delattr(m, "biases")
            m.mode = "affine-derived"


def load_bonsai(pack_dir, lazy=False, derived_kernels=None):
    """Load (model, tokenizer) honoring config layer_types. Use for every
    folded pack; safe (and verified identical) for unfolded ones.

    derived_kernels: None (default) enables the B1 in-kernel bias derivation
    automatically for packs stamped bonsai_bias_plane=="derived"; True enables
    it and errors when the pack is not stamped; False forces the
    materialized-bias path (parity/debug)."""
    import mlx.core as mx

    pack_dir = Path(pack_dir)
    config = load_config(pack_dir)
    stamped = config.get("text_config", config).get("bonsai_bias_plane") == "derived"
    if derived_kernels is True and not stamped:
        raise ValueError(f"{pack_dir} is not stamped bonsai_bias_plane=derived")
    use_derived = stamped if derived_kernels is None else derived_kernels
    # derived mode forces a lazy load: _apply_derived_kernels drops the
    # sanitize-materialized bias arrays before they are ever evaluated
    model, _ = load_model(
        pack_dir, lazy=lazy or use_derived, get_model_classes=_typed_classes
    )
    if use_derived:
        _apply_derived_kernels(model)
        if not lazy:
            mx.eval(model.parameters())
    tokenizer = load_tokenizer(pack_dir)
    return model, tokenizer


class LayerSubsetView:
    """Zero-copy drop-candidate view over a loaded model.

    Shares the source model's weight arrays; only the layer list and
    type-index bookkeeping are new. The source model is never mutated, so
    it stays valid as the sweep's reference. For KL screening only — to
    ship a candidate, write a real pack with fold.drop_blocks.
    """

    block_tap = None

    def __init__(self, model, keep):
        tm = model.language_model.model
        n = len(tm.layers)
        if not keep or any(i < 0 or i >= n for i in keep) or len(set(keep)) != len(keep):
            raise ValueError(f"invalid keep list for {n}-layer model: {keep}")
        self._lm = model.language_model
        self.embed_tokens = tm.embed_tokens
        self.norm = tm.norm
        self.layers = [tm.layers[i] for i in keep]
        self.fa_idx = _first_index(self.layers, linear=False)
        self.ssm_idx = _first_index(self.layers, linear=True)

    def make_cache(self):
        return [ArraysCache(size=2) if l.is_linear else KVCache() for l in self.layers]

    def __call__(self, inputs, cache=None, input_embeddings=None):
        """Token ids -> logits (mirrors Model.__call__ through the head)."""
        h = _hybrid_forward(self, inputs, cache, input_embeddings)
        if self._lm.args.tie_word_embeddings:
            return self.embed_tokens.as_linear(h)
        return self._lm.lm_head(h)


def layer_subset_view(model, keep):
    return LayerSubsetView(model, keep)


def drop_view(model, drop):
    """Convenience: view of `model` with the given block indices removed."""
    n = len(model.language_model.model.layers)
    dropped = set(drop)
    if not dropped or not dropped <= set(range(n)):
        # a silently-ignored typo here would screen the unmodified reference
        # and report KL ~ 0 for a nonexistent block
        raise ValueError(f"drop indices {sorted(dropped)} not all in 0..{n-1}")
    return LayerSubsetView(model, [i for i in range(n) if i not in dropped])


class SublayerAdapter:
    """Wraps a DecoderLayer, optionally skipping its attention or MLP
    sublayer (residual passthrough). Weight-sharing, non-mutating."""

    def __init__(self, layer, drop_attn=False, drop_mlp=False):
        self.layer = layer
        self.drop_attn = drop_attn
        self.drop_mlp = drop_mlp

    @property
    def is_linear(self):
        return self.layer.is_linear

    def __call__(self, x, mask=None, cache=None):
        l = self.layer
        if self.drop_attn:
            h = x
        else:
            attn = l.linear_attn if l.is_linear else l.self_attn
            h = x + attn(l.input_layernorm(x), mask, cache)
        if self.drop_mlp:
            return h
        return h + l.mlp(l.post_attention_layernorm(h))


def sublayer_view(model, drop_attn=(), drop_mlp=(), drop_blocks=()):
    """Zero-copy view with individual attention/MLP sublayers removed, and
    optionally whole blocks (all indices refer to ORIGINAL block numbers).

    Finer-grained than drop_view: a block whose attention is dropped keeps
    its MLP and vice versa. Dropping BOTH sublayers of a block equals
    dropping the block (kept for composition sweeps; verified in tests).
    Mask anchors (fa_idx/ssm_idx) are recomputed over surviving blocks
    whose attention survives.
    """
    tm = model.language_model.model
    n = len(tm.layers)
    da, dm, db = set(drop_attn), set(drop_mlp), set(drop_blocks)
    all_idx = da | dm | db
    if not all_idx or not all_idx <= set(range(n)):
        raise ValueError(f"drop indices {sorted(all_idx)} not all in 0..{n-1}")
    if db & (da | dm):
        raise ValueError(f"blocks {sorted(db & (da | dm))} both whole-dropped and sublayer-dropped")
    keep = [i for i in range(n) if i not in db]
    view = LayerSubsetView(model, keep)
    view.layers = [
        SublayerAdapter(tm.layers[i], i in da, i in dm)
        if (i in da or i in dm)
        else tm.layers[i]
        for i in keep
    ]
    attn_alive = [
        (pos, tm.layers[i]) for pos, i in enumerate(keep) if i not in da
    ]
    view.fa_idx = next((p for p, l in attn_alive if not l.is_linear), None)
    view.ssm_idx = next((p for p, l in attn_alive if l.is_linear), None)
    return view


def layer_types_of(pack_dir):
    config = load_config(Path(pack_dir))
    return config.get("text_config", config)["layer_types"]
