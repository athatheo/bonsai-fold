"""Layer-types-aware loading for qwen3_5 Bonsai packs, folded or not.

mlx-lm's qwen3_5 assigns block types POSITIONALLY:
`DecoderLayer.is_linear = (layer_idx + 1) % full_attention_interval != 0`,
and never reads `config.layer_types`. Folded packs (blocks dropped or
merged) break that rule for any pattern not aligned to whole groups of 4,
so this loader rebuilds the layer stack from the explicit `layer_types`
list in the pack's config. For unfolded packs the two derivations agree —
loading through here must be a logit-identical no-op (verified in
experiments/runtime_validation once per environment; see LAB_NOTEBOOK).

Only model construction is changed; weight loading, quantization wiring,
sanitize, and cache creation are stock mlx-lm.
"""
import json
from pathlib import Path

import mlx.core as mx
import mlx_lm.models.qwen3_5 as q35
from mlx_lm.utils import _get_classes, load_model, load_tokenizer


class TypedQwen3_5TextModel(q35.Qwen3_5TextModel):
    """Stock text model with block types taken from an explicit list."""

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
            q35.DecoderLayer(args=targs, layer_idx=force_idx[t]) for t in layer_types
        ]
        full = [i for i, t in enumerate(layer_types) if t == "full_attention"]
        linear = [i for i, t in enumerate(layer_types) if t == "linear_attention"]
        # Stock fa_idx/ssm_idx assume both types exist at fixed positions;
        # folded stacks may lack one type entirely, so mask creation below
        # guards on presence instead.
        self.fa_idx = full[0] if full else None
        self.ssm_idx = linear[0] if linear else None

    def __call__(self, inputs, cache=None, input_embeddings=None):
        if input_embeddings is not None:
            hidden_states = input_embeddings
        else:
            hidden_states = self.embed_tokens(inputs)
        if cache is None:
            cache = [None] * len(self.layers)
        fa_mask = (
            q35.create_attention_mask(hidden_states, cache[self.fa_idx])
            if self.fa_idx is not None
            else None
        )
        ssm_mask = (
            q35.create_ssm_mask(hidden_states, cache[self.ssm_idx])
            if self.ssm_idx is not None
            else None
        )
        for layer, c in zip(self.layers, cache):
            mask = ssm_mask if layer.is_linear else fa_mask
            hidden_states = layer(hidden_states, mask=mask, cache=c)
        return self.norm(hidden_states)


def _typed_classes(config):
    Model, ModelArgs = _get_classes(config=config)
    if config.get("model_type") != "qwen3_5":
        return Model, ModelArgs
    tcfg = config.get("text_config", config)
    layer_types = tcfg.get("layer_types")
    if not layer_types:
        return Model, ModelArgs

    class TypedModel(Model):
        def __init__(self, args):
            super().__init__(args)
            targs = q35.TextModelArgs.from_dict(args.text_config)
            self.language_model.model = TypedQwen3_5TextModel(targs, layer_types)

    return TypedModel, ModelArgs


def load_bonsai(pack_dir, lazy=False):
    """Load (model, tokenizer) honoring config layer_types. Use for every
    folded pack; safe (and verified identical) for unfolded ones."""
    pack_dir = Path(pack_dir)
    model, config = load_model(pack_dir, lazy=lazy, get_model_classes=_typed_classes)
    tokenizer = load_tokenizer(pack_dir)
    return model, tokenizer


def layer_types_of(pack_dir):
    config = json.loads((Path(pack_dir) / "config.json").read_text())
    return config.get("text_config", config)["layer_types"]
