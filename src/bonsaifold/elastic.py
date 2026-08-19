"""Q4: depth-on-demand from one pack (elastic-depth family).

The spec (experiments/elastic/elastic_spec.json, generated from measured
screens/benches) defines a nested drop order whose anchor prefixes are the
benched configs. A deployment picks the longest prefix inside its byte
budget and applies it as a zero-copy view over the loaded nobias pack —
byte-identical by construction (deletion only), no extra packs shipped.

    model, tok = load_bonsai("models/Bonsai-27B-mlx-1bit-nobias")
    view = elastic_view(model, budget_mb=300)   # -> folded-709 config

Every index in the spec order is a BASE-pack block number, so the model
handed to elastic_view must be the unfolded base pack — see check_base_pack.
"""
import json
from pathlib import Path

from .loader import sublayer_view

DEFAULT_SPEC = (
    Path(__file__).resolve().parents[2] / "experiments/elastic/elastic_spec.json"
)


def load_spec(spec_path=None):
    """The committed elastic spec (or an explicit path to one)."""
    return json.loads(Path(spec_path or DEFAULT_SPEC).read_text())


def prefix_for_budget(spec, budget_mb):
    """Longest op prefix with cumulative structural MB <= budget_mb."""
    ops = []
    for step in spec["steps"]:
        if step["cum_structural_mb"] > budget_mb:
            break
        ops.append(step["op"])
    return ops


def split_ops(ops):
    """Op tokens -> (drop_blocks, drop_attn, drop_mlp) index lists."""
    kinds = {"b": [], "a": [], "m": []}
    for op in ops:
        kinds[op[0]].append(int(op[1:]))
    return kinds["b"], kinds["a"], kinds["m"]


def check_base_pack(model, spec):
    """Refuse anything that is not the unfolded base pack.

    Spec ops are BASE-pack block numbers. On an already-folded pack the
    surviving blocks have been renumbered, so the same indices name
    different blocks and would silently build a config nobody measured
    (e.g. b16 on folded-709 hits original block 19). Depth is the cheap
    discriminator: only the base pack has base_num_blocks blocks.
    """
    lm = getattr(model, "language_model", None)
    if lm is None:
        raise ValueError(
            f"elastic_view needs a model loaded from {spec['base_pack']}, "
            "not a view or a bare language model"
        )
    n = len(lm.model.layers)
    if n != spec["base_num_blocks"]:
        raise ValueError(
            f"elastic_view expects the {spec['base_num_blocks']}-block base "
            f"pack ({spec['base_pack']}), got a {n}-block model; spec ops are "
            "base-pack block numbers and mean something else on a folded pack"
        )


def elastic_view(model, budget_mb=None, steps=None, spec_path=None):
    """Zero-copy view of the longest measured-order prefix within budget_mb
    (or exactly the first `steps` ops). The source model stays valid.

    `model` must be the unfolded base pack (see check_base_pack).
    """
    if (budget_mb is None) == (steps is None):
        raise ValueError("pass exactly one of budget_mb / steps")
    spec = load_spec(spec_path)
    order = spec["order"]
    if steps is not None and not 1 <= steps <= len(order):
        raise ValueError(f"steps must be in 1..{len(order)}, got {steps}")
    check_base_pack(model, spec)

    ops = order[:steps] if steps is not None else prefix_for_budget(spec, budget_mb)
    if not ops:
        raise ValueError(
            f"budget_mb={budget_mb} is below the first tier "
            f"({spec['steps'][0]['cum_structural_mb']} MB); nothing to drop"
        )
    blocks, attn, mlp = split_ops(ops)
    return sublayer_view(model, drop_attn=attn, drop_mlp=mlp, drop_blocks=blocks)
