"""Format-preserving folding mechanics (Phase 2: block dropping).

Dropping blocks never touches surviving weight values: kept tensors are
copied as raw bytes (byte-identical) under re-indexed names, and only the
config metadata changes. The resulting pack must be loaded with
bonsaifold.loader.load_bonsai — stock mlx-lm assigns block types
positionally and would mis-type most folded stacks.

Pack writing streams one tensor at a time (peak RAM = largest tensor, not
pack size). `write_pack` / `rewrite_folded_config` are shared with the
Phase 3 merge writer, which adds newly-encoded tensors and per-tensor
quantization overrides on top of the same machinery.

For KL screening of drop candidates, do NOT write packs — use
bonsaifold.loader.drop_view (zero-copy). Write a pack for winners only.
"""
import copy
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from .stio import PackReader, reindex_name, write_safetensors

SIDECAR_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "merges.txt",
    "vocab.json",
    "generation_config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "video_preprocessor_config.json",
]


def carry_provenance(src_config, stamp):
    """Chain the source pack's bonsai_fold record into a new op stamp so
    provenance (esp. the Group C flagged_arm marker) survives every
    downstream operator instead of being overwritten. Mutates and returns
    stamp."""
    prev = src_config.get("bonsai_fold")
    if prev:
        history = prev.get("history", []) + [
            {k: v for k, v in prev.items() if k != "history"}
        ]
        stamp["history"] = history
        flagged = prev.get("flagged_arm") or next(
            (h["flagged_arm"] for h in history if "flagged_arm" in h), None
        )
        if flagged and "flagged_arm" not in stamp:
            stamp["flagged_arm"] = flagged
    return stamp


def rewrite_folded_config(config, block_map, op_stamp, extra_quant_overrides=None):
    """Folded-pack config: trimmed layer_types, renamed per-tensor
    quantization overrides, provenance stamp. Returns a new dict."""
    config = copy.deepcopy(config)
    tcfg = config["text_config"]
    keep = sorted(block_map)
    tcfg["num_hidden_layers"] = len(keep)
    tcfg["layer_types"] = [tcfg["layer_types"][i] for i in keep]
    for section in (config, tcfg):
        q = section.get("quantization")
        if not isinstance(q, dict):
            continue
        renamed = {}
        for k, v in q.items():
            if not isinstance(v, dict):
                renamed[k] = v
                continue
            nk = reindex_name(k, block_map)
            if nk is not None:
                renamed[nk] = v
        section["quantization"] = renamed
    if extra_quant_overrides:
        config.setdefault("quantization", {}).update(extra_quant_overrides)
    config["bonsai_fold"] = carry_provenance(config, {
        **op_stamp,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": "load with bonsaifold.loader.load_bonsai (layer_types-aware)",
    })
    return config


def write_pack(dst, entries, config, sidecar_src):
    """Write a model pack: safetensors (streamed), index, config, sidecars."""
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=False)
    write_safetensors(dst / "model.safetensors", entries)
    total = sum(
        e["nbytes"] if "nbytes" in e else len(e["raw"]) for e in entries.values()
    )
    (dst / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": total},
                "weight_map": {name: "model.safetensors" for name in entries},
            }
        )
    )
    (dst / "config.json").write_text(json.dumps(config, indent=2))
    for fname in SIDECAR_FILES:
        p = Path(sidecar_src) / fname
        if p.exists():
            shutil.copy2(p, dst / fname)


def drop_blocks(src_pack, dst_pack, drop):
    """Create a folded pack at dst_pack with the given block indices removed.

    Returns the old->new block map for the surviving blocks.
    """
    src = PackReader(src_pack)
    n = src.config["text_config"]["num_hidden_layers"]
    drop = sorted(set(drop))
    if any(i < 0 or i >= n for i in drop):
        raise ValueError(f"drop indices {drop} out of range 0..{n-1}")
    keep = [i for i in range(n) if i not in set(drop)]
    if not keep:
        raise ValueError("cannot drop every block")
    block_map = {old: new for new, old in enumerate(keep)}

    entries = {}
    for name in src.header:
        new_name = reindex_name(name, block_map)
        if new_name is None:
            continue
        begin, end = src.header[name]["data_offsets"]
        entries[new_name] = {
            "dtype": src.header[name]["dtype"],
            "shape": src.header[name]["shape"],
            "nbytes": end - begin,
            "raw": (lambda n=name: src.read_raw(n)),  # streamed at write time
        }
    config = rewrite_folded_config(
        src.config,
        block_map,
        {
            "operation": "drop",
            "dropped_blocks": drop,
            "source_pack": str(src.pack_dir),
        },
    )
    write_pack(dst_pack, entries, config, src.pack_dir)
    return block_map


def strip_bias_plane(src_pack, dst_pack):
    """B1 pack-v2: drop every quantized tensor's biases plane (~420 MB).

    Phase 0 proved biases == f16(-scales/2) exhaustively (3 f16-subnormal
    edge groups, dequant error <= 6e-08), so the plane is derivable. The
    loader re-materializes it at load when the config stamp below is
    present; the eventual kernel change computes it in-register instead.
    All kept tensors are raw byte copies.
    """
    import numpy as np

    from .stio import is_quantized, weight_base

    src = PackReader(src_pack)
    quant_bases = {
        weight_base(n) for n in src.header if is_quantized(n, src.header)
    }
    # refuse to strip a plane that is not derivable: every bias tensor must
    # satisfy b == f16(-s/2) exactly (a promoted 2-bit block stores b == -s
    # and would be silently corrupted by the loader's -s/2 re-materialization)
    for base in sorted(quant_bases):
        if base + ".biases" not in src.header:
            continue
        s = src.read(base + ".scales")
        b = src.read(base + ".biases")
        expected = (-(s.astype(np.float32) / 2)).astype(s.dtype)
        if not np.array_equal(b, expected):
            raise ValueError(
                f"{base}: biases != f16(-scales/2); refusing to strip a "
                "non-derivable bias plane"
            )
    entries = {}
    for name in src.header:
        base = name[: -len(".biases")] if name.endswith(".biases") else None
        if base in quant_bases:
            continue  # derived at load
        begin, end = src.header[name]["data_offsets"]
        entries[name] = {
            "dtype": src.header[name]["dtype"],
            "shape": src.header[name]["shape"],
            "nbytes": end - begin,
            "raw": (lambda n=name: src.read_raw(n)),
        }
    config = copy.deepcopy(src.config)
    config["text_config"]["bonsai_bias_plane"] = "derived"
    config["bonsai_fold"] = carry_provenance(config, {
        "operation": "strip_bias_plane",
        "source_pack": str(src.pack_dir),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": "biases == f16(-scales/2); load with bonsaifold.loader.load_bonsai",
    })
    write_pack(dst_pack, entries, config, src.pack_dir)
    return len(quant_bases)


def trim_lm_head(src_pack, dst_pack, keep):
    """A4: delete lm_head rows outside `keep` (sorted token ids).

    Surviving rows are byte-identical (contiguous row slices of the packed
    weight and scales/biases). The config records the kept token ids; the
    loader builds a narrow head and scatters logits back to full-vocab
    positions, so samplers and eos handling are unchanged.
    """
    from .stio import weight_base

    src = PackReader(src_pack)
    if src.config["text_config"].get("bonsai_scale_plane"):
        # the row-slicer below only knows .weight/.scales/.biases; a Group C
        # pack's scales_q8/_lo/_step triplet would be copied untrimmed and
        # the pack would fail strict load much later
        raise ValueError("trim_lm_head does not support scale-quantized "
                         "(Group C) packs; trim first, then quantize_scale_plane")
    keep = sorted(set(keep))
    head = "language_model.lm_head"
    entries = {}
    for name in src.header:
        base = weight_base(name) or (
            name[: -len(".scales")] if name.endswith(".scales") else
            name[: -len(".biases")] if name.endswith(".biases") else name
        )
        if base == head:
            import numpy as np

            arr = src.read(name)
            rows = arr[np.array(keep)]
            entries[name] = {
                "dtype": src.header[name]["dtype"],
                "shape": list(rows.shape),
                "raw": rows.tobytes(),
            }
            continue
        begin, end = src.header[name]["data_offsets"]
        entries[name] = {
            "dtype": src.header[name]["dtype"],
            "shape": src.header[name]["shape"],
            "nbytes": end - begin,
            "raw": (lambda n=name: src.read_raw(n)),
        }
    config = copy.deepcopy(src.config)
    config["text_config"]["bonsai_lm_head_keep"] = keep
    config["bonsai_fold"] = carry_provenance(config, {
        "operation": "trim_lm_head",
        "kept_rows": len(keep),
        "source_pack": str(src.pack_dir),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": "load with bonsaifold.loader.load_bonsai (logit scatter-back)",
    })
    write_pack(dst_pack, entries, config, src.pack_dir)
    return len(keep)


def drop_sublayers_in_pack(src_pack, dst_pack, drop_attn=(), drop_mlp=()):
    """A1 as a pack operation: delete the tensors of the named sublayers and
    record the drops in config; the loader builds those blocks without the
    corresponding submodule. Block indices refer to THIS pack's numbering.
    """
    src = PackReader(src_pack)
    da, dm = sorted(set(drop_attn)), sorted(set(drop_mlp))

    def dropped(name):
        parts = name.split(".")
        if "layers" not in parts or name.startswith("vision_tower."):
            return False
        i = int(parts[parts.index("layers") + 1])
        rest = ".".join(parts[parts.index("layers") + 2 :])
        if i in da and (
            rest.startswith("linear_attn.") or rest.startswith("self_attn.")
            or rest.startswith("input_layernorm.")
        ):
            return True
        if i in dm and (
            rest.startswith("mlp.") or rest.startswith("post_attention_layernorm.")
        ):
            return True
        return False

    entries = {}
    for name in src.header:
        if dropped(name):
            continue
        begin, end = src.header[name]["data_offsets"]
        entries[name] = {
            "dtype": src.header[name]["dtype"],
            "shape": src.header[name]["shape"],
            "nbytes": end - begin,
            "raw": (lambda n=name: src.read_raw(n)),
        }
    config = copy.deepcopy(src.config)
    sub = config["text_config"].setdefault("bonsai_sublayer_drops", {"attn": [], "mlp": []})
    sub["attn"] = sorted(set(sub["attn"]) | set(da))
    sub["mlp"] = sorted(set(sub["mlp"]) | set(dm))
    config["bonsai_fold"] = carry_provenance(config, {
        "operation": "drop_sublayers",
        "drop_attn": da,
        "drop_mlp": dm,
        "source_pack": str(src.pack_dir),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": "load with bonsaifold.loader.load_bonsai",
    })
    write_pack(dst_pack, entries, config, src.pack_dir)
    return len(entries)


def verify_byte_identity(src_pack, dst_pack, block_map):
    """Check dst holds exactly the mapped tensors, each byte-identical to
    its source. Returns a report dict; report["ok"] is the verdict."""
    src, dst = PackReader(src_pack), PackReader(dst_pack)
    expected = {}  # dst name -> src name
    for name in src.header:
        new = reindex_name(name, block_map)
        if new is not None:
            expected[new] = name
    missing = sorted(set(expected) - set(dst.header))
    extra = sorted(set(dst.header) - set(expected))
    mismatched = [
        n
        for n in sorted(set(dst.header) & set(expected))
        if src.read_raw(expected[n]) != dst.read_raw(n)
    ]
    return {
        "tensors_checked": len(dst.header),
        "missing": missing,
        "extra": extra,
        "mismatched": mismatched,
        "ok": not (missing or extra or mismatched),
    }
