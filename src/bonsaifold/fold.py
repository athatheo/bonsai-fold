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
    config["bonsai_fold"] = {
        **op_stamp,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": "load with bonsaifold.loader.load_bonsai (layer_types-aware)",
    }
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
    from .stio import is_quantized, weight_base

    src = PackReader(src_pack)
    quant_bases = {
        weight_base(n) for n in src.header if is_quantized(n, src.header)
    }
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
    config["bonsai_fold"] = {
        "operation": "strip_bias_plane",
        "source_pack": str(src.pack_dir),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": "biases == f16(-scales/2); load with bonsaifold.loader.load_bonsai",
    }
    write_pack(dst_pack, entries, config, src.pack_dir)
    return len(quant_bases)


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
