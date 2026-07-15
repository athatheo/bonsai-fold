"""Format-preserving folding mechanics (Phase 2: block dropping).

Dropping blocks never touches surviving weight values: kept tensors are
copied as raw bytes (byte-identical) under re-indexed names, and only the
config metadata changes. The resulting pack must be loaded with
bonsaifold.loader.load_bonsai — stock mlx-lm assigns block types
positionally and would mis-type most folded stacks.
"""
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from .stio import PackReader, block_owner, reindex_name, write_safetensors

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


def drop_blocks(src_pack, dst_pack, drop):
    """Create a folded pack at dst_pack with the given block indices removed.

    Returns the old->new block map for the surviving blocks.
    """
    src = PackReader(src_pack)
    dst = Path(dst_pack)
    tcfg = src.config["text_config"]
    n = tcfg["num_hidden_layers"]
    drop = sorted(set(drop))
    if any(i < 0 or i >= n for i in drop):
        raise ValueError(f"drop indices {drop} out of range 0..{n-1}")
    keep = [i for i in range(n) if i not in set(drop)]
    if not keep:
        raise ValueError("cannot drop every block")
    block_map = {old: new for new, old in enumerate(keep)}

    dst.mkdir(parents=True, exist_ok=False)

    # ---- tensors: byte-identical raw copy under re-indexed names ----
    entries = {}
    weight_map = {}
    for name in src.header:
        owner = block_owner(name)
        if isinstance(owner, int) and owner in set(drop):
            continue
        new_name = reindex_name(name, block_map)
        info = src.header[name]
        entries[new_name] = {
            "dtype": info["dtype"],
            "shape": info["shape"],
            "raw": src.read_raw(name),
        }
        weight_map[new_name] = "model.safetensors"
    write_safetensors(dst / "model.safetensors", entries)
    total = sum(len(e["raw"]) for e in entries.values())
    (dst / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total}, "weight_map": weight_map})
    )

    # ---- config: trimmed layer_types is the folded ground truth ----
    config = json.loads(json.dumps(src.config))  # deep copy
    new_tcfg = config["text_config"]
    new_tcfg["num_hidden_layers"] = len(keep)
    new_tcfg["layer_types"] = [tcfg["layer_types"][i] for i in keep]
    # per-tensor quantization overrides (none in the shipped pack, but
    # folded/merged packs create them) follow the renaming
    for section in (config, new_tcfg):
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
    config["bonsai_fold"] = {
        "operation": "drop",
        "dropped_blocks": drop,
        "source_pack": str(src.pack_dir),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": "surviving tensors are byte-identical to the source; "
        "load with bonsaifold.loader.load_bonsai (layer_types-aware)",
    }
    (dst / "config.json").write_text(json.dumps(config, indent=2))

    for fname in SIDECAR_FILES:
        p = src.pack_dir / fname
        if p.exists():
            shutil.copy2(p, dst / fname)
    return block_map


def verify_byte_identity(src_pack, dst_pack, block_map):
    """Check every surviving tensor in dst is byte-identical to its source.

    Returns a report dict; report["ok"] is the verdict.
    """
    src, dst = PackReader(src_pack), PackReader(dst_pack)
    inverse = {new: old for old, new in block_map.items()}
    mismatched, missing = [], []
    for new_name in dst.header:
        owner = block_owner(new_name)
        if isinstance(owner, int) and owner not in inverse:
            missing.append(new_name)
            continue
        old_name = (
            reindex_name(new_name, inverse) if isinstance(owner, int) else new_name
        )
        if old_name not in src.header:
            missing.append(new_name)
        elif src.read_raw(old_name) != dst.read_raw(new_name):
            mismatched.append(new_name)
    expected = sum(
        1
        for name in src.header
        if not isinstance(block_owner(name), int) or block_owner(name) in block_map
    )
    report = {
        "tensors_checked": len(dst.header),
        "expected_tensors": expected,
        "count_ok": len(dst.header) == expected,
        "mismatched": mismatched,
        "unmapped_or_missing": missing,
    }
    report["ok"] = report["count_ok"] and not mismatched and not missing
    return report
