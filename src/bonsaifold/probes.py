"""Frozen probe/calibration set IO shared by all harnesses.

Schema: {"meta": {...}, "items": [{"id", "category", "source",
"token_ids": [int, ...]}, ...]}. Reasoning items carry
"prompt_token_ids" and a null "token_ids" until the generation stage
fills them. (That stage also stamps meta["complete"], but it is
informational — validation here re-derives completeness from the items,
and sets that never go through generation don't carry the key.)

Every consumer must call load_probe_set BEFORE loading a model: a
half-built set (null token_ids) should fail in milliseconds, not after a
27B load.
"""
import hashlib
import json
from pathlib import Path


def load_probe_set(path, require_complete=True):
    data = json.loads(Path(path).read_text())
    if "items" not in data or not data["items"]:
        raise ValueError(f"{path}: no items")
    if require_complete:
        pending = [it["id"] for it in data["items"] if not it.get("token_ids")]
        if pending:
            raise ValueError(
                f"{path}: {len(pending)} items have no token_ids (e.g. "
                f"{pending[:3]}); run build_calibration.py --stage completions first"
            )
    return data


def probe_fingerprint(data):
    """Stable identity of a frozen set (for checkpoint-resume safety):
    hashes item ids, lengths, and boundary tokens."""
    h = hashlib.sha256()
    for it in data["items"]:
        ids = it.get("token_ids") or []
        h.update(
            f"{it['id']}:{len(ids)}:{ids[:4]}:{ids[-4:]}".encode()
        )
    return h.hexdigest()[:16]
