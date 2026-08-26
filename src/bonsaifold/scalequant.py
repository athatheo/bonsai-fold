"""Group C (FLAGGED, value-modifying — authorized 2026-08-12): 8-bit
second-level quantization of the f16 scales plane.

This is the ONLY operator exempt from the byte-identity constraint: the
reconstructed scales s' = f16(min + q*step) differ from the originals, so
every effective weight moves by a ~2^-8 relative factor. It must be screened
on the full KL+bench ladder and reported separately from the byte-identical
line (CLAUDE.md, Flagged arm).

Scheme: per-row affine u8 over each scales tensor (rows are K/128 wide,
32-40 entries — finer than QLoRA's 256-groups). Row meta is (min, step) in
f16; reconstruction is computed in f32 and cast to f16 so it is
deterministic from the stored tensors alone. Run under B1 derived kernels
(bonsai_bias_plane=="derived"): biases re-derive from s' in-register, so the
pack stays self-consistent with no separate bias decision.
"""
import copy
from datetime import datetime, timezone

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .fold import write_pack
from .stio import PackReader, is_quantized, weight_base

BITS = 8
QMAX = (1 << BITS) - 1
_ST_DTYPE = {"uint8": "U8", "float16": "F16"}  # numpy dtype name -> st code


def quantize_scales(s, bits=BITS):
    """f16 scales (rows, cols) -> (u8 q, f16 min, f16 step), per-row affine.
    Accepts mx or numpy input (PackReader.read returns numpy). bits<=8;
    codes always stored u8 (sub-8 packing is a shipping decision, not a
    screening one)."""
    qmax = (1 << bits) - 1
    f = mx.array(s).astype(mx.float32)
    lo = f.min(axis=-1, keepdims=True)
    hi = f.max(axis=-1, keepdims=True)
    step = ((hi - lo) / qmax).astype(mx.float16).astype(mx.float32)
    lo = lo.astype(mx.float16).astype(mx.float32)
    q = mx.clip(mx.round((f - lo) / mx.where(step == 0, 1.0, step)), 0, qmax)
    return q.astype(mx.uint8), lo.astype(mx.float16), step.astype(mx.float16)


def reconstruct_scales(q, lo, step):
    """Inverse of quantize_scales; f32 compute, f16 result. Accepts mx or
    numpy input (sanitize receives whatever the weight file loader yields)."""
    q32 = mx.array(q).astype(mx.float32)
    lo32 = mx.array(lo).astype(mx.float32)
    step32 = mx.array(step).astype(mx.float32)
    return (q32 * step32 + lo32).astype(mx.float16)


def roundtrip_scales(s, bits=BITS):
    """s -> reconstruct(quantize(s)); the in-memory screening transform."""
    return reconstruct_scales(*quantize_scales(s, bits))


def quantize_scale_plane(src_pack, dst_pack):
    """Write the Group C pack: every 1-bit tensor's f16 scales become
    (scales_q8 u8, scales_lo f16, scales_step f16); all other tensors are
    raw byte copies. Requires a bias-derived (nobias) source pack so the
    in-register biases follow the reconstructed scales and the pack stays
    self-consistent. Loader reconstructs f16 scales in sanitize (stamp
    bonsai_scale_plane); reconstructed values are exactly apply_roundtrip's.

    Returns (n_tensors, bytes_saved).
    """
    src = PackReader(src_pack)
    if src.config["text_config"].get("bonsai_bias_plane") != "derived":
        raise ValueError("Group C requires a bias-derived (nobias) source pack")
    onebit_scales = {
        weight_base(name) + ".scales"
        for name in src.header
        if is_quantized(name, src.header) and src.quant_meta(name)[1] == 1  # bits
    }

    entries, saved, n_quantized = {}, 0, 0
    for name in src.header:
        begin, end = src.header[name]["data_offsets"]
        if name in onebit_scales:
            q, lo, step = quantize_scales(src.read(name))
            base = name[: -len(".scales")]
            triplet = {}
            for suffix, t in ((".scales_q8", q), (".scales_lo", lo), (".scales_step", step)):
                a = np.array(t)
                triplet[base + suffix] = {
                    "dtype": _ST_DTYPE[a.dtype.name],
                    "shape": list(a.shape),
                    "raw": a.tobytes(),
                }
            gain = (end - begin) - sum(len(e["raw"]) for e in triplet.values())
            if gain > 0:
                entries.update(triplet)
                saved += gain
                n_quantized += 1
                continue
            # unprofitable geometry (very narrow scales): keep the original
        entries[name] = {
            "dtype": src.header[name]["dtype"],
            "shape": src.header[name]["shape"],
            "nbytes": end - begin,
            "raw": (lambda n=name: src.read_raw(n)),
        }

    from .fold import carry_provenance

    config = copy.deepcopy(src.config)
    config["text_config"]["bonsai_scale_plane"] = {"bits": BITS, "grouping": "row"}
    config["bonsai_fold"] = carry_provenance(config, {
        "operation": "quantize_scale_plane",
        "flagged_arm": "group_c_value_modifying",
        "source_pack": str(src.pack_dir),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": "VALUE-MODIFYING (authorized 2026-08-12): scales are 8-bit "
                "reconstructions, NOT byte-identical. Load with load_bonsai.",
    })
    write_pack(dst_pack, entries, config, src.pack_dir)
    return n_quantized, saved


def apply_roundtrip(model, bits=BITS):
    """Mutate every 1-bit quantized module's scales in place to their 8-bit
    round-trip. Under derived kernels this fully defines the Group C model
    (biases follow as f16(-s'/2) in-register). Returns (n_modules,
    max relative error) for the log."""
    n, max_rel = 0, 0.0
    for m in model.modules():
        if isinstance(m, (nn.QuantizedLinear, nn.QuantizedEmbedding)) and m.bits == 1:
            s = m["scales"]
            s2 = roundtrip_scales(s, bits)
            s32 = s.astype(mx.float32)
            rel = mx.abs(s2.astype(mx.float32) - s32) / mx.maximum(mx.abs(s32), 1e-8)
            max_rel = max(max_rel, float(rel.max()))
            m.scales = s2
            n += 1
    return n, max_rel
