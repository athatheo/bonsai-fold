"""Phase 3: format-native merging of an adjacent-ish same-type block pair.

Host = i (keeps its position), absorbed = j (removed). Survivors stay
byte-identical; merged tensors are the only NEW bytes, built by closed-form
rules — no training, no calibration, never mx.quantize. Operators are
defined only on 1-bit g128 sources (the shipped Bonsai pack convention
w = s*q + b, q in {0,1}, biases == f16(-scales/2)):

  sign_election: binary pair -> binary block. W_avg = (s_a*t_a + s_b*t_b)/2;
      per-group scale = absmean(W_avg); sign = sign(W_avg), ties -> t_a.
  promotion:     binary pair -> ONE 2-bit block. Agreement keeps the shared
      sign, disagreement abstains to 0 (codes {0,1,2}, code 3 unreachable,
      biases == -scales exactly); s' = (s_a + s_b)/2.

FP params (A_log, dt_bias, conv1d, norm gains) merge by fp32 arithmetic
mean cast to f16 — for A_log that IS the geometric mean of the decay rates
(log-space), the parameterization where averaging is principled; both means
bias one-sidedly toward slower forgetting (watch long context downstream).
Both kernels are symmetric elementwise combines (no donor-row selection),
so head/channel index k always pairs with index k across the two blocks.

Load merged packs with bonsaifold.loader.load_bonsai only; speculative
decoding stays off. fold.verify_byte_identity fails on merge packs by
design — use verify_merge.
"""
import numpy as np

from .fold import rewrite_folded_config, write_pack
from .stio import (
    PackReader,
    block_owner,
    infer_bits,
    is_quantized,
    pack_codes,
    reindex_name,
    unpack_codes,
    weight_base,
)

GROUP_SIZE = 128


def _validate_pair(codes_a, scales_a, biases_a, codes_b, scales_b, biases_b, group_size):
    if codes_a.shape != codes_b.shape:
        raise ValueError(f"codes shape mismatch {codes_a.shape} != {codes_b.shape}")
    shapes = {scales_a.shape, scales_b.shape, biases_a.shape, biases_b.shape}
    if len(shapes) != 1:
        raise ValueError(f"scales/biases shape mismatch: {shapes}")
    if codes_a.shape[-1] != scales_a.shape[-1] * group_size:
        raise ValueError(
            f"codes cols {codes_a.shape[-1]} != groups {scales_a.shape[-1]} * {group_size}"
        )
    for c in (codes_a, codes_b):
        if c.max() > 1:
            raise ValueError("source codes not 1-bit (max code > 1)")
    for s, b in ((scales_a, biases_a), (scales_b, biases_b)):
        expect = (-(s.astype(np.float32)) / 2).astype(np.float16)
        if not np.array_equal(b, expect):
            raise ValueError("source biases invariant violated (biases != f16(-scales/2))")


def _signs_and_mags(codes, scales, group_size):
    t = (2 * codes.astype(np.int8) - 1)  # {0,1} -> {-1,+1}
    s_g = scales.astype(np.float32) / 2  # s_mlx = 2*s_g
    return t, s_g.repeat(group_size, axis=-1)


def sign_election(codes_a, scales_a, biases_a, codes_b, scales_b, biases_b,
                  group_size=GROUP_SIZE):
    """Binary pair -> binary block. Inputs: UNPACKED codes [rows, cols] in
    {0,1}, f16 scales/biases [rows, cols//group_size]. Returns
    (codes {0,1} uint8, scales f16, biases f16) in the pack's 1-bit affine
    convention (biases == f16(-scales/2) bit-exactly by construction)."""
    _validate_pair(codes_a, scales_a, biases_a, codes_b, scales_b, biases_b, group_size)
    t_a, m_a = _signs_and_mags(codes_a, scales_a, group_size)
    t_b, m_b = _signs_and_mags(codes_b, scales_b, group_size)
    w = (m_a * t_a + m_b * t_b) * 0.5  # float32
    s_new_g = np.abs(w).reshape(*w.shape[:-1], -1, group_size).mean(-1)
    # ties (exact zeros: equal magnitudes, opposite signs) break toward t_a
    t_new = np.where(w > 0, 1, np.where(w < 0, -1, t_a))
    codes_new = ((t_new + 1) >> 1).astype(np.uint8)
    scales_new = (2.0 * s_new_g).astype(np.float16)
    # biases FROM the rounded f16 scales, so the census invariant holds bit-exactly
    biases_new = (-(scales_new.astype(np.float32)) / 2).astype(np.float16)
    return codes_new, scales_new, biases_new


def promotion(codes_a, scales_a, biases_a, codes_b, scales_b, biases_b,
              group_size=GROUP_SIZE):
    """Binary pair -> ONE 2-bit block. Agreement keeps the sign, disagreement
    abstains: q' = q_a + q_b lands in {0,1,2} with 1 = abstain and code 3
    unreachable. Returns (codes uint8, scales f16, biases f16) in the
    verified 2-bit convention (biases == -scales exactly)."""
    _validate_pair(codes_a, scales_a, biases_a, codes_b, scales_b, biases_b, group_size)
    codes_new = (codes_a + codes_b).astype(np.uint8)
    scales_new = (
        (scales_a.astype(np.float32) + scales_b.astype(np.float32)) / 4
    ).astype(np.float16)  # stored 2-bit scale IS s' = (s_a_g + s_b_g)/2
    biases_new = np.negative(scales_new)  # f16 sign flip, exact
    return codes_new, scales_new, biases_new


OPERATORS = {"sign_election": (sign_election, 1), "promotion": (promotion, 2)}


def merge_fp_mean(a, b):
    """fp32 arithmetic mean, cast f16 — the merge rule every FP param class
    reduces to (see module docstring)."""
    return ((a.astype(np.float32) + b.astype(np.float32)) / 2).astype(np.float16)


def _partner(name, i, j):
    parts = name.split(".")
    k = parts.index("layers") + 1
    assert int(parts[k]) == i
    return ".".join(parts[:k] + [str(j)] + parts[k + 1 :])


def _merge_quant_tensor(src, base_i, base_j, kernel, out_bits, chunk_rows):
    """Chunked over rows (the g128 grid tiles the column axis, so any row
    partition is group-aligned). Returns payload dicts for the triple."""
    w_a, w_b = src.read(base_i + ".weight"), src.read(base_j + ".weight")
    s_a, s_b = src.read(base_i + ".scales"), src.read(base_j + ".scales")
    b_a, b_b = src.read(base_i + ".biases"), src.read(base_j + ".biases")
    codes_out, scales_out, biases_out = [], [], []
    for r in range(0, w_a.shape[0], chunk_rows):
        sl = slice(r, r + chunk_rows)
        cn, sn, bn = kernel(
            unpack_codes(w_a[sl], 1), s_a[sl], b_a[sl],
            unpack_codes(w_b[sl], 1), s_b[sl], b_b[sl],
        )
        codes_out.append(pack_codes(cn, out_bits))
        scales_out.append(sn)
        biases_out.append(bn)
    packed = np.concatenate(codes_out)
    scales = np.concatenate(scales_out)
    biases = np.concatenate(biases_out)
    return {
        base_i + ".weight": {"dtype": "U32", "shape": packed.shape,
                             "raw": np.ascontiguousarray(packed).tobytes()},
        base_i + ".scales": {"dtype": "F16", "shape": scales.shape,
                             "raw": np.ascontiguousarray(scales).tobytes()},
        base_i + ".biases": {"dtype": "F16", "shape": biases.shape,
                             "raw": np.ascontiguousarray(biases).tobytes()},
    }


def _block_tensor_kinds(src, i):
    """Split block i's tensor names into quantized bases and FP tensors."""
    bases, fp = [], []
    for name in src.header:
        if block_owner(name) != i:
            continue
        base = weight_base(name)
        if base is not None and is_quantized(name, src.header):
            bases.append(base)
        elif name.endswith((".scales", ".biases")) and (
            name.rsplit(".", 1)[0] + ".weight" in src.header
        ):
            continue  # member of a quantized triple, handled via its base
        else:
            fp.append(name)
    return bases, fp


def merge_blocks(src_pack, dst_pack, i, j, operator, chunk_rows=4096):
    """Write a folded pack with same-type blocks i and j replaced by one
    merged block at position i (i < j required, no silent swap). Returns the
    old->new block_map. All validation runs before dst is created."""
    if operator not in OPERATORS:
        raise ValueError(f"unknown operator {operator!r}; choose from {sorted(OPERATORS)}")
    kernel, out_bits = OPERATORS[operator]
    src = PackReader(src_pack)
    tcfg = src.config["text_config"]
    n = tcfg["num_hidden_layers"]
    if not (0 <= i < j < n):
        raise ValueError(f"need 0 <= i < j < {n}, got ({i}, {j})")
    if tcfg["layer_types"][i] != tcfg["layer_types"][j]:
        raise ValueError(
            f"cross-type merge: block {i} is {tcfg['layer_types'][i]}, "
            f"block {j} is {tcfg['layer_types'][j]}"
        )

    bases_i, fp_i = _block_tensor_kinds(src, i)
    for base in bases_i:
        for suffix in (".weight", ".scales", ".biases"):
            pi, pj = base + suffix, _partner(base, i, j) + suffix
            if pj not in src.header:
                raise ValueError(f"block {j} missing counterpart {pj}")
            if src.header[pi]["shape"] != src.header[pj]["shape"]:
                raise ValueError(f"pair shape mismatch on {pi}")
        for name in (base + ".weight", _partner(base, i, j) + ".weight"):
            gs, bits = src.quant_meta(name)
            if (gs, bits) != (GROUP_SIZE, 1):
                raise ValueError(
                    f"{name}: operators are defined on 1-bit g128 sources, "
                    f"got g{gs} b{bits}"
                )
    for name in fp_i:
        pj = _partner(name, i, j)
        if pj not in src.header:
            raise ValueError(f"block {j} missing counterpart {pj}")
        if (src.header[name]["dtype"], src.header[name]["shape"]) != (
            src.header[pj]["dtype"], src.header[pj]["shape"]
        ):
            raise ValueError(f"pair dtype/shape mismatch on {name}")

    block_map = {old: new for new, old in enumerate(k for k in range(n) if k != j)}

    merged = {}
    for base in bases_i:
        merged.update(
            _merge_quant_tensor(src, base, _partner(base, i, j), kernel, out_bits, chunk_rows)
        )
    for name in fp_i:
        arr = merge_fp_mean(src.read(name), src.read(_partner(name, i, j)))
        merged[name] = {"dtype": "F16", "shape": arr.shape,
                        "raw": np.ascontiguousarray(arr).tobytes()}

    entries = {}
    for name in src.header:  # header order keeps the pack diffable vs drops
        new_name = reindex_name(name, block_map)
        if new_name is None:
            continue
        if block_owner(name) == i:
            entries[new_name] = merged[name]
        else:
            begin, end = src.header[name]["data_offsets"]
            entries[new_name] = {
                "dtype": src.header[name]["dtype"],
                "shape": src.header[name]["shape"],
                "nbytes": end - begin,
                "raw": (lambda n=name: src.read_raw(n)),  # streamed, byte-identical
            }

    extra = None
    if out_bits == 2:
        extra = {
            reindex_name(base, block_map): {"group_size": GROUP_SIZE, "bits": 2}
            for base in bases_i
        }
    config = rewrite_folded_config(
        src.config,
        block_map,
        {
            "operation": "merge",
            "merged_pairs": [[i, j]],
            "rule": operator,
            "source_pack": str(src.pack_dir),
        },
        extra_quant_overrides=extra,
    )
    write_pack(dst_pack, entries, config, src.pack_dir)
    return block_map


def verify_merge(src_pack, dst_pack, block_map, i, j, operator):
    """Merge-aware integrity check (fold.verify_byte_identity fails on merge
    packs by design): survivors byte-identical under reindexed names, merged
    tensors exactly re-derivable from the source pair, config consistent.
    Returns a report dict; report["ok"] is the verdict."""
    kernel, out_bits = OPERATORS[operator]
    src, dst = PackReader(src_pack), PackReader(dst_pack)
    bases_i, fp_i = _block_tensor_kinds(src, i)

    expected = {}
    for name in src.header:
        if block_owner(name) in (i, j):
            continue
        new = reindex_name(name, block_map)
        if new is not None:
            expected[new] = name
    merged_names = {reindex_name(n, block_map) for b in bases_i
                    for n in (b + ".weight", b + ".scales", b + ".biases")}
    merged_names |= {reindex_name(n, block_map) for n in fp_i}
    missing = sorted((set(expected) | merged_names) - set(dst.header))
    extra = sorted(set(dst.header) - set(expected) - merged_names)
    mismatched = [
        n for n in sorted(set(expected) & set(dst.header))
        if src.read_raw(expected[n]) != dst.read_raw(n)
    ]

    merged_checks = {}
    for base in bases_i:
        new_base = reindex_name(base, block_map)
        checks = {}
        w, s, b = (dst.read(new_base + sfx) for sfx in (".weight", ".scales", ".biases"))
        checks["bits"] = infer_bits(w.shape, s.shape, GROUP_SIZE) == out_bits
        codes = unpack_codes(w, out_bits)
        checks["codes_range"] = int(codes.max()) <= (1 if out_bits == 1 else 2)
        if out_bits == 1:
            checks["biases_invariant"] = np.array_equal(
                b, (-(s.astype(np.float32)) / 2).astype(np.float16))
        else:
            checks["biases_invariant"] = np.array_equal(b, np.negative(s))
            over = dst.config.get("quantization", {}).get(new_base)
            checks["config_override"] = over == {"group_size": GROUP_SIZE, "bits": 2}
        ref = _merge_quant_tensor(src, base, _partner(base, i, j), kernel, out_bits, 4096)
        checks["recompute_exact"] = all(
            dst.read_raw(reindex_name(k, block_map)) == v["raw"] for k, v in ref.items()
        )
        merged_checks[base] = checks
    for name in fp_i:
        arr = merge_fp_mean(src.read(name), src.read(_partner(name, i, j)))
        merged_checks[name] = {
            "recompute_exact": dst.read_raw(reindex_name(name, block_map))
            == np.ascontiguousarray(arr).tobytes()
        }

    tcfg = dst.config["text_config"]
    src_types = src.config["text_config"]["layer_types"]
    config_ok = (
        tcfg["num_hidden_layers"] == len(src_types) - 1
        and tcfg["layer_types"][block_map[i]] == src_types[i]
    )
    ok = (
        not (missing or extra or mismatched)
        and config_ok
        and all(all(c.values()) for c in merged_checks.values())
    )
    return {
        "survivors_checked": len(expected),
        "missing": missing,
        "extra": extra,
        "mismatched": mismatched,
        "config_ok": config_ok,
        "merged_checks": merged_checks,
        "ok": ok,
    }
