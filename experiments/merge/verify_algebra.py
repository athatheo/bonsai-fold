"""Adversarial verification of merge.py: operator algebra + pack integration.

Every expectation here is reimplemented independently — own safetensors
parser (struct+json byte slicing), own LSB-first bit packer/unpacker
(Python-int loops), exact-rational (Fraction) algebra for the kernels —
so merge.py is never used to check merge.py. stio/fold are only *called*
as the system under test (via merge_blocks), never as the oracle.

Exactness argument used for bit-exact demands on sign_election scales:
with f16 source scales in [0.5, 4], magnitudes s_g = s/2 lie on a grid of
2^-13, so every |W| is a multiple of 2^-14 bounded by 4; any partial sum
over a 128-group is a multiple of 2^-14 bounded by 512 = 2^9, i.e. needs
at most 23 significand bits < fp32's 24 — every fp32 addition is exact
regardless of order, mean/128 is exact, and the final f16 cast is a
single rounding of the same exact real number my Fraction pipeline
produces. Sign decisions can never differ between fp32 and exact
arithmetic (terms are fp32-exact; a correctly-rounded sum has the sign of
the true sum, and is zero iff the true sum is zero).

CPU-only; no 27B load (the optional real-pack section reads .scales
tensors only — per-tensor file reads, no weights, no mlx).
Run: cd <repo> && uv run python experiments/merge/verify_algebra.py
"""
import json
import math
import struct
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tests"))

from conftest import LAYER_TYPES, N_BLOCKS, st_entry, write_merge_pack  # noqa: E402

from bonsaifold.merge import (  # noqa: E402  (system under test)
    merge_blocks,
    merge_fp_mean,
    promotion,
    sign_election,
)
from bonsaifold import stio  # noqa: E402  (under test where merge relies on it)

CHECKS = []


def check(name, ok, detail=""):
    CHECKS.append((name, bool(ok), detail))
    tag = "ok  " if ok else "FAIL"
    print(f"[{tag}] {name}" + (f" — {detail}" if detail and not ok else ""))


# ---------------------------------------------------------------- oracles ---

def my_parse_pack(st_path):
    """Independent safetensors parse: header dict + name -> exact bytes."""
    raw = Path(st_path).read_bytes()
    n = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8 : 8 + n].decode())
    header.pop("__metadata__", None)
    data = raw[8 + n :]
    blobs = {
        name: data[info["data_offsets"][0] : info["data_offsets"][1]]
        for name, info in header.items()
    }
    return header, blobs, len(data)


def my_owner(name):
    if name.startswith("vision_tower."):
        return "vision_tower"
    parts = name.split(".")
    if "layers" in parts:
        return int(parts[parts.index("layers") + 1])
    return "tail"


def my_rename(name, bm):
    if name.startswith("vision_tower."):
        return name
    parts = name.split(".")
    if "layers" not in parts:
        return name
    k = parts.index("layers") + 1
    old = int(parts[k])
    if old not in bm:
        return None
    parts[k] = str(bm[old])
    return ".".join(parts)


def my_unpack(words, bits):
    per, mask = 32 // bits, (1 << bits) - 1
    w2 = words.reshape(-1, words.shape[-1])
    out = np.empty((w2.shape[0], w2.shape[1] * per), dtype=np.uint8)
    for r in range(w2.shape[0]):
        k = 0
        for w in map(int, w2[r]):
            for p in range(per):
                out[r, k] = (w >> (p * bits)) & mask
                k += 1
    return out.reshape(*words.shape[:-1], -1)


def my_pack(codes, bits):
    per = 32 // bits
    c2 = codes.reshape(-1, codes.shape[-1])
    out = np.empty((c2.shape[0], c2.shape[1] // per), dtype=np.uint32)
    for r in range(c2.shape[0]):
        row = list(map(int, c2[r]))
        for w in range(out.shape[1]):
            acc = 0
            for p in range(per):
                acc |= row[w * per + p] << (p * bits)
            out[r, w] = acc
    return out.reshape(*codes.shape[:-1], -1)


def frac(x):
    """Exact rational value of a float16/float32 scalar."""
    return Fraction(float(x))


def inv_biases(scales):
    """The pack's 1-bit biases convention: f16(-scales/2) from f16 scales."""
    return (-(scales.astype(np.float32)) / 2).astype(np.float16)


def qbase(k):
    return f"language_model.model.layers.{k}.mlp.up_proj"


def fp_name(k):
    return f"language_model.model.layers.{k}.input_layernorm.weight"


def oracle_sign_election(qa, sa, qb, sb, gs):
    """Exact-rational sign_election: returns (codes u8, scales f16, biases
    f16) per the spec's algebra — W=(sA_g*tA + sB_g*tB)/2, ties->tA,
    per-group absmean, stored scale = f16(2*absmean), biases from the
    ROUNDED f16 scales. Independent of merge.py."""
    rows, cols = qa.shape
    codes = np.empty((rows, cols), dtype=np.uint8)
    scales = np.empty((rows, cols // gs), dtype=np.float16)
    for r in range(rows):
        for g in range(cols // gs):
            sag, sbg = frac(sa[r, g]) / 2, frac(sb[r, g]) / 2
            acc = Fraction(0)
            for c in range(g * gs, (g + 1) * gs):
                ta, tb = 2 * int(qa[r, c]) - 1, 2 * int(qb[r, c]) - 1
                w = (sag * ta + sbg * tb) / 2
                acc += abs(w)
                t = ta if w == 0 else (1 if w > 0 else -1)
                codes[r, c] = (t + 1) // 2
            scales[r, g] = np.float16(float(2 * acc / gs))
    return codes, scales, inv_biases(scales)


def oracle_promotion(qa, sa, qb, sb):
    """Exact-rational promotion oracle: t' = tA on agreement else 0
    (q' = t'+1); stored 2-bit scale = f16((storedA+storedB)/4) — this is
    s'=(sA_g+sB_g)/2 expressed in stored units; biases = -scales."""
    ta, tb = 2 * qa.astype(np.int8) - 1, 2 * qb.astype(np.int8) - 1
    tprime = np.where(ta == tb, ta, 0)
    codes = (tprime + 1).astype(np.uint8)
    scales = np.empty(sa.shape, dtype=np.float16)
    flat_a, flat_b, flat_s = sa.ravel(), sb.ravel(), scales.ravel()
    for k in range(flat_a.size):
        flat_s[k] = np.float16(float((frac(flat_a[k]) + frac(flat_b[k])) / 4))
    biases = np.negative(scales)
    return codes, scales, biases


def bits_equal(a, b):
    """Bit-exact f16 comparison (distinguishes -0.0 from +0.0)."""
    return a.shape == b.shape and np.array_equal(
        a.view(np.uint16), b.view(np.uint16)
    )


# ============================================================ (pre) packer ==
# merge.py trusts stio.pack_codes/unpack_codes for every merged payload;
# verify them against the independent Python-int implementation first.

rng = np.random.default_rng(4242)
for bits in (1, 2):
    words = rng.integers(0, 2**32, size=(3, 4), dtype=np.uint32)
    check(
        f"stio.unpack_codes matches independent unpacker (bits={bits})",
        np.array_equal(stio.unpack_codes(words, bits), my_unpack(words, bits)),
    )
    codes = rng.integers(0, 2**bits, size=(3, 128), dtype=np.uint8)
    check(
        f"stio.pack_codes matches independent packer (bits={bits})",
        np.array_equal(stio.pack_codes(codes, bits), my_pack(codes, bits)),
    )

# ===================================================== (a) sign_election ====
# Crafted 2x256, gs=128, UNEQUAL per-group scales, engineered to contain:
#   row0 g0: sA > sB (A wins disagreements), row0 g1: sA < sB (B wins),
#   row1 g0: sA == sB (exact-zero ties -> tA), row1 g1: sA > sB.
GS = 128
S_A = np.array([[1.0, 0.25], [0.25, 2.0]], dtype=np.float16)
S_B = np.array([[0.5, 1.0], [0.25, 0.5]], dtype=np.float16)
Q_A = rng.integers(0, 2, size=(2, 256), dtype=np.uint8)
Q_B = rng.integers(0, 2, size=(2, 256), dtype=np.uint8)
Q_B[1, :16] = 1 - Q_A[1, :16]  # force ties in the equal-scale group
Q_B[1, 16:32] = Q_A[1, 16:32]  # and agreements right next to them
B_A, B_B = inv_biases(S_A), inv_biases(S_B)

exp_codes, exp_scales, exp_biases = oracle_sign_election(Q_A, S_A, Q_B, S_B, GS)
got_codes, got_scales, got_biases = sign_election(Q_A, S_A, B_A, Q_B, S_B, B_B)

DIS = Q_A != Q_B
TIES = (S_A == S_B).repeat(GS, axis=-1) & DIS
A_WINS = (S_A > S_B).repeat(GS, axis=-1) & DIS
B_WINS = (S_A < S_B).repeat(GS, axis=-1) & DIS
check("(a) crafted example exercises ties and B-wins elections",
      TIES.any() and B_WINS.any(),
      f"ties={int(TIES.sum())} b_wins={int(B_WINS.sum())}")
check("(a) sign_election codes == exact-rational oracle (ties -> tA)",
      np.array_equal(got_codes, exp_codes) and got_codes.dtype == np.uint8)
check("(a) sign_election scales == f16(2*absmean(W_avg)), bit-exact",
      bits_equal(got_scales, exp_scales),
      f"got {got_scales.tolist()} want {exp_scales.tolist()}")
check("(a) sign_election biases == f16(-scales_f16/2) from ROUNDED scales",
      bits_equal(got_biases, exp_biases))
check("(a) where scales unequal, disagreement resolves to larger-|s| side",
      np.array_equal(got_codes[A_WINS], Q_A[A_WINS])
      and np.array_equal(got_codes[B_WINS], Q_B[B_WINS]))

# random torture within the exactness envelope (see module docstring)
for seed in range(5):
    r = np.random.default_rng(seed)
    qa = r.integers(0, 2, size=(4, 512), dtype=np.uint8)
    qb = r.integers(0, 2, size=(4, 512), dtype=np.uint8)
    sa = r.uniform(0.5, 4.0, size=(4, 4)).astype(np.float16)
    sb = r.uniform(0.5, 4.0, size=(4, 4)).astype(np.float16)
    # sprinkle exact scale collisions so ties occur
    sb[:, 0] = sa[:, 0]
    ec, es, eb = oracle_sign_election(qa, sa, qb, sb, GS)
    gc, gs_, gb = sign_election(qa, sa, inv_biases(sa), qb, sb, inv_biases(sb))
    if not (np.array_equal(gc, ec) and bits_equal(gs_, es) and bits_equal(gb, eb)):
        check(f"(a) sign_election random torture seed={seed}", False)
        break
else:
    check("(a) sign_election random torture x5 (codes+scales+biases bit-exact)", True)

# ======================================================== (b) promotion =====
exp_codes, exp_scales, exp_biases = oracle_promotion(Q_A, S_A, Q_B, S_B)
got_codes, got_scales, got_biases = promotion(Q_A, S_A, B_A, Q_B, S_B, B_B)

# the q'=qA+qB shortcut IS agreement-mask*tA + 1: exhaustive over all combos
combo_ok = all(
    int(qa + qb) == ((2 * qa - 1) if qa == qb else 0) + 1
    for qa in (0, 1) for qb in (0, 1)
)
check("(b) q'=qA+qB == agreement*tA + 1, exhaustive over {0,1}^2", combo_ok)
check("(b) promotion codes == mask-based oracle; code 3 unreachable",
      np.array_equal(got_codes, exp_codes) and int(got_codes.max()) <= 2)
check("(b) promotion stored scales == f16((storedA+storedB)/4), bit-exact",
      bits_equal(got_scales, exp_scales))
wrong_half = ((S_A.astype(np.float32) + S_B.astype(np.float32)) / 2).astype(np.float16)
check("(b) /4 not /2: stored 2-bit scale is s'=(sA_g+sB_g)/2 in STORED units "
      "(stored 1-bit scale = 2*s_g; stored 2-bit scale = s')",
      not bits_equal(got_scales, wrong_half)
      and bits_equal(got_scales, (wrong_half.astype(np.float32) / 2).astype(np.float16)))
check("(b) promotion biases == -scales (exact f16 sign flip, bitwise)",
      np.array_equal(got_biases.view(np.uint16),
                     got_scales.view(np.uint16) ^ np.uint16(0x8000)))

# dequant meaning under the affine convention w = s*q + b (my own dequant):
s32 = got_scales.astype(np.float32).repeat(GS, axis=-1)
b32 = got_biases.astype(np.float32).repeat(GS, axis=-1)
deq = got_codes.astype(np.float32) * s32 + b32
zero_exact = np.all(deq[DIS] == 0.0) and not np.any(
    np.signbit(deq[DIS]))  # +0.0 exactly, not -0.0
agree_pm = np.array_equal(deq[~DIS], np.where(Q_A == 1, s32, -s32)[~DIS])
check("(b) disagreeing positions dequant to EXACTLY +0.0 (affine s*1 + (-s))",
      zero_exact)
check("(b) agreeing positions dequant to exactly ±s'", agree_pm)

# ============================================== (c)(d)(e) pack integration ==
tmp = Path(tempfile.mkdtemp(prefix="verify_algebra_"))
src_dir = write_merge_pack(tmp / "src")
src_header, src_blobs, _ = my_parse_pack(src_dir / "model.safetensors")


def src_arr(name, dt):
    return np.frombuffer(src_blobs[name], dt).reshape(src_header[name]["shape"])


MERGES = [("promotion", 0, 1, 2), ("sign_election", 4, 5, 1)]
for op, i, j, out_bits in MERGES:
    dst_dir = tmp / f"dst_{op}"
    bm = merge_blocks(src_dir, dst_dir, i, j, op)
    keep = [k for k in range(N_BLOCKS) if k != j]
    exp_bm = {old: new for new, old in enumerate(keep)}
    check(f"(c) [{op}] block_map correct", bm == exp_bm, f"{bm}")

    base, pj = qbase(i), qbase(j)
    fp_i, fp_j = fp_name(i), fp_name(j)
    new_base = my_rename(base, bm)

    cfg = json.loads((dst_dir / "config.json").read_text())
    tcfg = cfg["text_config"]
    exp_types = [LAYER_TYPES[k] for k in keep]
    check(f"(c) [{op}] num_hidden_layers == {N_BLOCKS - 1}",
          tcfg["num_hidden_layers"] == N_BLOCKS - 1)
    check(f"(c) [{op}] layer_types trimmed, host type preserved at index {i}",
          tcfg["layer_types"] == exp_types
          and tcfg["layer_types"][i] == LAYER_TYPES[i])
    check(f"(c) [{op}] full_attention_interval untouched",
          tcfg.get("full_attention_interval") == 4)

    overrides = {k: v for k, v in cfg.get("quantization", {}).items()
                 if isinstance(v, dict)}
    if op == "promotion":
        check("(c) [promotion] override present ONLY for merged base at NEW "
              "index, top-level config only",
              overrides == {new_base: {"group_size": 128, "bits": 2}}
              and "quantization" not in tcfg
              and cfg["quantization"]["group_size"] == 128
              and cfg["quantization"]["bits"] == 1,
              f"overrides={overrides}")
    else:
        check("(c) [sign_election] adds NO overrides anywhere",
              overrides == {} and "quantization" not in tcfg,
              f"overrides={overrides}")
    stamp = cfg.get("bonsai_fold", {})
    check(f"(c) [{op}] provenance stamp fields",
          stamp.get("operation") == "merge"
          and stamp.get("merged_pairs") == [[i, j]]
          and stamp.get("rule") == op
          and stamp.get("source_pack") == str(src_dir)
          and isinstance(stamp.get("created_utc"), str)
          and "load_bonsai" in stamp.get("note", ""),
          f"stamp={stamp}")

    # ---- (d) survivors byte-identical, tail+vision untouched, no extras ----
    dst_header, dst_blobs, dst_data_len = my_parse_pack(dst_dir / "model.safetensors")
    # the recompute checks below must cover ALL of block i, or fixture drift
    # (a new per-block tensor) would silently escape value verification
    block_i_names = {n for n in src_header if my_owner(n) == i}
    covered = {base + sfx for sfx in (".weight", ".scales", ".biases")} | {fp_i}
    check(f"(d) [{op}] value recomputes cover ALL of block {i}'s tensors",
          block_i_names == covered, f"uncovered={block_i_names - covered}")
    exp_names = {my_rename(n, bm): n for n in src_header if my_owner(n) != j}
    check(f"(d) [{op}] dst tensor set == survivors + merged block, no extras",
          set(dst_header) == set(exp_names),
          f"missing={set(exp_names) - set(dst_header)} "
          f"extra={set(dst_header) - set(exp_names)}")
    bad = [
        n for n, old in exp_names.items()
        if my_owner(old) != i and (
            dst_blobs.get(n) != src_blobs[old]
            or dst_header[n]["dtype"] != src_header[old]["dtype"]
            or dst_header[n]["shape"] != src_header[old]["shape"])
    ]
    check(f"(d) [{op}] every survivor byte-identical (dtype/shape/raw)",
          not bad, f"mismatched={bad[:5]}")
    check(f"(d) [{op}] tail + vision names unchanged, bytes identical",
          dst_blobs["language_model.lm_head.weight"]
          == src_blobs["language_model.lm_head.weight"]
          and dst_blobs["vision_tower.blocks.0.attn.qkv.weight"]
          == src_blobs["vision_tower.blocks.0.attn.qkv.weight"])

    # index.json bookkeeping (not asserted by any pytest test — see critique)
    idx = json.loads((dst_dir / "model.safetensors.index.json").read_text())
    check(f"(d) [{op}] index.json total_size == payload bytes, weight_map complete",
          idx["metadata"]["total_size"] == dst_data_len
          and set(idx["weight_map"]) == set(dst_header))

    # ---- merged quantized triple: full independent recompute from source ---
    wa, wb = src_arr(base + ".weight", np.uint32), src_arr(pj + ".weight", np.uint32)
    sa, sb = src_arr(base + ".scales", np.float16), src_arr(pj + ".scales", np.float16)
    qa, qb = my_unpack(wa, 1), my_unpack(wb, 1)
    if op == "promotion":
        ec, es, ebi = oracle_promotion(qa, sa, qb, sb)
    else:
        ec, es, ebi = oracle_sign_election(qa, sa, qb, sb, GS)
    exp_words = my_pack(ec, out_bits)
    check(f"(b/c) [{op}] merged weight payload == independent recompute; "
          f"shape {list(exp_words.shape)}, {out_bits}-bit",
          dst_blobs[new_base + ".weight"] == exp_words.tobytes()
          and dst_header[new_base + ".weight"]["shape"] == list(exp_words.shape))
    check(f"(b/c) [{op}] merged scales+biases payloads == independent recompute",
          dst_blobs[new_base + ".scales"] == es.tobytes()
          and dst_blobs[new_base + ".biases"] == ebi.tobytes())

    # ---- (e) FP params: fp32 mean -> f16, recomputed independently ---------
    a, b = src_arr(fp_i, np.float16), src_arr(fp_j, np.float16)
    exp_fp = ((a.astype(np.float32) + b.astype(np.float32)) / 2).astype(np.float16)
    check(f"(e) [{op}] FP tensor == f16(fp32 mean), byte-exact",
          dst_blobs[my_rename(fp_i, bm)] == exp_fp.tobytes())

# merge_fp_mean really accumulates in fp32 (naive f16 a+b would overflow)
big = np.array([60000.0], dtype=np.float16)
check("(e) merge_fp_mean fp32 accumulation (f16 60000+60000 would be inf)",
      merge_fp_mean(big, big).tolist() == [60000.0])

# pre-existing per-tensor override gets RENAMED across the merge
src2 = write_merge_pack(tmp / "src_override")
cfg2 = json.loads((src2 / "config.json").read_text())
cfg2["quantization"][qbase(7)] = {"group_size": 128, "bits": 1}
(src2 / "config.json").write_text(json.dumps(cfg2))
merge_blocks(src2, tmp / "dst_override", 0, 1, "promotion")
q2 = json.loads((tmp / "dst_override" / "config.json").read_text())["quantization"]
check("(c) pre-existing override renamed 7 -> 6 alongside the merged-base "
      "override",
      {k for k, v in q2.items() if isinstance(v, dict)}
      == {qbase(0), qbase(6)} and q2[qbase(6)]["bits"] == 1)

# failed validation must not leave a partial dst behind
try:
    merge_blocks(src_dir, tmp / "dst_crosstype", 2, 3, "promotion")
    check("(c) cross-type merge refused", False)
except ValueError:
    check("(c) cross-type merge refused; no partial dst created",
          not (tmp / "dst_crosstype").exists())

# ADVERSARIAL: an EXTRA tensor in absorbed block j vanishes silently
src3 = write_merge_pack(tmp / "src_extra_j")
h3, b3, _ = my_parse_pack(src3 / "model.safetensors")
extra_name = "language_model.model.layers.1.post_attention_layernorm.weight"
entries3 = {n: {"dtype": h3[n]["dtype"], "shape": h3[n]["shape"], "raw": b3[n]}
            for n in h3}
entries3[extra_name] = st_entry(np.ones(4, dtype=np.float16))
stio.write_safetensors(src3 / "model.safetensors", entries3)
merge_blocks(src3, tmp / "dst_extra_j", 0, 1, "promotion")
h3d, _, _ = my_parse_pack(tmp / "dst_extra_j" / "model.safetensors")
vanished = not any("post_attention_layernorm" in n for n in h3d)
check("(!) finding demo: extra tensor present only in absorbed block j is "
      "silently dropped (no validation error, verify_merge blind to it)",
      vanished)  # 'ok' here means the (undesirable) behavior is confirmed

# ===================== (f) f16((s+s)/4) == f16(s/2) and the dequant claim ===
s = np.arange(65536, dtype=np.uint16).view(np.float16)
s = s[np.isfinite(s)]
s32 = s.astype(np.float32)
lhs = ((s32 + s32) / 4).astype(np.float16)   # promotion self-merge stored scale
rhs = (s32 / 2).astype(np.float16)
check("(f) f16((s+s)/4) == f16(s/2) for ALL finite f16 (bitwise, incl. "
      "subnormals)", np.array_equal(lhs.view(np.uint16), rhs.view(np.uint16)))

# ...but the DEQUANT no-op claim needs s == 2*f16(s/2):
b32 = inv_biases(s).astype(np.float32)     # source biases f16(-s/2), as fp32
s_new = lhs.astype(np.float32)             # promoted scale s', as fp32
src_q1 = s32 + b32                         # 1-bit dequant at q=1
src_q0 = b32                               # at q=0
new_hi = 2 * s_new - s_new                 # 2-bit dequant at q'=2
new_lo = -s_new                            # q'=0 (0*s' + b')
n_break_q1 = int(np.sum(src_q1 != new_hi))
n_break_q0 = int(np.sum(src_q0 != new_lo))
breakers = s[src_q1 != new_hi]
max_breaker = float(np.abs(breakers).max()) if n_break_q1 else 0.0
check("(f) promotion self-merge dequant no-op is NOT universal: breaks "
      f"exactly where f16(s/2) != s/2 ({n_break_q1} of {s.size} finite f16 "
      f"scales, all |s| <= {max_breaker:.3e}; q=0 side always exact "
      f"[{n_break_q0} breaks])",
      n_break_q1 > 0 and n_break_q0 == 0
      and np.array_equal(src_q1 != new_hi, rhs.astype(np.float32) != s32 / 2))

# does the REAL pack's scale range reach the breaking region? (raw .scales
# reads only — not a model load). MEASUREMENT, not an assertion: breakers in
# the real pack invalidate the universal form of the self-merge-dequant-no-op
# claim (spec §7 / test_self_merge_exactness), though the per-weight error is
# bounded by half an f16-subnormal ulp (2^-25) and self-merge never occurs in
# a real i<j merge.
real = REPO / "models" / "Bonsai-27B-mlx-1bit"
if real.exists():
    smallest, total, n_zero, n_break = math.inf, 0, 0, 0
    reader = stio.PackReader(real)
    for name in reader.header:
        if name.endswith(".scales") and name.startswith("language_model."):
            v = reader.read(name)
            v32 = v.astype(np.float32)
            total += v.size
            n_zero += int(np.sum(v32 == 0))
            half = v32 / 2  # exact in fp32
            n_break += int(np.sum(half.astype(np.float16).astype(np.float32) != half))
            nz = np.abs(v32[v32 != 0])
            if nz.size:
                smallest = min(smallest, float(nz.min()))
    print(f"  [measure] real pack language_model .scales: {total} values, "
          f"min nonzero |s| = {smallest:.3e}, exact zeros = {n_zero}, "
          f"self-merge-dequant breakers (f16(s/2) != s/2) = {n_break} "
          f"({100.0 * n_break / total:.4f}%)")
else:
    print("[note] real pack absent; skipped scale-floor scan")

# ================================================================= verdict ==
fails = [c for c in CHECKS if not c[1]]
print(f"\n{len(CHECKS)} checks, {len(fails)} failures")
for name, _, detail in fails:
    print(f"  FAIL {name} {detail}")
print("VERDICT:", "FAIL" if fails else "PASS")
sys.exit(1 if fails else 0)
