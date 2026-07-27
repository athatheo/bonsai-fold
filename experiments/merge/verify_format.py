"""Adversarial format verification for bonsaifold.merge + stio.pack_codes.

All expectations are computed by INDEPENDENT reimplementations in this file
(np.unpackbits-based bit codec, own dequant, own f32 kernel math, plain
python-int known vectors). merge.py / stio.py are called only as the
things-under-test. CPU + numpy only; reads a few real tensors from the 1-bit
pack via PackReader range reads (no model load).

Run: cd <repo> && uv run python experiments/merge/verify_format.py
"""
import sys
from pathlib import Path

import numpy as np

from bonsaifold.merge import promotion, sign_election
from bonsaifold.stio import PackReader, pack_codes, unpack_codes

REPO = Path(__file__).resolve().parents[2]
PACK = REPO / "models" / "Bonsai-27B-mlx-1bit"
GS = 128

RESULTS = []  # (section, name, ok, detail)


def check(section, name, ok, detail=""):
    RESULTS.append((section, name, bool(ok), detail))
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] ({section}) {name}" + (f" -- {detail}" if detail else ""))


# ---------------------------------------------------------------------------
# Independent reimplementations (do NOT mirror stio/merge internals)
# ---------------------------------------------------------------------------

def my_unpack(words_u32, bits):
    """LSB-first unpack via np.unpackbits on the little-endian byte stream --
    a genuinely different code path than stio's shift/mask formulation."""
    w = np.ascontiguousarray(words_u32.astype("<u4"))
    nbits = np.unpackbits(w.view(np.uint8).reshape(-1), bitorder="little")
    codes = nbits.reshape(-1, bits) @ (1 << np.arange(bits, dtype=np.uint8))
    return codes.reshape(*w.shape[:-1], w.shape[-1] * (32 // bits))


def my_pack(codes, bits):
    """Inverse of my_unpack via np.packbits(bitorder='little')."""
    c = np.ascontiguousarray(codes).astype(np.uint8)
    bitplane = (c[..., None] >> np.arange(bits, dtype=np.uint8)) & 1
    flat = bitplane.reshape(*c.shape[:-1], c.shape[-1] * bits)
    packed8 = np.packbits(flat, axis=-1, bitorder="little")
    return np.ascontiguousarray(packed8).view("<u4").astype(np.uint32)


def my_dequant(codes, scales, biases):
    """w = q*s + b in f32 with per-128 group broadcast (own repeat)."""
    s = np.repeat(scales.astype(np.float32), GS, axis=-1)
    b = np.repeat(biases.astype(np.float32), GS, axis=-1)
    return codes.astype(np.float32) * s + b


def bits16(a):
    return np.ascontiguousarray(a.astype(np.float16)).view(np.uint16)


def w_avg(qa, sa, qb, sb, dtype=np.float32):
    """Spec section-2 W_avg = (s_a*t_a + s_b*t_b)/2 with s_x_g = scales/2."""
    ta = qa.astype(dtype) * 2 - 1
    tb = qb.astype(dtype) * 2 - 1
    ma = np.repeat(sa.astype(dtype) / 2, GS, axis=-1)
    mb = np.repeat(sb.astype(dtype) / 2, GS, axis=-1)
    return (ma * ta + mb * tb) * dtype(0.5)


def expect_sign_election(qa, sa, qb, sb):
    """Spec section-2 math, reimplemented: f32 throughout, ONE f32->f16 cast
    for scales, biases derived from the already-rounded f16 scales."""
    w = w_avg(qa, sa, qb, sb)
    mag32 = np.abs(w).reshape(*w.shape[:-1], -1, GS).mean(-1, dtype=np.float32)
    ta = qa.astype(np.float32) * 2 - 1
    t_new = np.where(w > 0, 1.0, np.where(w < 0, -1.0, ta)).astype(np.float32)
    codes = (t_new > 0).astype(np.uint8)
    scales = (2.0 * mag32).astype(np.float16)
    biases = (-(scales.astype(np.float32)) / 2).astype(np.float16)
    return codes, scales, biases, mag32, t_new, int((w == 0).sum())


def expects_valueerror(section, name, fn):
    try:
        fn()
    except ValueError as e:
        check(section, name, True, f"ValueError: {e}")
    except Exception as e:  # wrong exception type
        check(section, name, False, f"raised {type(e).__name__} not ValueError: {e}")
    else:
        check(section, name, False, "no exception raised")


# ---------------------------------------------------------------------------
src = PackReader(PACK)


def read_wsb(base):
    return tuple(src.read(base + s) for s in (".weight", ".scales", ".biases"))


qcfg = src.config["quantization"]
check("e", "pack config declares group_size=128, bits=1 (matches hardcoded GS)",
      (qcfg.get("group_size"), qcfg.get("bits")) == (GS, 1), str(qcfg))

BASES = [
    "language_model.model.layers.{k}.linear_attn.in_proj_a",
    "language_model.model.layers.{k}.linear_attn.in_proj_b",
    "language_model.model.layers.{k}.linear_attn.out_proj",
]
I, J = 4, 5
STASH = None  # first-iteration in_proj_a arrays, reused by the sections below

for base_t in BASES:
    b4, b5 = base_t.format(k=I), base_t.format(k=J)
    short = b4.split("layers.")[1]
    w4, s4, bi4 = read_wsb(b4)
    w5, s5, bi5 = read_wsb(b5)

    # ---- (e) group alignment on every real tensor touched -----------------
    cols = w4.shape[-1] * 32
    check("e", f"{short}: scales_cols*128 == cols",
          s4.shape[-1] * GS == cols and s5.shape[-1] * GS == cols
          and bi4.shape == s4.shape and bi5.shape == s5.shape,
          f"packed {w4.shape} -> cols={cols}, scales {s4.shape}")

    # source pack invariant, BIT level (informational census on real data)
    for tag, s, b in ((b4, s4, bi4), (b5, s5, bi5)):
        exp = (-(s.astype(np.float32)) / 2).astype(np.float16)
        check("src", f"{tag.split('layers.')[1]}: pack biases==f16(-scales/2) bit-exact",
              np.array_equal(bits16(b), bits16(exp)))

    # ---- (c) codec cross-checks on the real weight -------------------------
    qa, qb = my_unpack(w4, 1), my_unpack(w5, 1)  # kernels get independent codes
    qa_stio = unpack_codes(w4, 1)
    check("c", f"{short}: stio.unpack_codes == unpackbits-based unpack (real, 1-bit)",
          np.array_equal(qa_stio, qa))
    check("c", f"{short}: pack_codes(unpack_codes(w,1),1) == w bit-exact",
          np.array_equal(pack_codes(qa_stio, 1), w4))
    check("c", f"{short}: my_pack(my_unpack(w)) == w (sanity of independent codec)",
          np.array_equal(my_pack(qa, 1), w4))

    if STASH is None:
        STASH = (qa, s4, bi4, qb, s5, bi5)

    # ======================= (a) sign_election ==============================
    cn, sn, bn = sign_election(qa, s4, bi4, qb, s5, bi5)
    e_codes, e_scales, e_biases, mag32, t_new, n_ties = expect_sign_election(qa, s4, qb, s5)

    u = np.unique(cn)
    check("a", f"{short}: codes dtype/values in {{0,1}}",
          cn.dtype == np.uint8 and set(u.tolist()) <= {0, 1}, f"unique={u}")
    check("a", f"{short}: codes == independent sign(W_avg) w/ ties->t_a",
          np.array_equal(cn, e_codes), f"ties broken: {n_ties}")
    # (d) single f32->f16 cast: bit-exact against the independent f32 pipeline
    # pins the rounding chain. (Power-of-2 "double-round" paths -- e.g.
    # f16(2*f16(mag)) -- provably coincide in the normal f16 range, so the
    # only extra discriminators that exist are f64 accumulation, counted here,
    # and the subnormal bias case tested after the loop.)
    w64 = w_avg(qa, s4, qb, s5, np.float64)
    s_f64 = (2.0 * np.abs(w64).reshape(*qa.shape[:-1], -1, GS).mean(-1)).astype(np.float16)
    f64n = int((bits16(e_scales) != bits16(s_f64)).sum())
    check("a,d", f"{short}: scales bit-exact vs single-cast f32 pipeline",
          np.array_equal(bits16(sn), bits16(e_scales)),
          f"f64-accumulation pipeline would differ at {f64n} groups")
    check("a", f"{short}: biases == f16(-scales/2) BIT-exact (uint16 view)",
          np.array_equal(bits16(bn), bits16(e_biases)))
    check("a", f"{short}: no -0.0 scales in output",
          not np.any(bits16(sn) == 0x8000))

    # dequant(output) vs independently computed sign(W_avg)*absmean
    packed1 = pack_codes(cn, 1)
    check("a", f"{short}: output packs to expected 1-bit width",
          packed1.shape == w4.shape)
    deq = my_dequant(my_unpack(packed1, 1), sn, bn)
    ideal = t_new * np.repeat(sn.astype(np.float32) / 2, GS, axis=-1)
    check("a", f"{short}: dequant(output) == sign(W_avg)*absmean (f32 exact)",
          np.array_equal(deq, ideal),
          f"max|diff|={np.abs(deq - ideal).max()}")
    # closeness to the un-rounded f32 absmean (f16 rounding only)
    raw_ideal = t_new * np.repeat(mag32, GS, axis=-1)
    nz = raw_ideal != 0
    ri = raw_ideal[nz]
    rel_max = float((np.abs(deq[nz] - ri) / np.abs(ri)).max()) if ri.size else 0.0
    check("a", f"{short}: dequant within f16 rounding of raw f32 absmean",
          rel_max <= 2.0 ** -11, f"max rel dev={rel_max:.3e}")

    # ======================== (b) promotion ================================
    cp, sp, bp = promotion(qa, s4, bi4, qb, s5, bi5)
    check("b", f"{short}: codes == q_a+q_b elementwise",
          cp.dtype == np.uint8 and np.array_equal(cp, qa + qb))
    n3 = int((cp == 3).sum())
    check("b", f"{short}: every code in {{0,1,2}}, code 3 never present",
          int(cp.max()) <= 2, f"count(code==3)={n3} over {cp.size} values")
    e_sp = ((s4.astype(np.float32) + s5.astype(np.float32)) / 4).astype(np.float16)
    check("b,d", f"{short}: scales == f16((s_a+s_b)/4) single cast",
          np.array_equal(bits16(sp), bits16(e_sp)),
          "one rounding site; correctly-rounded paths are bit-identical")
    check("b", f"{short}: biases == -scales exact incl. sign bits (uint16 xor 0x8000)",
          np.array_equal(bits16(bp), bits16(sp) ^ 0x8000))

    packed2 = pack_codes(cp, 2)
    check("b", f"{short}: 2-bit packed width == cols/16",
          packed2.shape == (*qa.shape[:-1], cols // 16))
    q2 = my_unpack(packed2, 2)
    check("b", f"{short}: 2-bit pack round-trips through independent codec",
          np.array_equal(q2, cp))
    deq2 = my_dequant(q2, sp, bp)
    disagree = qa != qb
    check("b", f"{short}: dequant exactly 0.0 where source signs disagree",
          bool(np.all(deq2[disagree] == 0.0)),
          f"{int(disagree.sum())} disagreement sites")
    s_rep = np.repeat(sp.astype(np.float32), GS, axis=-1)
    agree_exp = np.where(qa == 1, s_rep, -s_rep)
    check("b", f"{short}: dequant == +/-s' exactly where signs agree",
          bool(np.all(deq2[~disagree] == agree_exp[~disagree])))

# ---------------------------------------------------------------------------
# sections below reuse the first-iteration in_proj_a pair (explicit, no re-read)
qa, s4, bi4, qb, s5, bi5 = STASH
qflip = qa ^ 1

# (a/-0.0) all-cancel: real tensor vs its exact negation -> scales +0.0, biases -0.0
cn0, sn0, bn0 = sign_election(qa, s4, bi4, qflip, s4, bi4)
check("a", "all-cancel: scales bits all 0x0000 (+0.0)",
      bool(np.all(bits16(sn0) == 0x0000)), f"unique bits {np.unique(bits16(sn0))}")
check("a", "all-cancel: biases bits all 0x8000 (-0.0, sign preserved)",
      bool(np.all(bits16(bn0) == 0x8000)), f"unique bits {np.unique(bits16(bn0))}")
check("a", "all-cancel: value-level array_equal would hide the sign (bit view required)",
      bool(np.array_equal(bn0, np.zeros_like(bn0))))  # documents why we view uint16
check("a", "all-cancel: ties everywhere -> codes == codes_a",
      np.array_equal(cn0, qa))
check("a", "all-cancel: dequant all exactly 0.0",
      bool(np.all(my_dequant(cn0, sn0, bn0) == 0.0)))

# promotion with zero scales -> biases must be -0.0-consistent (bits ^ 0x8000)
sz = np.zeros_like(s4)
bz = np.zeros_like(bi4)  # +0.0 passes the value-level source invariant
cpz, spz, bpz = promotion(qa, sz, bz, qflip, sz, bz)
check("b", "zero-scale promotion: biases bits == scales bits ^ 0x8000 (-0.0 case)",
      np.array_equal(bits16(bpz), bits16(spz) ^ 0x8000),
      f"scale bits {np.unique(bits16(spz))}, bias bits {np.unique(bits16(bpz))}")

# ---------------------------------------------------------------------------
# (d) the ONE genuinely double-roundable site: biases derived from the rounded
# f16 scales vs directly from the raw f32 absmean. In the normal f16 range the
# two provably coincide (power-of-2 scaling commutes with rounding), so force
# the f16-subnormal regime where they diverge and assert the kernel takes the
# spec'd chain. Construction (Q = 2^-24, the f16 subnormal quantum):
#   s_a=5Q, s_b=1Q, one 128-col group, 90 sign-agreements ->
#   absmean = (90*1.5Q + 38*1Q)/128 = (173/128)Q exactly in f32
#   correct: scales=f16(2*absmean)=3Q (0x0003),
#            biases=f16(-f32(3Q)/2)=f16(-1.5Q)=-2Q (0x8002, tie-to-even)
#   wrong:   f16(-absmean)=f16(-1.3516Q)=-1Q (0x8001)
Q16 = np.float16(2.0 ** -24)
sa_s = np.full((1, 1), 5 * Q16, dtype=np.float16)
sb_s = np.full((1, 1), Q16, dtype=np.float16)
ba_s = (-(sa_s.astype(np.float32)) / 2).astype(np.float16)
bb_s = (-(sb_s.astype(np.float32)) / 2).astype(np.float16)
qa_s = np.ones((1, GS), dtype=np.uint8)
qb_s = np.ones((1, GS), dtype=np.uint8)
qb_s[0, 90:] = 0
cs, ss, bs = sign_election(qa_s, sa_s, ba_s, qb_s, sb_s, bb_s)
_, e_ss, e_bs, mag_s, _, _ = expect_sign_election(qa_s, sa_s, qb_s, sb_s)
b_wrong = (-mag_s).astype(np.float16)
check("d", "subnormal case: construction matches hand-computed bit patterns",
      bits16(e_ss)[0, 0] == 0x0003 and bits16(e_bs)[0, 0] == 0x8002
      and bits16(b_wrong)[0, 0] == 0x8001,
      f"s={bits16(e_ss)[0, 0]:#06x} b={bits16(e_bs)[0, 0]:#06x} "
      f"wrong={bits16(b_wrong)[0, 0]:#06x}")
check("d", "subnormal case: the two bias conventions actually differ (discriminating)",
      not np.array_equal(bits16(e_bs), bits16(b_wrong)))
check("d", "subnormal case: kernel biases come from the rounded f16 scales",
      np.array_equal(bits16(ss), bits16(e_ss))
      and np.array_equal(bits16(bs), bits16(e_bs)),
      f"kernel s={bits16(ss)[0, 0]:#06x} b={bits16(bs)[0, 0]:#06x}")

# ---------------------------------------------------------------------------
# (c) codec: known vectors from plain python ints + random round-trips
# ---------------------------------------------------------------------------
rng = np.random.default_rng(20260728)
for bits in (1, 2, 4):
    per = 32 // bits
    mask = (1 << bits) - 1
    words = rng.integers(0, 2 ** 32, size=(5, 6), dtype=np.uint32)
    # plain python-int expectation, word by word
    exp = np.array([[(int(w) >> (k * bits)) & mask for k in range(per)]
                    for w in words.reshape(-1)], dtype=np.uint32)
    exp = exp.reshape(5, 6 * per)
    check("c", f"bits={bits}: unpack_codes matches python-int LSB-first decode",
          np.array_equal(unpack_codes(words, bits), exp))
    # python-int packing expectation
    codes = rng.integers(0, mask + 1, size=(4, per * 3), dtype=np.uint32)
    exp_words = np.array([
        [sum(int(codes[r, w * per + k]) << (k * bits) for k in range(per))
         for w in range(3)] for r in range(4)], dtype=np.uint32)
    check("c", f"bits={bits}: pack_codes matches python-int LSB-first encode",
          np.array_equal(pack_codes(codes, bits), exp_words))
    # round-trips both directions
    check("c", f"bits={bits}: unpack(pack(codes)) == codes",
          np.array_equal(unpack_codes(pack_codes(codes, bits), bits), codes))
    w2 = rng.integers(0, 2 ** 32, size=(3, 7), dtype=np.uint32)
    check("c", f"bits={bits}: pack(unpack(words)) == words",
          np.array_equal(pack_codes(unpack_codes(w2, bits), bits), w2))
    check("c", f"bits={bits}: stio codec agrees with unpackbits-based codec",
          np.array_equal(unpack_codes(w2, bits), my_unpack(w2, bits))
          and np.array_equal(pack_codes(codes, bits), my_pack(codes, bits)))

# first element must land in the LOW bits of word 0 (LSB-first semantics)
first = np.zeros(32, dtype=np.uint32)
first[0] = 1
check("c", "LSB-first: codes=[1,0,...] packs to word 0x00000001",
      int(pack_codes(first, 1)[0]) == 1)
check("c", "LSB-first: 2-bit [3,1,0,2,...] packs to 0x87 low byte",
      int(pack_codes(np.array([3, 1, 0, 2] + [0] * 12, dtype=np.uint32), 2)[0]) == 0x87)

# ---------------------------------------------------------------------------
# (e) rejection: cols != groups*group_size must raise ValueError
# (inputs: the stashed in_proj_a pair)
# ---------------------------------------------------------------------------
for kname, kern in (("sign_election", sign_election), ("promotion", promotion)):
    expects_valueerror("e", f"{kname}: codes trimmed 32 cols -> ValueError",
                       lambda k=kern: k(qa[:, :-32], s4, bi4, qb[:, :-32], s5, bi5))
    expects_valueerror("e", f"{kname}: scales/biases trimmed one group -> ValueError",
                       lambda k=kern: k(qa, s4[:, :-1], bi4[:, :-1], qb, s5[:, :-1], bi5[:, :-1]))
    expects_valueerror("e", f"{kname}: codes shape mismatch a vs b -> ValueError",
                       lambda k=kern: k(qa[:-1], s4, bi4, qb, s5, bi5))
    expects_valueerror("e", f"{kname}: 2-bit codes rejected (max>1) -> ValueError",
                       lambda k=kern: k(np.full_like(qa, 2), s4, bi4, qb, s5, bi5))
    expects_valueerror("e", f"{kname}: broken source bias invariant -> ValueError",
                       lambda k=kern: k(qa, s4, np.negative(s4), qb, s5, bi5))

# ---------------------------------------------------------------------------
# probes: validation gaps beyond the spec'd checks (reported, informational)
# ---------------------------------------------------------------------------
print("\n-- gap probes (informational) --")
try:
    _, sg, bg = promotion(qa, s4[:-1], bi4[:-1], qb, s5[:-1], bi5[:-1])
    print(f"[GAP ] promotion accepts scales with mismatched ROW count "
          f"(codes rows {qa.shape[0]}, scales rows {sg.shape[0]}): emits "
          f"inconsistent triple, no error")
except ValueError:
    print("[ok  ] row-count mismatch rejected")
try:
    neg = qa.astype(np.int8).copy()
    neg[0, 0] = -1
    cneg, _, _ = promotion(neg, s4, bi4, qb, s5, bi5)
    print(f"[GAP ] negative code -1 passes validation (max()<=1 misses sign); "
          f"promotion emits code {int(cneg[0, 0])} (>2) which would corrupt "
          f"neighboring codes when packed at 2 bits")
except ValueError as e:
    print(f"[ok  ] negative codes rejected: {e}")

# ---------------------------------------------------------------------------
fails = [(s, n, d) for s, n, ok, d in RESULTS if not ok]
print(f"\n{'=' * 70}\n{len(RESULTS)} checks, {len(fails)} failures")
for s, n, d in fails:
    print(f"  FAIL ({s}) {n} {d}")
print("VERDICT:", "FAIL" if fails else "PASS")
sys.exit(1 if fails else 0)
