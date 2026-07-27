"""Merge operators: hand-computed kernel cases, property tests, and
end-to-end pack surgery on the synthetic merge_pack. No model load."""
import numpy as np
import pytest

from bonsaifold.merge import (
    merge_blocks,
    merge_fp_mean,
    promotion,
    sign_election,
    verify_merge,
)
from bonsaifold.stio import (
    PackReader,
    infer_bits,
    manual_dequant,
    pack_codes,
    unpack_codes,
)
from conftest import write_merge_pack

# --- worked example from the design spec (group_size=4 for hand-checkability)
Q_A = np.array([[1, 0, 1, 1, 0, 0, 1, 0]], dtype=np.uint8)
Q_B = np.array([[1, 1, 0, 1, 0, 1, 1, 1]], dtype=np.uint8)
S_A = np.array([[1.0, 0.5]], dtype=np.float16)
S_B = np.array([[0.5, 1.0]], dtype=np.float16)
B_A = (-(S_A.astype(np.float32)) / 2).astype(np.float16)
B_B = (-(S_B.astype(np.float32)) / 2).astype(np.float16)


def test_sign_election_worked_example():
    codes, scales, biases = sign_election(Q_A, S_A, B_A, Q_B, S_B, B_B, group_size=4)
    assert codes.tolist() == [[1, 0, 1, 1, 0, 1, 1, 1]]
    assert scales.tolist() == [[0.5, 0.5]]
    assert biases.tolist() == [[-0.25, -0.25]]
    assert codes.dtype == np.uint8 and scales.dtype == np.float16


def test_sign_election_tie_breaks_toward_a():
    # equal magnitudes, opposite signs at every position -> all W == 0
    q_a = np.array([[1, 0, 1, 0]], dtype=np.uint8)
    q_b = 1 - q_a
    s = np.array([[1.0]], dtype=np.float16)
    b = (-(s.astype(np.float32)) / 2).astype(np.float16)
    codes, scales, biases = sign_election(q_a, s, b, q_b, s, b, group_size=4)
    assert codes.tolist() == q_a.tolist()  # ties -> t_a
    # all-cancel group: scales 0.0, biases -0.0, dequants to zeros, no crash
    assert scales.tolist() == [[0.0]]
    t = 2 * codes.astype(np.float32) - 1  # direct dequant (8 cols < one u32 word)
    deq = t * (scales.astype(np.float32) / 2).repeat(4, axis=-1)
    assert np.all(deq == 0.0)


def test_promotion_worked_example():
    codes, scales, biases = promotion(Q_A, S_A, B_A, Q_B, S_B, B_B, group_size=4)
    assert codes.tolist() == [[2, 1, 1, 2, 0, 1, 2, 1]]
    assert scales.tolist() == [[0.375, 0.375]]
    assert np.array_equal(biases, np.negative(scales))
    assert 3 not in codes


def test_promotion_disagreement_dequants_to_zero():
    codes, scales, biases = promotion(Q_A, S_A, B_A, Q_B, S_B, B_B, group_size=4)
    trit = codes.astype(np.float32) - 1  # direct dequant (8 cols < one u32 word)
    deq = trit * scales.astype(np.float32).repeat(4, axis=-1)
    disagree = Q_A[0] != Q_B[0]
    assert np.all(deq[0][disagree] == 0.0)
    assert np.all(deq[0][~disagree] != 0.0)


@pytest.mark.parametrize("kernel,bits", [(sign_election, 1), (promotion, 2)])
def test_self_merge_exactness(kernel, bits):
    # exact for normal-range scales; NOT for ~2048 subnormal/min-normal f16
    # scales where s != 2*f16(s/2) (60 of 210M groups in the real pack,
    # per-weight error <= 2^-25 — see verify_algebra.py finding 1)
    rng = np.random.default_rng(1)
    codes = rng.integers(0, 2, size=(6, 256), dtype=np.uint8)
    scales = rng.uniform(0.25, 4.0, size=(6, 2)).astype(np.float16)
    biases = (-(scales.astype(np.float32)) / 2).astype(np.float16)
    cn, sn, bn = kernel(codes, scales, biases, codes, scales, biases)
    src_deq = manual_dequant(pack_codes(codes, 1), scales, biases, 1, 128)
    new_deq = manual_dequant(pack_codes(cn, bits), sn, bn, bits, 128)
    assert np.array_equal(src_deq, new_deq)  # self-merge is a dequant no-op
    if bits == 1:
        assert np.array_equal(cn, codes) and np.array_equal(sn, scales)


def test_dequant_consistency_random():
    rng = np.random.default_rng(2)
    q_a = rng.integers(0, 2, size=(4, 256), dtype=np.uint8)
    q_b = rng.integers(0, 2, size=(4, 256), dtype=np.uint8)
    s_a = rng.uniform(0.5, 2.0, size=(4, 2)).astype(np.float16)
    s_b = rng.uniform(0.5, 2.0, size=(4, 2)).astype(np.float16)
    b_a = (-(s_a.astype(np.float32)) / 2).astype(np.float16)
    b_b = (-(s_b.astype(np.float32)) / 2).astype(np.float16)
    cn, sn, bn = sign_election(q_a, s_a, b_a, q_b, s_b, b_b)
    deq = manual_dequant(pack_codes(cn, 1), sn, bn, 1, 128)
    t_new = 2 * cn.astype(np.float32) - 1
    expect = t_new * (sn.astype(np.float32) / 2).repeat(128, axis=-1)
    assert np.allclose(deq, expect, rtol=0, atol=0)  # exact given f16 params


def test_pack_unpack_roundtrip():
    rng = np.random.default_rng(3)
    for bits in (1, 2):
        codes = rng.integers(0, 2**bits, size=(5, 128), dtype=np.uint8)
        assert np.array_equal(unpack_codes(pack_codes(codes, bits), bits), codes)
        words = rng.integers(0, 2**32, size=(5, 4), dtype=np.uint32)
        assert np.array_equal(pack_codes(unpack_codes(words, bits), bits), words)


def test_kernel_input_validation():
    with pytest.raises(ValueError, match="shape mismatch"):
        sign_election(Q_A, S_A, B_A, Q_B[:, :4], S_B, B_B, group_size=4)
    with pytest.raises(ValueError, match="not 1-bit"):
        promotion(Q_A + 1, S_A, B_A, Q_B, S_B, B_B, group_size=4)
    with pytest.raises(ValueError, match="not 1-bit"):  # negative codes too
        promotion(Q_A.astype(np.int8) - 1, S_A, B_A, Q_B, S_B, B_B, group_size=4)
    with pytest.raises(ValueError, match="rows"):
        sign_election(Q_A, np.vstack([S_A, S_A]), np.vstack([B_A, B_A]),
                      Q_B, np.vstack([S_B, S_B]), np.vstack([B_B, B_B]), group_size=4)
    with pytest.raises(ValueError, match="invariant"):
        sign_election(Q_A, S_A, np.negative(S_A), Q_B, S_B, B_B, group_size=4)
    # read-only inputs (PackReader arrays are np.frombuffer) must not be mutated
    q = Q_A.copy(); q.setflags(write=False)
    s = S_A.copy(); s.setflags(write=False)
    b = B_A.copy(); b.setflags(write=False)
    sign_election(q, s, b, q, s, b, group_size=4)
    promotion(q, s, b, q, s, b, group_size=4)


def test_merge_fp_mean():
    a = np.array([1.0, 2.0], dtype=np.float16)
    b = np.array([2.0, 5.0], dtype=np.float16)
    out = merge_fp_mean(a, b)
    assert out.dtype == np.float16 and out.tolist() == [1.5, 3.5]
    # fp32 accumulation: naive f16 60000+60000 overflows to inf
    big = np.array([60000.0], dtype=np.float16)
    assert merge_fp_mean(big, big).tolist() == [60000.0]


# --- end-to-end pack surgery ---------------------------------------------


def _first_pair(types, want):
    for i in range(len(types) - 1):
        for j in range(i + 1, len(types)):
            if types[i] == types[j] == want:
                return i, j
    raise AssertionError


def test_merge_blocks_promotion(merge_pack, tmp_path):
    dst = tmp_path / "merged_promo"
    block_map = merge_blocks(merge_pack, dst, 0, 1, "promotion")
    assert block_map == {0: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6}
    out = PackReader(dst)
    assert out.config["text_config"]["num_hidden_layers"] == 7
    base = "language_model.model.layers.0.mlp.up_proj"
    assert out.config["quantization"][base] == {"group_size": 128, "bits": 2}
    assert base not in out.config["text_config"].get("quantization", {})
    w, s = out.read(base + ".weight"), out.read(base + ".scales")
    assert w.shape == (8, 16) and infer_bits(w.shape, s.shape, 128) == 2
    assert out.config["bonsai_fold"]["operation"] == "merge"
    report = verify_merge(merge_pack, dst, block_map, 0, 1, "promotion")
    assert report["ok"], report


def test_merge_blocks_sign_election(merge_pack, tmp_path):
    src = PackReader(merge_pack)
    types = src.config["text_config"]["layer_types"]
    i, j = 4, 5
    assert types[i] == types[j]
    dst = tmp_path / "merged_se"
    block_map = merge_blocks(merge_pack, dst, i, j, "sign_election")
    out = PackReader(dst)
    base = f"language_model.model.layers.{i}.mlp.up_proj"
    # no override added; output still 1-bit with the census biases invariant
    assert base not in out.config["quantization"]
    w, s, b = (out.read(base + sfx) for sfx in (".weight", ".scales", ".biases"))
    assert infer_bits(w.shape, s.shape, 128) == 1
    assert np.array_equal(b, (-(s.astype(np.float32)) / 2).astype(np.float16))
    # FP tensor is the fp mean of the sources
    fp = f"language_model.model.layers.{i}.input_layernorm.weight"
    expect = merge_fp_mean(
        src.read(fp), src.read(fp.replace(f".{i}.", f".{j}."))
    )
    assert np.array_equal(out.read(fp), expect)
    report = verify_merge(merge_pack, dst, block_map, i, j, "sign_election")
    assert report["ok"], report
    # survivors byte-identical incl. tail + vision
    assert report["mismatched"] == [] and report["missing"] == []


def _flip_tensor_byte(dst, name):
    pack_path = dst / "model.safetensors"
    data = bytearray(pack_path.read_bytes())
    out = PackReader(dst)
    _, header, data_start = out._src[name]
    begin = header[name]["data_offsets"][0]
    data[data_start + begin] ^= 0xFF
    pack_path.write_bytes(bytes(data))


def test_verify_merge_catches_merged_corruption(merge_pack, tmp_path):
    dst = tmp_path / "merged_c"
    block_map = merge_blocks(merge_pack, dst, 0, 1, "promotion")
    _flip_tensor_byte(dst, "language_model.model.layers.0.mlp.up_proj.scales")
    report = verify_merge(merge_pack, dst, block_map, 0, 1, "promotion")
    assert not report["ok"]


def test_verify_merge_catches_survivor_corruption(merge_pack, tmp_path):
    dst = tmp_path / "merged_s"
    block_map = merge_blocks(merge_pack, dst, 0, 1, "promotion")
    _flip_tensor_byte(dst, "language_model.model.layers.3.mlp.up_proj.weight")
    report = verify_merge(merge_pack, dst, block_map, 0, 1, "promotion")
    assert not report["ok"] and report["mismatched"]


def test_verify_merge_catches_dropped_override(merge_pack, tmp_path):
    import json as _json

    dst = tmp_path / "merged_o"
    block_map = merge_blocks(merge_pack, dst, 0, 1, "promotion")
    cfg = _json.loads((dst / "config.json").read_text())
    del cfg["quantization"]["language_model.model.layers.0.mlp.up_proj"]
    (dst / "config.json").write_text(_json.dumps(cfg))
    report = verify_merge(merge_pack, dst, block_map, 0, 1, "promotion")
    assert not report["ok"]


def test_index_total_size(merge_pack, tmp_path):
    import json as _json

    dst = tmp_path / "merged_t"
    merge_blocks(merge_pack, dst, 0, 1, "promotion")
    idx = _json.loads((dst / "model.safetensors.index.json").read_text())
    out = PackReader(dst)
    actual = sum(
        info["data_offsets"][1] - info["data_offsets"][0] for info in out.header.values()
    )
    assert idx["metadata"]["total_size"] == actual


def test_merge_blocks_validation(merge_pack, tmp_path):
    with pytest.raises(ValueError, match="0 <= i < j"):
        merge_blocks(merge_pack, tmp_path / "x1", 3, 3, "promotion")
    with pytest.raises(ValueError, match="0 <= i < j"):
        merge_blocks(merge_pack, tmp_path / "x2", 5, 4, "promotion")
    with pytest.raises(ValueError, match="0 <= i < j"):
        merge_blocks(merge_pack, tmp_path / "x3", 0, 99, "promotion")
    with pytest.raises(ValueError, match="cross-type"):
        merge_blocks(merge_pack, tmp_path / "x4", 2, 3, "promotion")  # linear vs full
    with pytest.raises(ValueError, match="unknown operator"):
        merge_blocks(merge_pack, tmp_path / "x5", 0, 1, "ternary_election")
    dst = tmp_path / "exists"
    merge_blocks(merge_pack, dst, 0, 1, "promotion")
    with pytest.raises(FileExistsError):
        merge_blocks(merge_pack, dst, 0, 1, "promotion")


def test_merge_blocks_refuses_two_bit_source(tmp_path):
    # both members 2-bit so shapes match and the bits guard itself must fire
    pack = write_merge_pack(tmp_path / "twobit_src", two_bit_blocks=(0, 1))
    with pytest.raises(ValueError, match="1-bit g128"):
        merge_blocks(pack, tmp_path / "out", 0, 1, "promotion")
    # mismatched pair (one 1-bit, one 2-bit) is also refused, at the shape check
    pack2 = write_merge_pack(tmp_path / "mixed_src", two_bit_blocks=(1,))
    with pytest.raises(ValueError, match="shape mismatch"):
        merge_blocks(pack2, tmp_path / "out2", 0, 1, "promotion")


def test_chunking_invariance(merge_pack, tmp_path):
    a = merge_blocks(merge_pack, tmp_path / "chunk3", 0, 1, "promotion", chunk_rows=3)
    b = merge_blocks(merge_pack, tmp_path / "chunkbig", 0, 1, "promotion",
                     chunk_rows=10**9)
    assert a == b
    pa = (tmp_path / "chunk3" / "model.safetensors").read_bytes()
    pb = (tmp_path / "chunkbig" / "model.safetensors").read_bytes()
    assert pa == pb  # bit-exact regardless of chunking
