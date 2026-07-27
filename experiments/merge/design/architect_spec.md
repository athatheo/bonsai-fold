# merge.py — Phase 3 merge design (sign_election + promotion)

Verified against live sources: /Users/atheocharis/repos/bonsai-fold/src/bonsaifold/stio.py, src/bonsaifold/fold.py, tests/conftest.py, tests/test_fold.py. Baseline suite green (374 passed, 12.5 s, no model load). All numerics are numpy-only; merge.py imports nothing from mlx and never calls mx.quantize.

## 0. Files touched
- `src/bonsaifold/stio.py`: add one function, `pack_codes(codes, bits)` — the exact inverse of `unpack_codes`, placed next to it (verbatim from the verified contract):
```python
def pack_codes(codes, bits):            # codes integer in [0, 2**bits)
    per_word = 32 // bits
    c = codes.astype(np.uint32).reshape(*codes.shape[:-1], -1, per_word)
    shifts = np.arange(per_word, dtype=np.uint32) * bits
    return (c << shifts).sum(axis=-1, dtype=np.uint32)
```
- `src/bonsaifold/merge.py`: new module (everything below).
- `tests/test_merge.py` + a `merge_pack` fixture in `tests/conftest.py`; one round-trip test added to `tests/test_stio.py`.
No changes to fold.py or loader.py — merge.py reuses `write_pack`, `rewrite_folded_config`, `SIDECAR_FILES` semantics via `write_pack`, and `PackReader`/`reindex_name`/`block_owner`/`infer_bits` as-is.

## 1. Public API

```python
# src/bonsaifold/merge.py
"""Phase 3: format-native merging of an adjacent-ish same-type block pair.
Host = i (keeps its position), absorbed = j (removed). Survivors byte-identical;
merged tensors are NEW, built by closed-form rules only. Load results with
bonsaifold.loader.load_bonsai; speculative decoding must stay off."""

OPERATORS = {"sign_election": sign_election, "promotion": promotion}

def sign_election(codes_a, scales_a, biases_a, codes_b, scales_b, biases_b,
                  group_size=128):
    """Binary pair -> binary block. Inputs: UNPACKED codes [rows, cols] in {0,1}
    (any int dtype), f16 scales/biases [rows, cols//group_size]. Returns
    (codes_new {0,1} uint8 [rows, cols], scales_new f16, biases_new f16).
    Pure; never mutates inputs (PackReader arrays are read-only np.frombuffer)."""

def promotion(codes_a, scales_a, biases_a, codes_b, scales_b, biases_b,
              group_size=128):
    """Binary pair -> ONE 2-bit block. Same input contract. Returns
    (codes_new {0,1,2} uint8 [rows, cols], scales_new f16, biases_new f16
    with biases == -scales exactly)."""

def merge_fp_mean(a, b):
    """All FP-param rules reduce to this: fp32 arithmetic mean, cast f16.
    return ((a.astype(np.float32) + b.astype(np.float32)) / 2).astype(np.float16)"""

def merge_blocks(src_pack, dst_pack, i, j, operator, chunk_rows=4096):
    """Write a folded pack with blocks i and j replaced by one merged block at
    position i. operator in OPERATORS. Returns block_map (old->new); provenance
    goes into config["bonsai_fold"] via rewrite_folded_config. dst must not exist."""

def verify_merge(src_pack, dst_pack, block_map, i, j, operator):
    """Two-part check (fold.verify_byte_identity fails on merges by design).
    Returns report dict with 'ok' plus per-part detail (see section 5)."""
```
Both kernels are SYMMETRIC ELEMENTWISE combines (no donor-row selection). This discharges the head/row-alignment caveat by construction: index k of A_log/dt_bias/conv1d channels corresponds to index k in both blocks under exactly the same average-with-average scheme the quantized tensors use. No donor-select code path exists in v1.

## 2. Operator kernels — exact numerics (all in float32; inputs validated first)

Shared preamble (both kernels):
- Validate: `codes_a.shape == codes_b.shape`; `scales_a.shape == scales_b.shape == biases shapes`; `codes.shape[-1] == scales.shape[-1] * group_size`; codes values ⊆ {0,1} (`codes.max() <= 1`); 1-bit source invariant `biases_x == (-(scales_x.astype(np.float32))/2).astype(np.float16)` elementwise for both inputs (this IS "biases == f16(-scales/2)", re-verified true on the live pack). Any failure -> `ValueError` with tensor context added by the caller.
- Sign/magnitude domain: `t = (2*codes - 1)` as int8; per-group magnitude `s_g = scales.astype(np.float32) / 2` (identity s_g = s_mlx/2). biases are consumed only by the invariant check — they carry no independent information.

**sign_election** (binary -> binary):
1. `W = (rep(s_a_g)*t_a + rep(s_b_g)*t_b) * 0.5` where `rep` = `.repeat(group_size, axis=-1)`; float32 [rows, cols].
2. Per g128 group absmean: `s_new_g = np.abs(W).reshape(rows, -1, group_size).mean(-1)` (float32 [rows, cols/gs]).
3. New sign with ties (exact zeros, which occur when s_a_g == s_b_g and t_a == -t_b) broken toward t_a: `t_new = np.where(W > 0, 1, np.where(W < 0, -1, t_a))`; `codes_new = ((t_new + 1) >> 1).astype(np.uint8)`.
4. Output must satisfy the pack's 1-bit affine convention reproducing ±s_new_g: `scales_new = (2.0 * s_new_g).astype(np.float16)`; then `biases_new = (-(scales_new.astype(np.float32)) / 2).astype(np.float16)` — i.e. biases computed FROM the already-rounded f16 scales, so the census invariant `biases == f16(-scales/2)` holds bit-exactly by construction. (Do NOT compute biases from the f32 s_new_g independently; double-rounding could break the invariant.)
- Degenerate all-cancel group (W == 0 across a whole group) yields scales 0 / biases -0.0; valid f16, dequants to zeros. Documented, tested, not an error.

**promotion** (binary pair -> one 2-bit block):
1. Agreement trit t' = t_a where t_a == t_b else 0, and q' = t'+1. Branchless identity (prove once, use everywhere): since q ∈ {0,1}, `codes_new = (codes_a + codes_b).astype(np.uint8)` — agree-at-1 gives 2, agree-at-0 gives 0, disagree gives 1. Code 3 is unreachable by construction.
2. `s_new_g = (s_a_g + s_b_g) / 2`, i.e. `scales_new = ((scales_a.astype(np.float32) + scales_b.astype(np.float32)) / 4).astype(np.float16)` (stored 2-bit scale IS s', dequant {−s′,0,+s′}).
3. `biases_new = np.negative(scales_new)` — f16 sign-bit flip, exact, preserving the verified 2-bit invariant biases == -scales.

## 3. FP-tensor merge
Every non-quantized F16 tensor owned by block i is merged with its block-j counterpart via `merge_fp_mean` (fp32 accumulate, f16 cast), per the approved semantics rules — they ALL reduce to arithmetic mean of the stored tensor: A_log (stored value is already log-space, so arithmetic mean == geometric mean of λ), dt_bias, conv1d.weight (elementwise in stored MLX layout [10240,4,1] — no moveaxis, no sanitize +1 shift), linear_attn.norm.weight, input/post_attention_layernorm.weight, and (full-attn pairs) q_norm/k_norm.weight — the q/k-norm conditional (head-channel correspondence) is satisfied because the kernels are symmetric elementwise. Identification: a block-i tensor is FP iff `not is_quantized(name, header)` and it is not a `.scales`/`.biases` member of a triple; equivalently, handle triples keyed by their `.weight` name and mean everything else. The dt_bias resting-decay re-derivation from the caveats is explicitly OUT OF SCOPE for v1 (note it in the module docstring as a follow-up if merged-pair KL screens show long-context drift).

## 4. merge_blocks pipeline

Validation (all `ValueError` unless noted):
- `0 <= i < j < n` where n = `src.config["text_config"]["num_hidden_layers"]` (merged block sits at the pair's FIRST position; require i < j, do not silently swap).
- `layer_types[i] == layer_types[j]` — same-type merges only (rewrite_folded_config derives the merged block's type from the host index; a cross-type merge would make layer_types lie).
- `operator in OPERATORS`.
- For every quantized base in block i: counterpart base exists in block j; `quant_meta` returns `(128, 1)` for BOTH (bits inferred from shapes; refuse non-1-bit or non-g128 inputs — operators are defined only on the 1-bit pack); weight/scales/biases shapes match across the pair.
- FP tensors: counterpart exists in block j with identical dtype/shape.
- dst existing -> `FileExistsError` (inherited from `write_pack`, mkdir exist_ok=False).

Flow:
1. `src = PackReader(src_pack)`; run all validation up front (fail before creating dst).
2. `block_map = {old: new for new, old in enumerate(k for k in range(n) if k != j)}` — identical construction to `drop_blocks(drop=[j])`; host i keeps `block_map[i] = i` (i < j always).
3. Precompute merged payloads for block i into `merged: {src_name: (st_dtype, shape, bytes)}`:
   - Per quantized triple (host base P_i.x, partner P_j.x), inside `_merge_quant_tensor(src, base_i, base_j, kernel, chunk_rows)`:
     read packed U32 + f16 scales/biases for both; loop over row chunks `rows[r:r+chunk_rows]` (the group grid tiles the LAST/column axis, so any row partition is group-aligned): `unpack_codes(w_chunk, 1)` -> kernel -> `pack_codes(codes_new, out_bits)`; concatenate chunk outputs. out_bits = 1 (sign_election) or 2 (promotion). Emit three payloads: weight U32 [out, in*out_bits/32] (2-bit: in/16, e.g. gate_proj [17408, 320] — NOT in/8), scales F16 [out, in/128], biases F16 [out, in/128], via `np.ascontiguousarray(arr).tobytes()`.
     RAM: largest tensor is gate/up [17408, 5120]; chunk_rows=4096 keeps the f32 W chunk at ≤ 4096×17408×4 B ≈ 285 MB (down_proj rows), a few temporaries -> peak well under 2 GB. Merged block materialized in full is ≤ ~350 MB (2-bit) — fine to hold in `merged`.
   - Per FP tensor: `merge_fp_mean(src.read(name_i), src.read(name_j))` (np.frombuffer arrays are read-only; merge_fp_mean allocates, never mutates).
4. Build `entries` by iterating `src.header` IN ORDER (keeps on-disk order diffable vs drop packs): `new_name = reindex_name(name, block_map)`; skip None (block j vanishes); if `block_owner(name) == i` substitute the merged payload `{"dtype": d, "shape": shape, "raw": payload_bytes}` (nbytes optional for bytes); else stream byte-identical with the default-arg closure exactly as drop_blocks: `{"dtype": ..., "shape": ..., "nbytes": end-begin, "raw": (lambda n=name: src.read_raw(n))}`. This passes through the whole tail and all 333 vision_tower tensors untouched.
5. Config: `extra = None` for sign_election; for promotion, `extra = {reindex_name(base_i, block_map): {"group_size": 128, "bits": 2} for each quantized base of block i}` — key is the base name WITHOUT ".weight", carrying the NEW post-fold index (host's new index == i), landing verbatim in TOP-LEVEL config["quantization"] only (never text_config), which is exactly what both stio.quant_params and mlx-lm's class_predicate read. Then `config = rewrite_folded_config(src.config, block_map, {"operation": "merge", "merged_pairs": [[i, j]], "rule": operator, "source_pack": str(src.pack_dir)}, extra_quant_overrides=extra)` — this also trims num_hidden_layers/layer_types and stamps created_utc + the load_bonsai note. full_attention_interval stays 4 and is positionally wrong: folded pack is loadable ONLY via bonsaifold.loader.load_bonsai.
6. `write_pack(dst_pack, entries, config, src.pack_dir)`; return `block_map`.

## 5. verify_merge (fold.verify_byte_identity fails on merge packs by design)
Part (a) — survivors byte-identical: build `expected = {reindex_name(name): name}` over src.header EXCLUDING names with `block_owner(name) in (i, j)`; require dst.header == expected ∪ merged-block names (missing/extra computed accordingly) and every expected tensor's read_raw bytes equal.
Part (b) — merged-tensor value checks, per quantized base at dst index i:
- shapes: `infer_bits(weight.shape, scales.shape, 128) == (1 | 2)` per operator; scales.shape == biases.shape == [out, in/128].
- codes: `unpack_codes(...).max()` ≤ 1 (sign_election) or ≤ 2 (promotion) — code 3 never present.
- invariants: sign_election `biases == (-(scales.astype(np.float32))/2).astype(np.float16)` elementwise; promotion `biases == np.negative(scales)` exact.
- exact recompute: rerun the kernel from the SOURCE pair (chunked) and require bit-equality of codes/scales/biases — closed-form rules are deterministic, so equality is exact, not approximate.
- config: promotion override present per base at new index with {"group_size":128,"bits":2}; sign_election adds NO overrides; num_hidden_layers == n-1; layer_types[i] == source layer_types[i].
- FP tensors: recompute merge_fp_mean from source, require byte equality.
Report: {"survivors_ok", "missing", "extra", "mismatched", "merged_checks": {base: {...}}, "ok"}.

## 6. Error handling summary
ValueError: i/j out of range or i >= j; cross-type pair; unknown operator; source tensor not 1-bit g128 (quant_meta/infer_bits); pair shape mismatch (weight, scales, biases, or FP); missing counterpart tensor in block j; source biases invariant violated (refuse to merge a nonconforming tensor); write_safetensors payload/size mismatch (existing behavior). FileExistsError: dst exists. No silent fallbacks, no partial packs (validate everything before mkdir).

## 7. Test plan (tests/test_merge.py; NO model load, numpy-only, seconds to run)
Fixture `merge_pack` in conftest.py (alongside mini_pack, which lacks biases and quant-consistent shapes): 8 blocks with LAYER_TYPES as-is; per block one quantized triple `...layers.{k}.mlp.up_proj` with out=8, in=256 (2 groups of 128: w U32 [8,8], s/b F16 [8,2], biases = f16(-scales/2) BY CONSTRUCTION) plus one FP tensor `...layers.{k}.input_layernorm.weight` F16 [16]; tail lm_head + one vision tensor; config with scalar quantization {group_size:128, bits:1}; tokenizer_config.json sidecar.

Kernel unit tests (group_size=4 for hand-checkability — kernels take gs as a parameter):
- sign_election worked example, 1 row × 8 cols, 2 groups: q_a=[1,0,1,1,0,0,1,0], scales_a=[1.0,0.5] (s_a_g=[0.5,0.25]); q_b=[1,1,0,1,0,1,1,1], scales_b=[0.5,1.0]. Expected W=[.375,-.125,.125,.375, -.375,.125,.375,.125] signed as computed; absmean per group = 0.25 both -> scales_new=[0.5,0.5], biases_new=[-0.25,-0.25], codes_new=[1,0,1,1,0,1,1,1]. Assert exact.
- tie-break: scales_a == scales_b, t_a == -t_b at one position -> W=0 there -> code equals code_a.
- all-cancel group: entire group W==0 -> scales 0.0, biases -0.0, no crash.
- promotion identity on the same example: codes_new == q_a + q_b == [2,1,1,2,0,1,2,1]; scales_new = f16((scales_a+scales_b)/4) = [0.375,0.375]; biases == -scales exact; 3 not in codes.
- input validation: shape mismatch, codes containing 2, broken source biases invariant -> ValueError; inputs not mutated (pass read-only arrays via arr.setflags(write=False)).
Property tests (rng-seeded, realistic small shapes, gs=128, in=256):
- self-merge exactness: sign_election(A, A) returns A's codes and scales bit-identically (W=±s_g, absmean=s_g, f16 round-trip exact); promotion(A, A) has manual_dequant equal to A's manual_dequant exactly (codes 2q_a, s'=s_g).
- pack_codes/unpack_codes round-trip for bits ∈ {1,2}: random codes -> pack -> unpack identity, and random U32 -> unpack -> pack identity (goes in test_stio.py).
- dequant consistency: manual_dequant of each kernel's packed output == rep(s'_g) * t' elementwise (float32 exact given f16 params).
- chunking invariance: _merge_quant_tensor with chunk_rows=3 (non-divisor) equals chunk_rows=10**9 bit-exactly.
Integration tests on merge_pack:
- promotion merge_blocks(pack, dst, 0, 1): block_map {0:0, 2:1, ..., 7:6}; num_hidden_layers 7; layer_types trimmed with host's type at 0; top-level override "language_model.model.layers.0.mlp.up_proj" == {"group_size":128,"bits":2} and absent from text_config; written weight shape [8,16] (in/16), infer_bits==2; verify_merge(...)["ok"].
- sign_election merge_blocks(pack, dst, 4, 5): no overrides added; infer_bits==1; biases==f16(-scales/2); FP tensor equals merge_fp_mean of sources; all survivors + tail + vision byte-identical to source under reindexed names.
- verify_merge catches corruption: flip a byte in a survivor payload -> mismatched; overwrite a merged scales value -> merged_checks fail; drop the config override -> fail.
- validation: (i,j)=(3,3), (5,4), (2,3) cross-type (linear vs full), out-of-range 99, existing dst -> correct exception types; a pre-promoted 2-bit source tensor (fixture variant) -> ValueError.
Run `uv run pytest` (full suite) — must stay green; baseline is 374 passed.

## 8. Non-goals / notes for the caller
- One pair per pack in v1 (merged_pairs is a list for stamp-format stability, but merge_blocks takes a single (i, j); multi-pair packs can chain merge_blocks->merge_blocks later if needed, at the cost of re-reading — do not build the abstraction now).
- Downstream evaluation (not merge.py's job): load via load_bonsai only, speculative decoding OFF, KL screens vs unfolded 1-bit baseline (baseline row always included), watch long-context specifically for merged pairs (both FP means bias one-sidedly toward slower forgetting), and prefer shortlist pairs with small per-head |A_log_i - A_log_j| — expose those deltas via a trivial `report_alog_gap(src, i, j)` helper only if the screening notebook asks for it, not preemptively.
- Never re-encode survivors; never call mx.quantize; merged tensors are the only new bytes in the pack.