MERGE.PY IMPLEMENTATION CONTRACT (verified against /Users/atheocharis/repos/bonsai-fold/src/bonsaifold/stio.py, src/bonsaifold/fold.py, src/bonsaifold/loader.py, experiments/census/census_Bonsai-27B-mlx-1bit.md, models/Bonsai-27B-mlx-1bit/config.json, and installed mlx_lm.utils.load_model source)

== (1) TENSOR-NAME TEMPLATES (prefix P = language_model.model.layers.{i}) ==
Quantized tensors are TRIPLES: {base}.weight (U32 packed), {base}.scales (F16), {base}.biases (F16). Shapes below as [out, in] logical; stored weight is [out, in*bits/32], scales/biases [out, in/128].

LINEAR-ATTENTION block (30 tensors; 8 quantized triples + 6 FP):
- P.linear_attn.in_proj_qkv  out=10240 in=5120  (w [10240,160], s/b [10240,40])
- P.linear_attn.in_proj_a    out=48    in=5120  (w [48,160],    s/b [48,40])
- P.linear_attn.in_proj_b    out=48    in=5120  (w [48,160],    s/b [48,40])
- P.linear_attn.in_proj_z    out=6144  in=5120  (w [6144,160],  s/b [6144,40])
- P.linear_attn.out_proj     out=5120  in=6144  (w [5120,192],  s/b [5120,48])
- P.mlp.gate_proj / P.mlp.up_proj  out=17408 in=5120 (w [17408,160], s/b [17408,40])
- P.mlp.down_proj            out=5120  in=17408 (w [5120,544],  s/b [5120,136])
FP16: P.input_layernorm.weight [5120]; P.post_attention_layernorm.weight [5120]; P.linear_attn.A_log [48]; P.linear_attn.dt_bias [48]; P.linear_attn.conv1d.weight [10240,4,1]; P.linear_attn.norm.weight [128]

FULL-ATTENTION block (25 tensors; 7 quantized triples + 4 FP):
- P.self_attn.q_proj  out=12288 in=5120 (w [12288,160], s/b [12288,40])
- P.self_attn.k_proj / P.self_attn.v_proj  out=1024 in=5120 (w [1024,160], s/b [1024,40])
- P.self_attn.o_proj  out=5120 in=6144 (w [5120,192], s/b [5120,48])
- P.mlp.{gate,up,down}_proj  same as linear block
FP16: P.input_layernorm.weight [5120]; P.post_attention_layernorm.weight [5120]; P.self_attn.q_norm.weight [256]; P.self_attn.k_norm.weight [256]

Tail (pass through raw): language_model.model.embed_tokens.{weight,scales,biases} ([248320,160]/[248320,40]), language_model.lm_head.{weight,scales,biases} (same shapes), language_model.model.norm.weight F16 [5120]. Vision tower: 333 tensors under vision_tower.* — pass through raw; reindex_name() passes vision/non-layer names through unchanged.

== (2) READING ONE QUANTIZED TENSOR (no model load; all from bonsaifold.stio) ==
    from bonsaifold.stio import PackReader, unpack_codes, manual_dequant
    r = PackReader("models/Bonsai-27B-mlx-1bit")
    base = "language_model.model.layers.12.mlp.gate_proj"
    w = r.read(base + ".weight")    # np.uint32 [out, in*bits/32]
    s = r.read(base + ".scales")    # np.float16 [out, in/128]
    b = r.read(base + ".biases")    # np.float16 [out, in/128]
    gs, bits = r.quant_meta(base + ".weight")   # (128, 1); bits inferred from header shapes via infer_bits, config bits is only a hint
    q = unpack_codes(w, bits)                   # codes [out, in], values {0,1}
    W = manual_dequant(w, s, b, bits, gs)       # float32 [out, in], w = s*q + b
For byte-identical pass-through of survivors use r.read_raw(name) plus r.header[name] ({"dtype","shape","data_offsets"}). For the 1-bit pack: s_g = s_mlx/2, sign = 2*q - 1; biases == f16(-scales/2) elementwise (re-verified True on a live tensor).

== (3) WRITING A 2-BIT TENSOR — EXACT WORD-PACKING CONVENTION ==
stio.unpack_codes is LSB-first within each uint32: per_word = 32//bits codes per word; code slot j of a word occupies bit range [j*bits, (j+1)*bits); words tile the LAST axis in row-major order, so unpacked column c lives in word c//per_word, slot c%per_word. For bits=2: per_word=16, packed shape [out, in/16]. The exact inverse (round-trip verified exact against a real 1-bit tensor and synthetic 2-bit codes):
    def pack_codes(codes, bits):            # codes integer in [0, 2**bits)
        per_word = 32 // bits
        c = codes.astype(np.uint32).reshape(*codes.shape[:-1], -1, per_word)
        shifts = np.arange(per_word, dtype=np.uint32) * bits
        return (c << shifts).sum(axis=-1, dtype=np.uint32)
2-bit code convention: trit t in {-1,0,+1} -> q = t+1 in {0,1,2} (3 unused); scales F16 [out, in/128]; biases = (-scales) computed in float16 (sign-bit flip, exact) so biases == -scales holds exactly; dequant = s*q - s in {-s,0,+s}. Acceptance check after writing: infer_bits(weight.shape, scales.shape, 128) == 2 (e.g. gate_proj 2-bit: w [17408,320], s/b [17408,40]).

== (4) ENTRIES DICT FOR write_pack (fold.write_pack -> stio.write_safetensors) ==
entries: {tensor_name: {"dtype": st_dtype_str, "shape": [ints], "nbytes": int, "raw": bytes | zero-arg-callable -> bytes}}
- dtype strings: "U32" for packed weights, "F16" for scales/biases/FP tensors (see stio.DTYPE_NP).
- "raw" MAY be materialized bytes (np.ascontiguousarray(arr).tobytes(); then "nbytes" is optional — computed as len(raw)). "raw" MAY be a zero-arg callable, invoked one tensor at a time at write time (streaming; peak RAM = largest tensor) — in that case "nbytes" is REQUIRED (write_safetensors does len(e["raw"]) otherwise, which raises TypeError on a callable).
- Payload length is validated against declared size at write time (ValueError on mismatch).
- Survivors: {"dtype": src.header[n]["dtype"], "shape": src.header[n]["shape"], "nbytes": end-begin from data_offsets, "raw": (lambda n=name: src.read_raw(n))} — NOTE the default-arg binding, exactly as fold.drop_blocks does. Merged tensors: new bytes from pack_codes/scales/biases, either materialized or a callable.
- Dict insertion order = on-disk tensor order; keep source header order for diffability. write_pack writes single-shard model.safetensors + model.safetensors.index.json (weight_map all -> "model.safetensors"), config.json (indent=2), and copies SIDECAR_FILES from sidecar_src. dst directory MUST NOT already exist (mkdir exist_ok=False).

== (5) extra_quant_overrides FORMAT + layer_types/CONFIG IMPLICATIONS ==
Override format (one entry per promoted tensor, key = tensor base name WITHOUT ".weight", using NEW post-fold layer indices):
    extra_quant_overrides = {
        "language_model.model.layers.<new_idx>.mlp.gate_proj": {"group_size": 128, "bits": 2},
        ...
    }
Why exactly this: rewrite_folded_config first renames PRE-EXISTING per-tensor override keys through reindex_name(block_map), then applies extra_quant_overrides VERBATIM into the TOP-LEVEL config["quantization"] (setdefault + update) — so extras must already carry new indices. Consumption is dual and both match: (a) stio.PackReader.quant_params looks up config["quantization"][base]; (b) mlx-lm load_model's class_predicate(p, m) checks `if p in config["quantization"]: return config["quantization"][p]` where p is the module path == base name, and nn.quantize uses the returned dict's group_size/bits for that module. text_config["quantization"] holds scalars only ({"group_size":128,"bits":1}) and needs no extras. Current pack has ZERO pre-existing per-tensor overrides at either level; scalar defaults are group_size=128, bits=1 at both levels.
Replacing two blocks with one: build block_map as in drop_blocks — keep the HOST block index, drop the ABSORBED index; write merged tensors under the host's NEW-index names (replacing its raw entries). rewrite_folded_config then sets text_config.num_hidden_layers = len(keep) and layer_types = [old layer_types[i] for i in keep]; the merged block therefore inherits the host's layer_types entry, so merges MUST be same-type (linear+linear or full+full) or layer_types lies. full_attention_interval stays 4 in config and becomes positionally WRONG — the folded pack is only loadable via bonsaifold.loader.load_bonsai (layer_types-aware TypedModel); stock mlx-lm mis-types blocks (census NOTE). Pass an op_stamp dict, e.g. {"operation": "merge", "merged_pairs": [[i, j], ...], "rule": "<closed-form rule id>", "source_pack": str(src.pack_dir)} — rewrite_folded_config adds created_utc + loader note under config["bonsai_fold"].

== (6) VERIFICATION HOOK ==
fold.verify_byte_identity requires EVERY mapped tensor byte-identical, so it fails on merged blocks by construction. merge.py must verify in two parts: (a) byte-identity over block_map with the host block's names excluded from `expected` (all survivors, tail, vision must be byte-identical); (b) separate value-check of merged tensors: unpack_codes round-trip, codes subset of {0,1,2}, biases == -scales exact, shapes satisfy infer_bits == 2, and config override present for each promoted base.

GOTCHAS:
- Packed uint32 weight shape is [out, in*bits/32] — NOT [out, in/8]: 1-bit gives [out, in/32] (in=5120 -> 160 words), 2-bit gives [out, in/16] (in=5120 -> 320). [out, in/8] is a 4-bit convention and would be wrong here.
- Group axis is the LAST (input/column) axis: scales/biases are [out, in/128]; in must be divisible by group_size 128 (hence also by per_word 16 for 2-bit). All block matrices here have in ∈ {5120, 6144, 17408}, all divisible by 128.
- scales and biases must be numpy float16 before .tobytes() and declared dtype 'F16'; compute biases as -(scales.astype(np.float16)) — f16 negation is a sign-bit flip, so biases == -scales holds exactly, preserving the verified 2-bit pack invariant.
- 2-bit codes are q ∈ {0,1,2} with 3 UNUSED (trit = q-1, dequant s*q - s ∈ {-s,0,+s}); emitting code 3 would produce +2s-ish values with no pack precedent and break the census packing-relation check.
- write_safetensors: if 'raw' is a callable, 'nbytes' is mandatory (else len(callable) -> TypeError); if 'raw' is bytes, nbytes is optional. Payload/declared size mismatch raises ValueError at write time.
- Late-binding trap: build streaming closures as (lambda n=name: src.read_raw(n)) with the default-arg capture, exactly like fold.drop_blocks — a bare lambda would read the last tensor 500+ times.
- write_pack refuses an existing destination (mkdir exist_ok=False) — pick a fresh dst or delete first.
- extra_quant_overrides keys: base name sans '.weight', NEW (post-fold) layer indices, and they land only in TOP-LEVEL config['quantization'] — which is exactly what both mlx-lm's class_predicate and stio.quant_params read; do not add '.weight' or old indices, and do not touch text_config['quantization'].
- PackReader.quant_meta infers bits from shapes (config is a hint only), so stio-side reads of a 2-bit tensor work even without the override — but mlx-lm load NEEDS the config override to build the module as bits=2; omitting it makes nn.quantize build a 1-bit QuantizedLinear whose packed shape won't match the stored [out, in/16] weight.
- Merges must be same-type (linear+linear or full+full): rewrite_folded_config derives the merged block's layer_types entry from the surviving host index.
- Folded/merged packs load ONLY via bonsaifold.loader.load_bonsai (config full_attention_interval=4 remains and stock mlx-lm types blocks positionally); also disable speculative decoding on every folded model (project rule, DSpark drafter taps fixed layer indices).
- fold.verify_byte_identity fails on merge packs by design (it demands all mapped tensors identical); verify survivors with the host block excluded and value-check merged tensors separately.
- read_tensor arrays come from np.frombuffer (read-only backing) — np.copy() before any in-place mutation.
- Never re-encode or mx.quantize surviving tensors: pass every non-merged tensor through read_raw byte-identical, including the whole vision_tower.* (333 tensors) and tail (embed_tokens/lm_head/norm).
- 1-bit decode identities for the merge math: s_g = s_mlx/2, sign = 2q-1, w = s_mlx*q + b with b == f16(-s_mlx/2) (re-verified elementwise True on the live pack).
- Sharding: write_pack emits a single-shard model.safetensors (~5.1 GB source is fine on this machine); PackReader reads it back through the index it writes, so round-trip reads work.