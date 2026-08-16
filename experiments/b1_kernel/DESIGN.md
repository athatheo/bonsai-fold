# B1 kernel stage: mode="affine-derived" (in-register bias, 1-bit only)

Goal: eliminate the 420 MB biases plane from RAM (already gone from disk).
Constraint: bit-identical logits vs the load-time-derived path.

Approach — a new quantization mode mirroring the fork's existing 3-input
(mxfp4) plumbing, not a per-call-site fork of the affine kernels:

1. `QuantizationMode::AffineDerived` ("affine-derived"), valid only for
   bits==1, g128: same params as Affine, NO biases input.
   - ops.cpp: `validate_mode_with_type` accepts biases==nullopt for this
     mode; `quantized_matmul`/`affine_dequantize`(+gather variants used by
     QuantizedEmbedding) build 3-input primitives.
2. Metal kernels (quantized.h): add `template <bool DERIVE_BIAS>` to the
   OUTER kernel bodies at the ~sites where the `biases` device buffer is
   indexed (buffer reads, not the 22 bias-value arithmetic uses):
   `U bias = DERIVE_BIAS ? U(-0.5f) * scale : biases[g];`
   f16 exactness: -0.5*s in f16 == stored f16(-f32(s)/2) for all normals;
   subnormals reproduce the same last-bit rounding (validated empirically
   in the loader-derivation stage; re-validate at kernel stage).
3. Dispatch (quantized.cpp): mode suffix already flows into kernel names;
   AffineDerived binds {w, scales, x, ...} (no biases buffer), mirroring
   the mxfp4 conditional that already exists at lines ~221-224.
4. Python: mode="affine-derived" through the existing mode kwarg.
5. bonsaifold.loader: `DerivedBiasQuantizedLinear` (drop-in; no biases
   parameter; calls quantized_matmul(mode="affine-derived")) — installed
   for packs with bonsai_bias_plane=="derived" INSTEAD of the sanitize
   materialization; QuantizedEmbedding equivalent for embed/lm_head.

Gates (in order): fork test_quantized.py extended with derived-mode
parity tests (synthetic, GPU+CPU) -> 27B logit bit-identity vs the
materialized path on the identity sequences -> RAM measurement
(expect ~-420 MB resident) -> full validation battery -> smoke test.

Scope cut: qvm / non-inference gather variants deferred unless the model
path exercises them (verify via kernel-name logging during a smoke run).
