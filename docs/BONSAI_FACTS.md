# Bonsai 27B: distilled facts

Source: PrismML whitepaper (July 2026) and release post. Section and table numbers refer to the whitepaper PDF.

## Links
- Announcement: https://prismml.com/news/bonsai-27b
- Whitepaper: https://github.com/PrismML-Eng/Bonsai-demo/blob/main/bonsai-27b-whitepaper.pdf
- HF collection (27B packs): https://huggingface.co/collections/prism-ml/bonsai-27b
- Smaller family (GGUF, no thinking mode): prism-ml/Ternary-Bonsai-8B-gguf, -4B-gguf, -1.7B-gguf
- Code: https://github.com/PrismML-Eng/Bonsai-demo and the custom fork PrismML-Eng/llama.cpp (contains tools/kv-mean-center)
- Docs: https://docs.prismml.com/  |  License: Apache 2.0
- Free limited-time preview API (via Together, per the release page): sanity-checking unmodified baselines only.

## Architecture (Section 4, Table 2)
- Base: Qwen3.6-27B, architecture unchanged. 64 language blocks (~24.8B params) + embeddings/LM head (~2.5B) + vision tower (0.46B, 27 blocks, separate).
- Hybrid attention: ~48 linear-attention blocks (fixed-size recurrent state) and ~16 full-attention blocks (growing KV cache). Exact interleave indices unknown: resolve in census.
- SwiGLU MLP, RoPE, RMSNorm. Context 262K. Conversion method is proprietary (post-training, from the off-the-shelf pretrained model); no recipe details published, which is why all our operators must be training-free and format-preserving.

## Weight formats (Sections 3, 4.1, 4.2)
- End-to-end low-bit across embeddings, attention projections, MLP projections, LM head. No mixed-precision escape hatches; only a small normalization/scale tail stays higher precision. Vision tower is separate at 4-bit HQQ.
- Ternary g128: {-1,0,+1}, one FP16 scale per 128 weights, effective 1.71 bpw (log2(3) + 16/128).
- Binary g128: {-1,+1}, same scaling, effective 1.125 bpw (1 + 16/128).

## Sizes (Section 4.3, Table 7)
- Ideal: binary 3.9 GB, ternary 5.9 GB. FP16 reference: 54 GB.
- Deployed language-only: GGUF Q1_0 3.79 GB; GGUF Q2_0 7.15 GB (ternary currently stored in 2-bit slots); MLX 1-bit 4.21 GB; MLX 2-bit 7.57 GB.
- Published MLX safetensors packs: 5.13 GB (1-bit) and 8.49 GB (ternary), each bundling the vision tower.
- MLX affine stores scale plus bias per group. 1-bit mapping (Section 4.3): w = s_mlx*q + b_mlx with q in {0,1}, s_mlx = 2*s_g, b_mlx = -s_g, reproducing +/- s_g exactly. Ternary 2-bit mapping: verify empirically.

## KV cache (Section 4.4)
- Only the 16 full-attention layers cache. FP16 cache = 64 KiB/token total, so ~4 KiB/token per full-attention layer. A 4-bit KV quantizer ships (~4x cut).
- Tolerance anchor (Table 6): 4-bit KV induces 0.0009 nats on-policy / 0.0023 off-policy forward KL on the 1-bit model, vs 0.0137 / 0.222 for FP16 weights. Use as the noise floor when setting our KL gates.

## Throughput and peak memory (Tables 7, 8; decode tg128 / prefill pp512, tok/s)
- M5 Max: binary 66.4/874, ternary 44.0/830. M5 Pro: 44.2/421, 26.2/393. M4 Pro: 26.0/133, 18.0/125. iPhone 17 Pro Max: binary 11.0/111.
- Peak memory (MLX, language only): 1-bit ~5.9 GB at 4K ctx, ~6.3 GB at 10K; ternary ~9.2 / 9.6 GB. Trivial on a 64 GB machine; the unfolded reference and a folded candidate coexist comfortably.

## Published benchmark reference values (Tables 10-11, thinking mode; context only, no FP arm in our experiments)
| Category | FP16 | Ternary | 1-bit |
|---|---|---|---|
| Overall (15 benchmarks) | 85.07 | 80.49 | 76.11 |
| Math | 95.33 | 93.40 | 91.66 |
| Coding | 88.74 | 85.96 | 81.88 |
| Knowledge and reasoning | 83.15 | 76.96 | 73.39 |
| Instruction following | 78.47 | 71.77 | 65.74 |
| Agentic and tool calling | 80.00 | 74.01 | 66.03 |
| Vision | 72.61 | 65.19 | 59.57 |

## Eval methodology to mirror (Appendix B)
- Thinking mode throughout; a parser splits the reasoning block from the final answer; score answers only.
- Sampling for Bonsai models: temperature 0.7, top-p 0.95, top-k 20.
- Token budget tiers: short 16384, medium 20480, long 30000, extended 81920.
- Rule-first scoring. AIME uses 8 samples per item in the paper; excluded from our mini-bench for cost.

## Caveats
- The DSpark speculative drafter taps five evenly spaced target layers; folding breaks it. Disable speculative decoding on every folded model.
- The smaller Bonsai releases predate thinking mode and reliable tool use; the 27B is the only valid vehicle for reasoning/agentic retention claims.
- Deployed ternary uses 2-bit slots; quote size wins against deployed footprints, never ideal bpw.

## Day-1 unknowns (resolve in Phase 0)
1. Exact HF repo ids inside the collection for the 27B MLX and GGUF packs.
2. Does stock mlx-lm run the Qwen3.6 hybrid arch, or do we need PrismML's MLX code (their GitHub org, docs.prismml.com, the "easymlx" runtime named in the paper)?
3. Actual full-attention layer indices and interleave pattern from config.json.
4. Ternary 2-bit MLX affine mapping (codes to {-1,0,+1}).
5. Linear-attention block parametrization (any conv/state/gating tensors and shapes).
6. Whether vision-tower tensors can be stripped from the safetensors for text-only runs.
7. Chat template and thinking-tag conventions of the shipped tokenizer.
