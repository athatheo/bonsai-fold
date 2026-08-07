# bonsai-fold experiment shortlist (ranked)

Baseline reality check used throughout: the pack stores u32 sign bits + f16 scale AND f16 bias per g128 group (census: separate scales/biases tensors), i.e. ~1.25 bpw stored; embed + lm_head are untied (2 x 248320 x 5120, ~397 MB); 48/64 blocks are linear-attention; depth axis (whole-block drop + H4 merges) is already built. Savings percentages below are against the ~4.2 GB pack.

---

## Group A — do now: delete-only, no constraint changes, no kernel work

**A1. Sublayer-granular dropping (drop-attn / drop-mlp operators)** — BlockPruner / FinerCut / data-free Gate-Norm family, merged with MultiPruner stage-1.
- Mechanism: treat each block's attention sublayer and MLP sublayer as independent deletion candidates (residual stream makes removal a pure module skip). Selection: black-box calibration-KL (BlockPruner/FinerCut style — transfers to both linear-attn and full-attn sublayers) with the existing KL harness as inner metric; Gate-Norm's data-free criterion only on the 16 softmax-attn blocks.
- Savings: attn sublayer ~16 MB, MLP ~38 MB vs ~54 MB whole block; plausibly 300–600 MB beyond the whole-block budget at equal damage (FinerCut: 42% of attn sublayers removable at ~99% perf on FP 70B — re-measure within-format).
- Cost: model-class + per-layer KV-cache-list patch, same tier as the existing drop_blocks/drop_view machinery; zero kernels. Sidesteps the positional layer_types hazard since layer count can be preserved.
- Novelty: first sublayer-drop results on a hybrid linear/full-attention **1-bit** model; direct extension of the paper's depth thesis. Highest priority overall.
- Gate: expect linear-attn sublayers to be the viable candidates (full-attn carries all global mixing, only 16 exist).

**A2. Calibration-configuration sweep (methodology control, arXiv 2604.24938)**
- Mechanism: before trusting any ranking (A1, A5, width arms), sweep calibration domain/length/amount and measure drop-set stability; paper shows calibration config dominates search algorithm choice, and calibration PPL correlates poorly with downstream accuracy.
- Savings: none directly; protects every other experiment and is a cheap, citable rigor section. Hours of forward passes, well under the 2 h cap per chunk.

**A3. Cheap census jobs that gate Groups B/C (run this week, minutes-to-hours, zero risk)**
- (i) **Bias-redundancy check**: exact-equality test whether biases ≡ −scale (or −scale/2) for every g128 group across the pack. If true, the entire bias sideband (~0.125 bpw, ~10% of pack, ~420 MB) is derivable in-kernel → unlocks B1 with zero value modification.
- (ii) **Entropy census**: per-tensor symbol histograms of u32 sign words and f16 scale/bias exponents → bounds ANS/entropy-coding payoff (B3) before any Metal work.

**A4. Output-side-only lm_head row trimming (FR-Spec/VocabTrim recipe, applied to the target head)**
- Mechanism: delete lm_head rows outside a frequency-ranked subset; tokenizer and input embeddings untouched; sampler-side int32 logit-index→token-id map. Rows are g128-clean (40 groups along hidden per row) — surviving rows byte-identical.
- Savings: ~133 MB (keep 64k) to ~156 MB (keep 32k) of the 179 MB head, plus 74–87% of lm_head GEMV per token.
- Cost: trivial; no kernels. Flags: lossy renormalization over surviving rows (neither cited paper validates target-side trimming — that itself is a small novel result); force-include thinking/chat/tool/byte/digit specials; forward-KL gate is degenerate on dropped mass — use subset-restricted/renormalized KL + dropped-mass coverage stats.

**A5. Full static vocabulary trim (embed + lm_head + tokenizer), after A4 validates**
- Mechanism: frequency-rank the 248320-token multilingual vocab; keep top-k + specials + all 256 byte tokens; delete rows from both untied matrices with one keep-list; prune BPE merges with Qwen-Tokenizer-Pruner "lossless" discipline (merge closure, equivalence-checked). Keep trimmed vocab a multiple of 128.
- Savings: ~199 MB at 50% trim up to ~311 MB at 32k rows (~5–7%).
- Cost: pure Python + config; the real work is the merge-closure discipline and the id-remap table for off-policy KL (commit it with the pack). Gates on target-language corpora only.

**A6. EvoPress-style evolutionary search over per-span profiles {keep, drop-block, drop-attn, drop-mlp, sign-election merge, promotion merge}**
- Mechanism: (1+λ)-EA with level-switch mutation, TopK-KL fitness (K≥512 — full logit caching at vocab 248320 is ~1 GB per 2K-token sequence), routed through the logit-verified layer_types loader shim. Key justification: per-layer error is non-additive, so greedy ranking is provably suboptimal — this directly upgrades the existing H4 results.
- Savings: no new bytes, but more deletions at the same damage threshold; produces the paper's Pareto frontier.
- Cost: small driver reimplementation atop run_kl_screen (repo is PyTorch/CUDA, nothing portable); budgets must be cut (λ≈16–32, 4–8K-token stage-1 sequences, ~15–35 s/candidate) or chunked under launchd with prior approval. Import only GeLaCo's Pareto framing — its collapse operator is value-modifying, and EvoPress's SparseGPT/GPTQ level databases are out of scope.

---

## Group B — worth it, needs kernel/format work

**B1. Bias-sideband elimination (derive bias from scale in-kernel)** — contingent on A3(i) confirming exact redundancy.
- Mechanism: if bias ≡ f(scale) exactly, drop the biases tensors from the pack and compute the bias inline in the fork's qmv/qmm/qvm/gather + quantized-embedding kernels.
- Savings: ~0.125 bpw → ~10% of the pack (~420 MB), **exact and fully value-preserving** — the single largest byte-identical win on the table, bigger than any constraint-crossing metadata scheme in Group C.
- Cost: touches every quantized kernel variant in the patched fork, but is a one-line decode change per kernel; no new format machinery.
- Novelty: a clean format-level finding ("the 1-bit affine bias plane is redundant") that halves the metadata story for the whole paper. If A3(i) fails (bias not exactly derivable), this dies and Group C matters more.

**B2. Width arm: 128-bundle MLP pruning (MultiPruner stage-2 mechanics, FLAP-fluctuation or Wanda-sp scoring as ablation baselines)**
- Mechanism: MLP-only (linear-attn head pruning is unvalidated new work; softmax heads cover only 16 blocks), input-dim deletion in contiguous 128-channel bundles (17408 → 136 bundles/layer) with group-aggregated scores; output-row deletion at any granularity. Bias-free scoring only (FLAP's bias compensation → Group C). Compute saliency on dequantized effective weights.
- Savings: unknown on an already-1-bit substrate — that is the experiment; FP literature suggests 10–20% of block params (~0.35–0.7 GB) before steep damage, but Wanda-sp's magnitude term is degenerate within a 1-bit group (ranking collapses to activation norms) and FP numbers do not transfer.
- Cost: packed-u32 bit-surgery in pack tooling + per-layer heterogeneous-shape plumbing in the loader/model classes (no Metal kernels if bundle-aligned). Calibration budgets must be cut ~10–30x vs published defaults (FLAP's "3–5 min" is ~1.5–2.5 h here).
- Novelty: first structured width results inside a 1-bit format; orthogonal axis for the paper. Adopt MultiPruner's global-budget greedy loop (reimplemented, ~600 lines) and its GQA-integral head constraint if the 16 softmax blocks are ever included.

**B3. ANS entropy coding of the scale/bias sideband (tile-ANS paper, arXiv 2606.15789)** — contingent on A3(ii).
- Mechanism: rANS decode fused into the 1-bit dequant path (Metal restructuring as barrier-phased decode/compute; no public code — build from paper + DietGPU). Signs are near-incompressible if balanced; the payoff lives in the f16 metadata.
- Savings: realistic 3–8% (less if B1 already deleted the bias plane — these overlap); more only if the census finds sign-plane skew.
- Cost: moderate-to-high; batch-1 decode is the paper's own worst-case regime (decoder-bound, TPOT regressions in their Table II). Rank last in B; only proceed if the census surprises.

---

## Group C — constraint-crossing (modifies stored scale/bias values) — needs Thanasis's sign-off before any GPU time

All of these leave sign bits byte-identical but re-round the f16 scales (± biases), perturbing every dequantized weight in each group — a direct "never alter surviving weight values" violation. In a 1-bit format the scales carry all magnitude information, so published near-losslessness (4-bit substrates) does not transfer; each must be gated within-format like a fold. Note the interaction: if B1 lands, biases vanish for free and all savings below halve to scale-only. Recommend proposing only C1 as the flagged arm, with C2/C3 as ablations.

**C1. GGUF k-quant-style superblock scales** — 6-bit sub-scales + one f16 super-scale per 8 groups. ~211 MB (5.6%), ~0.8% max scale error, best accuracy-per-saved-byte, strongest Apple Silicon precedent (llama.cpp Metal k-quants) — but it is a new custom layout, precedent-guided rewrite of the fork's decode path, not a port. Data-free rounding only; no imatrix.

**C2. FP8-E4M3 scales (NVFP4-style, + f32 per-tensor pre-scale)** — ~211 MB (5.6%); kernel-trivially close to MLX's shipping fp_quantized.h dequantize_scale, but coarser (up to 6.25% relative scale error). Simplest to implement, worst error of the three.

**C3. SpQR bilevel int3 stats** — biggest crush (~290 MB, 7.6%, up to ~13% with biases) and biggest distortion (8 levels per 16-scale window); fiddliest kernel. Only if C1 proves 1-bit scales are robust. (QLoRA-style DQ is dominated by C1 — same decode work, worse ratio/accuracy trade — fold it into the ablation table, don't run standalone.)

**C4. FLAP bias compensation** (calibration-derived new bias tensors) — only if B2's bias-free scoring visibly underperforms; it is the entire published gap between FLAP and Wanda-sp, but it is calibration-fitted values the model never had.

---

## Group D — rejected

- **DFloat11**: premise is BF16 exponent redundancy; our mass is near-max-entropy sign bits. Ceiling ~2–4%, CUDA-only, runtime decompress-to-dense (constraint 2), 2x slower at batch 1.
- **EntroLLM**: decode is a once-per-sequence preprocessing stage — DRAM holds the decoded model (fails constraint 2); no code release; Huffman over a ~balanced binary source saves ~nothing; their best endpoint (1.39–1.62 bpw) is *above* our starting point.
- **STBLLM**: pipeline inherits BiLLM/GPTQ error compensation = calibration weight optimization (hard constraint 1 violation); kernel unreleased (patent); honest N:M mask accounting is *worse* than dense 1.125 bpw packing until ≥75% sign sparsity; needs FP magnitudes we don't have. The stripped in-house N:M idea inherits the same mask arithmetic — do not pursue.
- **Post-hoc weight tying**: measured on the actual pack — embed vs lm_head cosine 0.0059, sign agreement 50.3% (chance). Installing either matrix in the other's role is a near-random projection. Keep as a one-line negative-result table row (one cheap KL probe max); also would force a shared keep-set that conflicts with A4/A5.
- **DIET hidden-dim pruning**: ~8-point accuracy loss per 10% memory in FP, zero quantized-model evidence, alpha rescale is constraint-crossing, and shrinking hidden breaks the H4 merge closed forms and every g128 alignment downstream. Bad trade on every axis.
- **mx.block_masked_mm / mlx-sparse / any unstructured or N:M sparsity on this stack**: upstream MLX formally won't-fix sparse matmul (#2728 closed 2026-05-05; #2796 rejected); block_masked_mm is dense-float-only with zero memory savings (dequantizing 1.125 bpw → f16 is a ~14x blow-up); mlx-sparse stores f16 values at ≥48 bits/nonzero vs our 1.125 bpw — break-even needs <2.3% density. Nothing composes with the packed 1-bit format. Report as a one-paragraph infrastructure finding, not an experiment. (gather_qmm is verified 1-bit-native but is matrix-granularity MoE selection — useful machinery note, not compression.)
- **VocabTailor dynamic vocab restriction**: 0 bytes saved on disk; decode-compute ceiling ~4–5% on a 27B; task-conditional and risky for long CoT with thinking mode on. At most an appendix latency add-on via the verified take+qmm path — not a compression experiment.
- **Wanda-sp as a standalone arm**: not rejected but subsumed — its metric is degenerate within 1-bit groups; it survives only as the baseline scoring criterion inside B2's ablation table.

---

## Suggested execution order

1. A3 census jobs (gates B1/B3, hours) + A2 calibration sweep — this week, trivially safe.
2. A1 sublayer dropping (the headline depth extension), then A6 EvoPress search over the enlarged operator set.
3. A4 lm_head trim → A5 full vocab trim (independent axis, composes additively).
4. B1 bias elimination if the census confirms redundancy (largest value-preserving byte win, ~10%).
5. B2 width arm.
6. Take the C1 proposal (single flagged metadata arm) to Thanasis with the census scale-histograms in hand; B3 only if the entropy census surprises.

Composability note for the paper: A1 + A5 + B1 stack additively (deleted sublayers also delete their metadata), plausibly ~0.9–1.3 GB total off the pack with every surviving byte identical — that stacked table, with the mandatory unfolded Bonsai-27B-1bit baseline row, is the strongest framing available. All folded variants keep speculative decoding disabled.