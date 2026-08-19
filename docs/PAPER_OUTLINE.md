# Is There Depth Left to Fold? — paper outline (draft 2, 2026-08-19; adds B1 kernel stage, A6 frontier, Group C flagged section)

Working title: **"Is There Depth Left to Fold? Training-Free Structural
Compression of an End-to-End 1-Bit LLM"**

All numbers below are final measured results from docs/LAB_NOTEBOOK.md;
within-format only (folded vs unfolded 1-bit), per the pre-registered design.

## 1. Introduction
- Question: does extreme precision compression (1.125 bpw, post-training
  conversion) consume the structural depth redundancy that layer pruning
  exploits in FP models — or does it survive (robustness transfer)?
- Why it matters: the two compression axes multiply if orthogonal; the
  frontier device budget is bytes, and Bonsai 27B is the first public
  27B-class end-to-end binary model to test on.
- Contributions: (i) first redundancy map + drop/merge study of an
  end-to-end 1-bit LLM at block and sublayer granularity; (ii) a
  training-free, format-preserving operator suite whose surviving weights
  are byte-identical (exactness as a verification discipline, not an
  aspiration); (iii) a two-component damage law; (iv) a 4.2 GB shipped
  artifact at −15.4% model bytes with GSM8K at reference parity; (v) a catalog of
  measured negative results (width, entropy, merging, vocab both sides) that
  bounds what does NOT work at 1 bit.

## 2. Setup
- Model: Bonsai 27B 1-bit MLX pack (64 blocks, 48 linear + 16 full attn,
  g128 affine binary; packing law verified exhaustively incl. the 3
  f16-subnormal edge groups).
- Measurement ladder: teacher-forced full-vocab forward KL (on-policy =
  frozen self-generated MATH-500 thinking traces; off-policy = held-out
  wikitext), then 500-item mini-bench (GSM8K/MATH-500/IFEval/MMLU-Redux)
  with rule-first scoring; frozen seeds and probe fingerprints throughout.
- Exactness discipline: zero-copy candidate views; byte-identity
  verification for every written pack; bit-identical loader gate.

## 3. The redundancy map (H1)
- Block-level BI + KL screens: slack EXISTS — best single-block drop
  0.0033 nats on-policy (~3.7x the 4-bit-KV noise floor). All-but-one of
  the top-9 pass off-policy too; block 1 as the cautionary reshuffle case
  (0.0076 on / 0.113 off) → two-regime screening is mandatory.
- Sublayer granularity: attention halves of linear blocks dominate the
  droppable tail (10/12 top candidates), including LATE blocks whose whole
  blocks are expensive; worst sublayer single (0.0036) ≈ best block single.
- BI ranks imperfectly predict KL ranks at every granularity → cheap maps
  shortlist, KL decides.

## 4. Composition: the damage law
- Near-additivity at small k (tax 1.00 at k=2) growing to ~1.9x at k=8;
  the SAME tax curve in both probe regimes (regime-independence).
- The interaction tax is a same-type phenomenon (H5 reframed): cross-type
  sets compose at ~1.02-1.05; same-type at 1.14-1.19. Block type is NOT a
  clean droppability axis; type-matched controls split.
- Benchmark damage decomposes into (a) a ~2.4-pt truncation floor
  (thinking-length compensation: +10-30% gen tokens, truncations 26→33/100
  at 16K budget — an inference-budget artifact, buyable back) and (b) an
  accelerating knowledge term (MMLU monotone through all anchors). No
  single closed form in KL fits both; the practical accept region is
  ~0.02 nats on-policy per set.

## 5. What does NOT work at 1 bit (measured negatives)
- Width: scale-saliency spread is ERASED by the conversion (min/med 0.95
  across all 8,704 chunks) — group-aligned width pruning DOA.
- Entropy coding of signs: 8.000 bits/byte — the sign plane is
  information-optimal.
- Merging (H4): promotion > sign-election everywhere, and merging beats
  dropping BOTH blocks — but never beats dropping ONE (byte-matched loss
  1.5-2.5x across 4 pairs × 2 regimes). The shared-computation dividend is
  real but insufficient.
- Vocabulary trimming, BOTH sides (tri-part negative): input tail fat
  (held-out OOV 0.4-8.1%); mixed-domain emission tail fat (1e-4 dropped
  mass keeps 87% of rows; 1e-5 keeps 98%); and single-domain keep-sets
  fail task transfer end-to-end (+26% gen length, 2.3x truncations,
  despite bit-identical teacher-forced logits — generation-path fidelity
  is not implied by logit identity). The 248K vocab is load-bearing.

## 6. Free format wins
- B1 bias-plane redundancy: biases == f16(-scales/2) exhaustively → strip
  420 MB (8.2%) with BIT-IDENTICAL logits. The g128-affine binary format
  stores a derivable plane.
- B1 kernel stage (DONE 2026-08-17): mode "affine-derived" computes the
  bias in-register — 6 matmul routes + dequantize bit-identical to affine
  (synthetic parity + 27B identity gate); resident RAM −420,225,024 B
  measured (3.919 → 3.527 GB). The disk win becomes a RAM win with zero
  quality cost, no new module classes (stock layers + mode flip).
- (Head trimming initially screened at 165 MB on single-task emission
  mass; moved to §5 after the transfer failure — kept here as the
  methodological lesson about screening vs benching.)

## 7. The shipped artifact
- Bonsai-27B-1bit-folded-709: operator chain (bias-strip → drop blocks
  {16,12,13,9} → drop attn sublayers {37,38,58}), 4.8→4.2 GB on disk
  (−709 MB of model bytes, −15.4%), FULL-vocab logits bit-identical to
  the screened views — the artifact and the benched config are the same
  computation.
- Benched: GSM8K .910 (= reference), MATH .670, IFEval .880, MMLU .725,
  macro .796 (−4.5 pts); pack-level confirmation bench reproduced the row
  with 500/500 per-item agreement (bit-identity → trajectory identity).
- Pareto: strictly dominates k6 (2x bytes at −2 pts less damage); k4-class
  damage at nearly 3x k4's savings. Positioning: a folded 27B as a
  "smaller model" vs native ~4 GB-class models (published-numbers table).
- Efficiency honesty: decode tok/s gain ≈ bytes removed, but thinking-length
  compensation eats part of it on hard tasks; report wall-clock both ways.

## 7b. The searched frontier (A6)
- (1+λ) EA over per-block {keep, drop_block, drop_attn, drop_mlp}: 311
  genomes, 3 seeds, 2 byte tiers; champions confirmed at 100 probes both
  regimes. Search dominates or ties the hand frontier at every tier and
  the margin GROWS with aggressiveness (off-policy: 5% → 14% → 40%
  better) — it routes around the same-type interaction tax via operators
  no shortlist proposed (block 36, mlp-only drops incl. full-attn-block
  MLPs, attn-kept partial blocks). Data: Table 3 in experiments/paper_tables.

## 7c. FLAGGED arm — Group C scale quantization (VALUE-MODIFYING; separate reporting)
- Reported strictly apart from every byte-identical claim (authorized
  2026-08-12): per-row 8-bit second-level quantization of the f16 scales
  plane, riding the derived-bias kernels so biases follow s' consistently.
- Full ladder: KL 2.7e-06 on / 9.2e-06 off (100+100 probes) — ~1000× below
  the gentlest structural op; 500-item bench macro .8363 vs reference
  .8413 (−0.5 pts, under sampling σ), flips 13/18 bidirectional, gen
  tokens +1.0%, truncations 0 — no A4-style tail signature.
- Line: 4.71 → 4.19 GB via metadata-only ops (−420 MB derivable plane,
  −192.5 MB scale quantization), sign planes untouched. Stacking on
  folded-709's structural drops (~4.0 GB combined flagged artifact) is
  unbuilt/unbenched — future decision, outside authorized scope.

## 8. Limitations & open lines
- Single model/format (Bonsai 27B g128 binary); single machine; N=500
  bench (sublayer add-on cost not separable from k4 at this N — McNemar
  p=.27-1.0); 16K thinking budget confounds the floor; ternary arm cut by
  design decision; A6 champions KL-confirmed but not bench-confirmed;
  Group C stacked-with-structural-drops artifact unbuilt; qvm
  (transpose=false) route of the derived kernels is a documented loud-error
  gap (unused by inference).

## 9. Reproducibility
- Everything training-free on public weights, one 64 GB Mac; repo with
  frozen probe sets (fingerprinted), operator suite with exactness tests
  (418, incl. Group C interaction guards), patched-runtime build recipe +
  validation battery, config-level provenance chains (flagged_arm survives
  every operator), and the full append-only lab notebook as the record.

## Figures/tables shortlist
1. Redundancy map heat-strip (block × sublayer BI, both regimes).
2. Damage-composition curve: measured KL vs additive prediction, k=2..8,
   both regimes (the tax curve).
3. Anchor table + KL→macro scatter with the two-component fit.
4. H4 quad chart: merge vs drop-one vs drop-both per pair.
5. The negatives panel: width saliency flatness + sign-entropy + embed OOV.
6. Final artifact ledger waterfall: 4.8 GB → 4.2 GB by operator.
