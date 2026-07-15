# Related work and positioning

NOTE: compiled July 2026 from a quick pass. Re-run a fresh literature search before any external writeup and update this file.

## Depth pruning (FP models)
- ShortGPT: Block Influence via hidden-state in/out similarity; one-shot layer removal. Our Phase 1 metric.
- LLM-Streamline, SLEB, Shortened LLaMA: alternative importance signals (cosine similarity plus stability, adjacent-block perplexity, gradient/magnitude).
- LoRP (2026): locality-aware clustering of inter-layer similarity to allocate pruning.
- Prune&Comp (2025): training-free magnitude compensation after layer pruning. Possible trit-space analogue: compensate through the group scales of surviving layers. Small experiment candidate.
- Gromov et al., "The Unreasonable Ineffectiveness of the Deeper Layers": large fractions of layers removable with QLoRA healing. Precedent for our deferred Phase 4.

## Merging
- TIES-merging: sign election plus disjoint merge for task vectors. Our election operators are the depth-wise analogue, executed natively in sign/trit space.
- LayerFold (Thanasis's prior project): similarity-driven layer-merging framework with multi-strategy selection. Port pair-selection ideas; keep this repo standalone.

## Low-bit LLMs
- BitNet and BitNet b1.58: from-scratch 1-bit/ternary pretraining; source of the absmean ternary projection reused in our projection-merge variant.
- OneBit, FBI-LLM, BiLLM, STBLLM: binarization lines.
- Sparse-BitNet (2026): N:M sparsity inside ternary layers, trained from scratch. Adjacent and distinct: width-level sparsity, from-scratch training; ours is depth-level structural surgery on post-training-converted models.
- PrismML Bonsai family: post-training conversion of pretrained models to end-to-end binary/ternary; the method is proprietary, which forces our training-free, format-preserving design and makes it reproducible by anyone on the public weights.

## The gap we occupy (verify all three with a fresh search before publishing)
1. No published structural depth compression (dropping or merging) of end-to-end binary or ternary LLMs.
2. No depth-pruning study on hybrid-attention (linear plus full) stacks at any precision.
3. No format-native merge operators for sign/trit weights; the binary-pair-to-ternary promotion merge and the resulting heterogeneous-precision stack appear to be new.
