# bonsai-fold

Training-free structural compression of [PrismML's Bonsai 27B](https://huggingface.co/prism-ml/Bonsai-27B-mlx-1bit), the openly released 1-bit (1.125 bits/weight) 27B model, measured end-to-end on a single 64 GB Apple Silicon laptop. This project is a direct continuation of PrismML's work: they showed a 27B model can survive binarization ([whitepaper](https://github.com/PrismML-Eng/Bonsai-demo/blob/main/bonsai-27b-whitepaper.pdf), [model collection](https://huggingface.co/prism-ml)); we ask what structural redundancy remains inside their artifact afterwards, and answer it under strict rules: no training, no fine-tuning, no calibration-based updates to surviving weights, and every claim gated by a 500-item, four-task evaluation run. All credit for the base models and the 1-bit conversion belongs to PrismML; everything here operates on their public checkpoints.

Write-up: [`docs/paper/paper.pdf`](docs/paper/paper.pdf). Full provenance: [`docs/LAB_NOTEBOOK.md`](docs/LAB_NOTEBOOK.md) (append-only) and [`docs/RESULTS.md`](docs/RESULTS.md) (canonical numbers). Generated result tables: [`experiments/paper_tables/tables.md`](experiments/paper_tables/tables.md).

## Headline results

| configuration | size | macro (500 items) | notes |
|---|---|---|---|
| baseline (unmodified) | 4.71 GB | .8413 | reference under this harness |
| **depth-pruned** (4 blocks + 3 attn sublayers + zero-point removal) | 4.2 GB (−15.4%) | .7963 | surviving weights byte-for-byte unchanged; GSM8K stays at .910 |
| pruned + error compensation | 4.2 GB | .8063 | closed-form update, 1% of sign bits changed, no added parameters |
| pruned + 8-bit scales | **3.82 GB (−18.9%)** | .7900 | second-order quantization of scale parameters |

Other findings: the MLX packing's stored zero-points satisfy z = f16(−s/2) everywhere and can be recomputed in-register (−420 MB resident memory, logits unchanged); degradation composes near-additively across removed components; six controlled cases where KL-based proxy metrics mispredict end-task outcomes, including an evolutionary search that dominates the proxy frontier and loses the benchmark; and a replication on a dense 8B sibling showing the structural laws generalize while the removable-redundancy budget does not.

## Layout

- `src/bonsaifold/` — loader (layer-types-aware), pack surgery operators (`fold.py`, `merge.py`, `scalequant.py`), zero-copy pruning views, head-group removal, elastic-depth serving, error-compensation fitting.
- `experiments/` — every experiment with its committed result JSONs: KL screens, the mini-bench harness and all runs, the evolutionary search ledgers, the head map, the 8B replication, repair fits and screens.
- `patches/` — the MLX fork patch adding the `affine-derived` kernel mode (in-register zero-point recomputation).
- `docs/` — research plan, lab notebook, canonical results, paper source and PDF.
- `tests/` — 430 tests covering the operators, format invariants, and guards.

## Reproducing

Model packs are not committed (multi-GB); the base checkpoints are PrismML's public releases on Hugging Face ([`prism-ml/Bonsai-27B-mlx-1bit`](https://huggingface.co/prism-ml/Bonsai-27B-mlx-1bit), [`prism-ml/Bonsai-8B-mlx-1bit`](https://huggingface.co/prism-ml/Bonsai-8B-mlx-1bit)). Setup:

```
uv sync                      # builds the patched MLX (see scripts/setup_env.sh)
uv run python -m pytest -q   # 430 tests
```

Pack operators rebuild every artifact from the public checkpoint (see `docs/RESULTS.md` for the operator chain of each artifact). Benchmarks: `experiments/minibench/run_minibench.py`. Long runs checkpoint per item and resume by id.

Note: 1-bit (`bits=1`) support requires the patched MLX build in this repo; stock MLX rejects it.
