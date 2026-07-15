# bonsai-fold — working rules

Training-free depth compression (block dropping, format-native block merging) of PrismML's Bonsai 27B. 1-bit is the primary arm; ternary is the comparison arm and fallback. See docs/RESEARCH_PLAN.md and docs/BONSAI_FACTS.md.

## Hard constraints
- **Never alter surviving weight values.** Operators may delete blocks or produce merged blocks via the defined closed-form rules, but weights of surviving (unmerged) blocks are byte-identical to the originals. No training, no fine-tuning, no calibration-based weight updates of any kind.
- **Everything runs on this one 64 GB Apple Silicon machine.** No cloud compute for experiments.
- **No jobs over 2 hours without asking first.**
- No FP baselines; published whitepaper numbers are reference context only. All damage metrics are within-format (folded vs unfolded, same format).
- Disable speculative decoding on every folded model (the DSpark drafter taps fixed layer indices).

## Conventions
- Environment is uv-managed (`uv run ...`), Python pinned via .python-version.
- Model packs live under `models/` (gitignored). Experiment outputs under `experiments/`, committed.
- docs/LAB_NOTEBOOK.md is append-only, newest at the bottom; every entry has date, phase, what was done, numbers, surprises, decisions, open questions.
- Sampling for all Bonsai generation: temperature 0.7, top-p 0.95, top-k 20, thinking mode on.
