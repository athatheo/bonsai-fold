# bonsai-fold — working rules

Training-free depth compression (block dropping, format-native block merging) of PrismML's Bonsai 27B. 1-bit is the ONLY arm — the ternary comparison arm (H2/H3) was cut by Thanasis on 2026-07-28; do not download or evaluate the ternary pack. (2-bit-format merged blocks *inside* the 1-bit model, via the Phase 3 promotion merge, are not the ternary arm and remain in scope.) See docs/RESEARCH_PLAN.md and docs/BONSAI_FACTS.md.

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

## Flagged arm (authorized 2026-08-12)
Group C (scale-metadata quantization) is authorized by Thanasis as a VALUE-MODIFYING experimental arm. It is exempt from the byte-identity constraint but must be (a) implemented as a separate operator, (b) screened on the full KL+bench ladder, and (c) reported separately from the byte-identical result line in all tables and the paper. The main artifact line remains byte-identical.

## Flagged arm R (authorized 2026-08-24)
Group R (activation-informed scale repair) is authorized by Thanasis as a second VALUE-MODIFYING experimental arm ("do more weight modification if high impact"). Scope: closed-form per-channel gain repair of FOLDED models, applied ONLY to the f16 scales plane of designated surviving modules — sign planes, packed weights, and all other tensors stay byte-identical. Gains are fitted by least squares on the CALIBRATION set exclusively (never on probe sets — leakage); no iterative training, no gradient descent. Same rules as Group C: (a) separate operator, (b) full KL+bench ladder with THE BENCH as the decision gate (screens are demonstrably Goodhartable — A6), (c) separate reporting from every byte-identical line. The main artifact line remains byte-identical.
