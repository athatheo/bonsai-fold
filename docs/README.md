# bonsai-fold

Training-free depth compression of end-to-end low-bit LLMs. We map, drop, and merge transformer blocks of PrismML's 1-bit (primary) and ternary (comparison/fallback) Bonsai 27B using format-preserving operators, and measure whether extreme precision compression has consumed the structural redundancy that depth pruning exploits in full-precision models. Runs entirely on a 64 GB Apple Silicon machine with MLX. See CLAUDE.md and docs/RESEARCH_PLAN.md.
