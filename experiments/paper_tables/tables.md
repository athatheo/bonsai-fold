# bonsai-fold result tables (generated; do not edit by hand)

## Table 1 — mini-bench (500 items), byte-identical arm
All surviving weights byte-identical to the original pack. Sizes vs the 4.71 GB nobias-inclusive baseline.

| config | MB removed | GSM8K | MATH500 | IFEval | MMLU-R | macro | n |
|---|---|---|---|---|---|---|---|
| Bonsai-27B-1bit (unfolded reference) | 0 | 0.910 | 0.720 | 0.930 | 0.805 | 0.8413 | 500 |
| k2: drop blocks {16,12} | 118 | 0.920 | 0.660 | 0.880 | 0.790 | 0.8125 | 500 |
| k4: drop blocks {16,12,13,9} | 236 | 0.910 | 0.680 | 0.920 | 0.760 | 0.8175 | 500 |
| k6: drop 6 blocks | 360 | 0.880 | 0.640 | 0.860 | 0.725 | 0.7762 | 500 |
| k8: drop 8 blocks | 478 | 0.870 | 0.720 | 0.790 | 0.620 | 0.7500 | 500 |
| folded-709 (k4 + attn{37,38,58}); SHIPPING | 709 | 0.910 | 0.670 | 0.880 | 0.725 | 0.7963 | 500 |
| A6 T350 search champion (KL-tier winner) | 365 | 0.880 | 0.610 | 0.830 | 0.730 | 0.7625 | 500 |

## Table 2 — FLAGGED arm: Group C scale quantization (value-modifying)
Reported separately per the flagged-arm rule: scales are 8-bit reconstructions; weights are NOT byte-identical. Authorized 2026-08-12.

| config | MB removed | GSM8K | MATH500 | IFEval | MMLU-R | macro | n |
|---|---|---|---|---|---|---|---|
| Group C: nobias + 8-bit scale plane (VALUE-MODIFYING) | 612 | 0.930 | 0.730 | 0.910 | 0.775 | 0.8363 | 500 |
| COMBINED: folded-709 + 8-bit scales (VALUE-MODIFYING) | 888 | 0.870 | 0.690 | 0.860 | 0.740 | 0.7900 | 500 |
| knee probe: nobias + 4-bit scales (DEGRADED) | 717 | 0.910 | 0.710 | 0.860 | 0.770 | 0.8125 | 500 |

## Table 2b — Generality: dense Bonsai-8B (byte-identical drops)
Same ladder, non-thinking scoring (the 8B family predates thinking mode). The structural laws replicate; the slack magnitude does not.

| config | MB removed | GSM8K | MATH500 | IFEval | MMLU-R | macro | n |
|---|---|---|---|---|---|---|---|
| Bonsai-8B-1bit (unfolded reference) | 0 | 0.800 | 0.630 | 0.830 | 0.605 | 0.7163 | 500 |
| 8B k2-analog: drop {26,17} | 33 | 0.750 | 0.530 | 0.820 | 0.595 | 0.6737 | 500 |
| 8B k4-analog: drop {26,17,21,31} | 66 | 0.630 | 0.300 | 0.760 | 0.585 | 0.5687 | 500 |

## Table 3 — KL Pareto frontier, hand-built vs A6 search (100 probes, both regimes)

| frontier | config | structural MB | KL on-policy | KL off-policy |
|---|---|---|---|---|
| hand | folded-709 config | 289 | 0.0260 | 0.160 |
| hand | k6 | 360 | 0.0310 | 0.205 |
| hand | k8_mixed | 478 | 0.0640 | 0.392 |
| search | A6 T350 champion | 364.6 | 0.0314 | 0.176 |
| search | A6 T350-s2 champion | 374.8 | 0.0322 | 0.187 |
| search | A6 T450 champion | 456.8 | 0.0504 | 0.233 |

## Memory/disk engineering results (B1, exact)

- B1 in-kernel bias derivation: resident RAM −420,225,024 B on the 27B (4,207,606,792 → 3,787,381,768), logits bit-identical (6-probe identity gate); disk −420 MB via the nobias pack.
- Group C disk: −192.5 MB further (498 scale tensors, per-row 8-bit).
