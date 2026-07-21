# b_math_anomaly — is k8 math500=.72 > k6=.64 real non-monotonicity?

Anchor chain (nested drops): ref ⊂ k2=drop[16,12] ⊂ k6=+[13,9,8,4] ⊂ k8=+[5,15]. Paired analysis on the same 100 frozen math500 items (seed fixed by data; script is deterministic).

## KL anchors (mean nats, read from kl_screen results)

| config | on-policy KL | off-policy KL |
|---|---|---|
| k2=drop[16,12] | 0.0076 | 0.0330 |
| k6=drop[16,12,13,9,8,4] | 0.0311 | 0.2046 |
| k8=drop[16,12,13,9,8,4,5,15] | 0.0643 | 0.3915 |

KL damage is strictly monotone k2 < k6 < k8; the math500 accuracy bump at k8 is therefore not explained by the distributional-damage metric.

## 1. Pairwise flip tables (math500, n=100)

| pair | both✓ | A-only✓ | B-only✓ | both✗ | discordant | McNemar exact p |
|---|---|---|---|---|---|---|
| ref vs k6 | 64 | 8 | 0 | 28 | 8 | 0.008 |
| ref vs k8 | 66 | 6 | 6 | 22 | 12 | 1.000 |
| k6 vs k8 | 61 | 3 | 11 | 25 | 14 | 0.057 |
| ref vs k2 | 64 | 8 | 2 | 26 | 10 | 0.109 |
| k2 vs k6 | 61 | 5 | 3 | 31 | 8 | 0.727 |
| k2 vs k8 | 63 | 3 | 9 | 25 | 12 | 0.146 |

gsm8k companion (n=100):

| pair | both✓ | A-only✓ | B-only✓ | both✗ | McNemar exact p |
|---|---|---|---|---|---|
| ref vs k6 | 86 | 5 | 2 | 7 | 0.453 |
| ref vs k8 | 84 | 7 | 3 | 6 | 0.344 |
| k6 vs k8 | 82 | 6 | 5 | 7 | 1.000 |

Per-item non-monotone patterns along ref→k2→k6→k8: 14/100 items show at least one recovery (wrong at an earlier anchor, right at a later one).

## 2. Wilson 95% CIs (math500)

| config | acc | Wilson 95% CI |
|---|---|---|
| ref | 0.72 | [0.625, 0.799] |
| k2 | 0.66 | [0.563, 0.745] |
| k6 | 0.64 | [0.542, 0.727] |
| k8 | 0.72 | [0.625, 0.799] |

k6 and k8 CIs overlap: True. All four CIs mutually overlap heavily; unpaired CIs are uninformative at n=100 for 8-point gaps — the paired tests above are the honest view.

## 3. Truncation × correctness (math500)

| config | stop✓ | stop✗ | length✓ | length✗ | truncated | mean gen_tokens | median |
|---|---|---|---|---|---|---|---|
| ref | 72 | 2 | 0 | 26 | 26 | 8223 | 5148 |
| k2 | 66 | 1 | 0 | 33 | 33 | 8759 | 5668 |
| k6 | 64 | 1 | 0 | 35 | 35 | 9026 | 6636 |
| k8 | 72 | 2 | 0 | 26 | 26 | 9337 | 7920 |

gsm8k truncation counts: {'ref': 6, 'k2': 6, 'k6': 6, 'k8': 8}

### Flip-set token comparison (k6 vs k8, math500)

- **k6 wrong → k8 right (k8 recoveries)** (n=11): k6 tokens mean 16384 / median 16384 (11 truncated); k8 tokens mean 11004 / median 12528 (0 truncated); reference got 5/11 of these right.
- **k6 right → k8 wrong (k8 losses)** (n=3): k6 tokens mean 4908 / median 4039 (0 truncated); k8 tokens mean 16384 / median 16384 (3 truncated); reference got 3/3 of these right.

Paired gen_tokens (k8 − k6) over all 100 math500 items: mean +311, median +0, sign test +41/−36 (p=0.649).
Restricted to items both configs solved (n=61): mean +946, median +884, sign test +38/−23 (p=0.072).

## 4. Difficulty stratification

minibench_items.json math500 entries carry only {id, task, prompt, gold}; the MATH-500 'level' field was not persisted by build_minibench.py and ids are ordered by (sorted) source index, not difficulty. Skipped.

## 5. Verdict

- Discordant k6/k8 pairs: 14 (11 k8-only correct vs 3 k6-only correct; net margin 8 items = 8.0%).
- P(k8 ≥ k6 by ≥ this margin | equal true accuracy) = 0.029 (one-sided exact binomial on discordant pairs); two-sided McNemar p = 0.057. Borderline: unlikely to be pure symmetric noise, but not decisive at n=100 with a single seed per config.
- **Mechanism is a budget interaction, not a capability inversion.** Truncation ⇒ wrong in every config (length_correct = 0 everywhere); 97.2% of k6's math500 errors are budget exhaustions. Every one of the 11 k8 recoveries was a k6 truncation at 16384 tokens that k8 finished naturally (mean ~11.0k tokens), and every one of the 3 k8 losses was a k8 truncation of an item k6 solved in 2.7k–8.0k tokens. All 8 ref→k6 losses were also k6 truncations.
- **Thinking-length compensation hypothesis: refuted in the stated direction.** k8 does not recover by thinking longer — it recovers by *avoiding runaway thinking* (k6: 35/100 truncated; k8: 26/100, same as reference). On items both solve, k8 is only mildly longer (median +884 tokens, sign test p≈0.07); over all 100 items the paired difference is null (median 0, p≈0.65).
- k8 ≠ ref behaviorally despite equal accuracy: 12 discordant items (6/6 split, McNemar p=1.0), and several k8 recoveries finished at 13.4k–15.3k tokens, barely under budget — accuracy parity is partly luck of the budget boundary.
- See JSON for full per-item flip lists and token details.

