# a_mmlu_tail — why did k8's MMLU fall to .620?

k8 = drop[16,12,13,9,8,4,5,15]; paired vs reference on the frozen 200-item MMLU slice (alphabetical-by-subject order). Seed=0, 20000 permutations, exact stdlib tests (no scipy).

## MMLU accuracy (Wilson 95% CI)

| config | acc | 95% CI | on-policy KL (nats) |
|---|---|---|---|
| ref | 0.805 | [0.745, 0.854] | 0.0000 |
| k2 | 0.790 | [0.728, 0.841] | 0.0076 |
| k6 | 0.725 | [0.659, 0.782] | 0.0311 |
| k8 | 0.620 | [0.551, 0.684] | 0.0643 |

## McNemar exact, k8 vs ref (MMLU, N=200)

- both right 111, both wrong 26, ref-right/k8-wrong b=50, ref-wrong/k8-right c=13 (net 37 flips, delta=-0.185)
- exact two-sided p = 3.02e-06 — the drop is not chance.
- 31.1% of items ref got right are lost by k8.

## Subject groups (paired; official MMLU categories)

| group | n | ref | k2 | k6 | k8 | Δ(k8−ref) | McNemar p | power |
|---|---|---|---|---|---|---|---|---|
| STEM | 56 | 0.929 | 0.839 | 0.821 | 0.786 | -0.143 | 0.039 | LOW |
| humanities | 49 | 0.857 | 0.837 | 0.755 | 0.531 | -0.327 | 0.000 | LOW |
| social_sciences | 44 | 0.727 | 0.773 | 0.682 | 0.614 | -0.114 | 0.302 | LOW |
| other | 51 | 0.686 | 0.706 | 0.627 | 0.529 | -0.157 | 0.115 | LOW |

All 57 per-subject cells have n<=9 — individually uninterpretable; see JSON `per_subject` for the full table. Worst subjects by paired delta (n in parens): high_school_statistics -1.00 (n=1), medical_genetics -1.00 (n=1), miscellaneous -1.00 (n=3), prehistory -1.00 (n=3), professional_law -1.00 (n=1), professional_medicine -0.75 (n=4).

## Positional head (0–99) vs tail (100–199), paired

| config | head Δ | tail Δ | DiD (tail−head) |
|---|---|---|---|
| k2 | -0.020 | -0.010 | +0.010 |
| k6 | -0.080 | -0.080 | +0.000 |
| k8 | -0.150 | -0.220 | -0.070 |

k8 DiD permutation p (two-sided, 20000 shuffles) = 0.421.
Head/tail composition differs by construction: {'head': {'STEM': 51, 'other': 19, 'social_sciences': 21, 'humanities': 9}, 'tail': {'humanities': 40, 'other': 32, 'social_sciences': 23, 'STEM': 5}}.

## Truncation / anomaly screen (MMLU)

| config | stop | length | len head/tail | acc on length items | gen_tokens mean/med/p90 | zero-token |
|---|---|---|---|---|---|---|
| ref | 192 | 8 | 5/3 | 0.0 | 2754.7/1361/6525 | 0 |
| k2 | 191 | 9 | 6/3 | 0.0 | 2711.3/1203/6222 | 0 |
| k6 | 187 | 13 | 8/5 | 0.0 | 3018.1/1320/9364 | 0 |
| k8 | 188 | 12 | 4/8 | 0.0 | 3023.4/1469/7921 | 0 |

k8-only errors: 50; of those, 6 hit the 16384 budget in k8; 30 are in the tail half.

## Breakpoint scan (w=20 sliding paired delta, k8−ref)

- Worst window: items 141–160 (paired Δ=-0.450); subjects: management, marketing, medical_genetics, miscellaneous, moral_disputes, moral_scenarios, nutrition.
- Best two-segment split at t=143 (…management | marketing…): Δ=-0.133 before vs -0.316 after (gap 0.183, max-gap permutation p=0.737).
- Split sits 0 items from a subject boundary; 28% of all candidate indices are at least that close to one of the 57 boundaries, so alignment is uninformative.
- Paired delta by index quartile (k2/k6/k8): {'k2': [-0.02, -0.02, -0.04, 0.02], 'k6': [-0.08, -0.08, -0.02, -0.14], 'k8': [-0.18, -0.12, -0.18, -0.26]}.

## Decomposition: composition vs position vs subject damage

- Observed tail−head DiD -0.070; expected from head/tail subject-group composition alone -0.058; residual positional component -0.012.
- Humanities (n=49, 40/49 in the tail): paired Δ -0.327 vs rest -0.139; difference -0.188, permutation p = 0.0447.
- Of the 13 ref-wrong/k8-right items, 4 were ref budget truncations.

