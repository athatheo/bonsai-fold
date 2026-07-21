# verify_b_math_anomaly — independent re-derivation

All numbers computed from raw JSONs, paired by id (id sets asserted identical across all 4 configs and frozen items).

## Wilson 95% CIs (math500, z=1.96)

| config | acc | CI |
|---|---|---|
| ref | 0.72 | [0.625, 0.799] |
| k2 | 0.66 | [0.563, 0.745] |
| k6 | 0.64 | [0.542, 0.727] |
| k8 | 0.72 | [0.625, 0.799] |

## Flip tables math500 (McNemar exact two-sided)

| pair | both✓ | A-only | B-only | both✗ | p |
|---|---|---|---|---|---|
| ref_vs_k6 | 64 | 8 | 0 | 28 | 0.007812 |
| ref_vs_k8 | 66 | 6 | 6 | 22 | 1.000000 |
| k6_vs_k8 | 61 | 3 | 11 | 25 | 0.057373 |
| ref_vs_k2 | 64 | 8 | 2 | 26 | 0.109375 |
| k2_vs_k6 | 61 | 5 | 3 | 31 | 0.726562 |
| k2_vs_k8 | 63 | 3 | 9 | 25 | 0.145996 |

k6 vs k8 one-sided exact binomial P(X>=11 of 14) = 0.028687

## gsm8k companion

- ref_vs_k6: 86/5/2/7, p=0.4531
- ref_vs_k8: 84/7/3/6, p=0.3438
- k6_vs_k8: 82/6/5/7, p=1.0000
- gsm8k truncations: {'ref': 6, 'k2': 6, 'k6': 6, 'k8': 8}

## Truncation x correctness (math500)

| config | stop✓ | stop✗ | len✓ | len✗ | wrong-share-from-trunc |
|---|---|---|---|---|---|
| ref | 72 | 2 | 0 | 26 | 0.9286 |
| k2 | 66 | 1 | 0 | 33 | 0.9706 |
| k6 | 64 | 1 | 0 | 35 | 0.9722 |
| k8 | 72 | 2 | 0 | 26 | 0.9286 |

length_correct == 0 everywhere: True

## k6/k8 flip mechanism

- recoveries n=11: all k6 truncated at budget = True; all k8 stop = True; k8 tokens mean 11003.6 / median 12528; ref correct 5/11
- losses n=3: all k8 truncated = True; k6 tokens range [2734, 7951]; ref correct 3/3
- all 8 ref→k6 losses are k6 truncations: True

## Paired gen_tokens k8−k6 (math500)

- all 100: mean 311.03, median 0.0, sign +41/−36 (ties 23), p=0.6488
- both-correct n=61: mean 946.2, median 884, sign +38/−23, p=0.0722

## Chain + KL

- chain non-monotone items (≥1 recovery): 14/100
- KL strictly monotone k2<k6<k8: True
- on-policy: [0.008, 0.031, 0.064]; off-policy: [0.033, 0.205, 0.391]
