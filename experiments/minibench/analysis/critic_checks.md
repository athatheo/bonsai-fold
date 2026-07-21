# critic_checks — completeness pass over the four anchor analyses

## 1. Integrity
- id sets identical across 4 configs: True; match frozen items: True; duplicates: 0; expected id pattern: True
- ref: n=500, correct=['bool'], gen_tokens=['int'], finish=['length', 'stop'], order==frozen: True
- k2: n=500, correct=['bool'], gen_tokens=['int'], finish=['length', 'stop'], order==frozen: True
- k6: n=500, correct=['bool'], gen_tokens=['int'], finish=['length', 'stop'], order==frozen: True
- k8: n=500, correct=['bool'], gen_tokens=['int'], finish=['length', 'stop'], order==frozen: True

| config | task_acc matches | macro matches | stored macro | micro | macro-micro (pts) |
|---|---|---|---|---|---|
| ref | True | True | 0.8413 | 0.8340 | 0.73 |
| k2 | True | True | 0.8125 | 0.8080 | 0.45 |
| k6 | True | True | 0.7762 | 0.7660 | 1.02 |
| k8 | True | True | 0.7500 | 0.7240 | 2.6 |

Censoring: every 'length' item has gen_tokens==16384 and no 'stop' item reaches budget; no nonpositive/over-budget token counts.

## 2. truncation => wrong
- Holds globally across all 2000 (config,item) pairs: True (violations: 0)

## 3. Damage decomposition (acc = (1-trunc)*cond_acc_on_stop)

| task | cfg | acc | Δacc | trunc% | Δtrunc | cond acc (stop) | Δcond | McNemar all (b,c,p) | McNemar both-stop (b,c,p) | losses trunc/total |
|---|---|---|---|---|---|---|---|---|---|---|
| gsm8k | ref | 0.910 | — | 6.0 | — | 0.968 | — | — | — | — |
| gsm8k | k2 | 0.920 | +1.0 | 6.0 | +0.0 | 0.979 | +1.1 | (2,3,1) | (1,2,1) | 1/2 |
| gsm8k | k6 | 0.880 | -3.0 | 6.0 | +0.0 | 0.936 | -3.2 | (5,2,0.453) | (2,1,1) | 3/5 |
| gsm8k | k8 | 0.870 | -4.0 | 8.0 | +2.0 | 0.946 | -2.2 | (7,3,0.344) | (3,2,1) | 4/7 |
| math500 | ref | 0.720 | — | 26.0 | — | 0.973 | — | — | — | — |
| math500 | k2 | 0.660 | -6.0 | 33.0 | +7.0 | 0.985 | +1.2 | (8,2,0.109) | (0,0,1) | 8/8 |
| math500 | k6 | 0.640 | -8.0 | 35.0 | +9.0 | 0.985 | +1.2 | (8,0,0.00781) | (0,0,1) | 8/8 |
| math500 | k8 | 0.720 | +0.0 | 26.0 | +0.0 | 0.973 | +0.0 | (6,6,1) | (0,0,1) | 6/6 |
| ifeval | ref | 0.930 | — | 5.0 | — | 0.979 | — | — | — | — |
| ifeval | k2 | 0.880 | -5.0 | 5.0 | +0.0 | 0.926 | -5.3 | (8,3,0.227) | (4,0,0.125) | 4/8 |
| ifeval | k6 | 0.860 | -7.0 | 11.0 | +6.0 | 0.966 | -1.3 | (9,2,0.0654) | (3,1,0.625) | 6/9 |
| ifeval | k8 | 0.790 | -14.0 | 20.0 | +15.0 | 0.988 | +0.9 | (18,4,0.00434) | (1,1,1) | 17/18 |
| mmlu | ref | 0.805 | — | 4.0 | — | 0.839 | — | — | — | — |
| mmlu | k2 | 0.790 | -1.5 | 4.5 | +0.5 | 0.827 | -1.1 | (16,13,0.711) | (11,7,0.481) | 5/16 |
| mmlu | k6 | 0.725 | -8.0 | 6.5 | +2.5 | 0.775 | -6.3 | (27,11,0.0139) | (20,9,0.0614) | 7/27 |
| mmlu | k8 | 0.620 | -18.5 | 6.0 | +2.0 | 0.660 | -17.9 | (50,13,3.02e-06) | (44,9,1.22e-06) | 6/50 |

## 4. IFEval instruction families
- items with 1 instruction: 65, >=2: 35 (max 3)

| family | n | ref | k2 | k6 | k8 | monotone? | k8 vs ref (b,c,p) |
|---|---|---|---|---|---|---|---|
| change_case | 18 | 16 | 16 | 11 | 13 | N | (4,1,0.375) |
| detectable_content | 14 | 14 | 14 | 14 | 12 | Y | (2,0,0.5) |
| detectable_format | 33 | 30 | 27 | 28 | 23 | N | (8,1,0.0391) |
| keywords | 21 | 19 | 16 | 16 | 13 | Y | (7,1,0.0703) |
| length_constraints | 25 | 23 | 22 | 22 | 21 | Y | (4,2,0.688) |
| punctuation | 9 | 8 | 8 | 8 | 8 | Y | (1,1,1) |
| startend | 15 | 15 | 14 | 12 | 12 | Y | (3,0,0.25) |

- families monotone along the chain: 5/7 (ALL n<=~25: individually underpowered)
- k2 losses vs ref: 8, of which 4 are k2 budget truncations; families: {'detectable_format': 4, 'keywords': 3, 'startend': 1, 'length_constraints': 1, 'punctuation': 1, 'change_case': 1}
- k6 losses vs ref: 9, of which 6 are k6 budget truncations; families: {'change_case': 5, 'keywords': 3, 'startend': 3, 'length_constraints': 2, 'punctuation': 1, 'detectable_format': 3}
- k8 losses vs ref: 18, of which 17 are k8 budget truncations; families: {'change_case': 4, 'keywords': 7, 'detectable_format': 8, 'startend': 3, 'length_constraints': 4, 'punctuation': 1, 'detectable_content': 2}
- paired delta multi-instruction (n=35) vs single (n=65): k2: -0.086 vs -0.031, k6: -0.171 vs -0.015, k8: -0.229 vs -0.092

## 5. GSM8K k2 anomaly
- acc: {'ref': 0.91, 'k2': 0.92, 'k6': 0.88, 'k8': 0.87}
- k2 vs ref: b=2, c=3, exact p=1
- recoveries with ref truncated: 1/3; losses with k2 truncated: 1/2
- flip details: {"k2_recoveries": [{"id": "gsm8k-018", "ref_tokens": 10508, "ref_finish": "stop", "k2_tokens": 6804, "k2_finish": "stop"}, {"id": "gsm8k-060", "ref_tokens": 4679, "ref_finish": "stop", "k2_tokens": 1768, "k2_finish": "stop"}, {"id": "gsm8k-089", "ref_tokens": 16384, "ref_finish": "length", "k2_tokens": 15106, "k2_finish": "stop"}], "k2_losses": [{"id": "gsm8k-010", "ref_tokens": 3110, "ref_finish": "stop", "k2_tokens": 2620, "k2_finish": "stop"}, {"id": "gsm8k-071", "ref_tokens": 1626, "ref_finish": "stop", "k2_tokens": 16384, "k2_finish": "length"}]}
- k8 gsm8k losses truncation-linked: 4/7

## 5b. Budget dependence of headline damage (all 500 items)

| config | regressions b | trunc-mediated | share | recoveries c | both-stop McNemar (b,c,p) |
|---|---|---|---|---|---|
| k2 | 34 | 18 | 0.529 | 21 | (16,9,0.23) |
| k6 | 49 | 24 | 0.49 | 15 | (25,11,0.0288) |
| k8 | 81 | 33 | 0.407 | 26 | (48,12,3.18e-06) |

- k8 ifeval truncations by #instructions: {2: 7, 3: 3, 1: 10} (base rates {1: 65, 2: 25, 3: 10})

## 6. KL anchors (re-read)
- on-policy: {"drop[16,12]": 0.007554768374456167, "drop[16,12,13,9]": 0.01770242127241983, "drop[16,12,13,9,8,4]": 0.03111115261509143, "drop[16,12,13,9,8,4,5,15]": 0.06426152843219171}
- off-policy: {"drop[16,12]": 0.033044363180810506, "drop[16,12,13,9]": 0.08642891088764396, "drop[16,12,13,9,8,4]": 0.204609790459549, "drop[16,12,13,9,8,4,5,15]": 0.3914830479536029}

## 7. Consistency spot checks
- a_mmlu_k8_b50_c13: True
- b_math_k6k8_discordants_3_11: True
- c_overall_trunc_counts: {'ref': 45, 'k2': 53, 'k6': 65, 'k8': 66}
- c_trunc_counts_match_45_53_65_66: True
- d_anchor_macros: {'ref': 0.84125, 'k2': 0.8125, 'k6': 0.77625, 'k8': 0.75}
- math500_mean_tokens_k6_k8: [9025.69, 9336.72]
- math500_trunc_k6_k8: [35, 26]
- math500_bothstop_k8_minus_k6_tokens: {'n': 62, 'mean': 920.7258064516129, 'median': 843.0}

