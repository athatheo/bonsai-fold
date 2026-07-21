# verify_d_kl_fit — independent re-derivation of d_kl_fit (2026-07-21)

Checks passed: 56/56

| check | ok | re-derived value |
|---|---|---|
| 500 unique ids in reference | PASS | 500 |
| identical id sets across 4 configs | PASS |  |
| task Ns 100/100/100/200 | PASS | {'gsm8k': 100, 'math500': 100, 'ifeval': 100, 'mmlu': 200} |
| stored task_accuracy matches recomputed: reference/gsm8k | PASS | recomputed 0.910000 stored 0.91 |
| stored task_accuracy matches recomputed: reference/math500 | PASS | recomputed 0.720000 stored 0.72 |
| stored task_accuracy matches recomputed: reference/ifeval | PASS | recomputed 0.930000 stored 0.93 |
| stored task_accuracy matches recomputed: reference/mmlu | PASS | recomputed 0.805000 stored 0.805 |
| stored macro_avg matches recomputed: reference | PASS | recomputed 0.841250 stored 0.84125 |
| stored task_accuracy matches recomputed: k2/gsm8k | PASS | recomputed 0.920000 stored 0.92 |
| stored task_accuracy matches recomputed: k2/math500 | PASS | recomputed 0.660000 stored 0.66 |
| stored task_accuracy matches recomputed: k2/ifeval | PASS | recomputed 0.880000 stored 0.88 |
| stored task_accuracy matches recomputed: k2/mmlu | PASS | recomputed 0.790000 stored 0.79 |
| stored macro_avg matches recomputed: k2 | PASS | recomputed 0.812500 stored 0.8125 |
| stored task_accuracy matches recomputed: k6/gsm8k | PASS | recomputed 0.880000 stored 0.88 |
| stored task_accuracy matches recomputed: k6/math500 | PASS | recomputed 0.640000 stored 0.64 |
| stored task_accuracy matches recomputed: k6/ifeval | PASS | recomputed 0.860000 stored 0.86 |
| stored task_accuracy matches recomputed: k6/mmlu | PASS | recomputed 0.725000 stored 0.725 |
| stored macro_avg matches recomputed: k6 | PASS | recomputed 0.776250 stored 0.77625 |
| stored task_accuracy matches recomputed: k8/gsm8k | PASS | recomputed 0.870000 stored 0.87 |
| stored task_accuracy matches recomputed: k8/math500 | PASS | recomputed 0.720000 stored 0.72 |
| stored task_accuracy matches recomputed: k8/ifeval | PASS | recomputed 0.790000 stored 0.79 |
| stored task_accuracy matches recomputed: k8/mmlu | PASS | recomputed 0.620000 stored 0.62 |
| stored macro_avg matches recomputed: k8 | PASS | recomputed 0.750000 stored 0.75 |
| macro drops 0/2.875/6.5/9.125 | PASS | {'reference': 0.0, 'k2': 2.875, 'k6': 6.5, 'k8': 9.125} |
| claimed KL_on anchors | PASS | {'reference': '0.000000', 'k2': '0.007555', 'k6': '0.031111', 'k8': '0.064262'} |
| claimed KL_off anchors | PASS | {'k2': '0.033044', 'k6': '0.204610', 'k8': '0.391483'} |
| sqrt coef ~36.04, RMSE ~0.15, R2 ~0.998 | PASS | c=36.0405 rmse=0.1474 r2=0.99820 |
| sqrt residuals <0.3 pts at all anchors | PASS | [0.0, -0.2576, 0.143, -0.0112] |
| linear-origin slope ~157.2, RMSE ~1.26, R2 ~0.867 | PASS | b=157.207 rmse=1.2641 r2=0.86745 |
| linear-origin misses k2 by +1.7 pts | PASS | resid k2 = +1.6873 |
| linear+intercept: intercept ~1.19, R2 ~0.925 | PASS | a=1.1925 b=133.395 r2=0.92480 |
| concavity: drop/KL falls ~380 -> ~209 -> ~142 pts/nat | PASS | [380.6, 208.9, 142.0] |
| boot linear slope 1.57 pts/0.01nats, CI ~[1.00, 2.14] | PASS | point 1.5721 CI [0.9954, 2.1445] |
| boot sqrt coef CI ~[22.8, 49.1] | PASS | CI [22.781, 49.053] |
| macro-drop CIs k2/k6/k8 | PASS | {'k2': [0.117, 5.673], 'k6': [3.725, 9.38], 'k8': [5.391, 12.794]} |
| tax(2..8) = 1.000/1.175/1.334/1.878 | PASS | {'k2': 0.9999, 'k4': 1.175, 'k6': 1.3345, 'k8': 1.8778} |
| off/on ratios 4.37/4.88/6.58/6.09 mean 5.48 pooled 5.93 | PASS | {'k2': 4.374, 'k4': 4.882, 'k6': 6.577, 'k8': 6.092, 'mean': 5.4813, 'pooled': 5.9319} |
| 1.5pt budget 0.00173 CI [0.00093,0.00434], off 0.00949, blk 1.15e-4 | PASS | {'sqrt_set_nats': '0.001732', 'sqrt_ci': ['0.000935', '0.004336'], 'off_ceiling_nats': '0.009495', 'per_block_k8_nats': '0.000115', 'lin_set_nats': '0.009542', 'lin_per_block_nats': '0.000635'} |
| 3.0pt budget 0.00693 CI [0.00374,0.01734], off 0.0380, blk 4.61e-4; lin 0.0191/1.27e-3 | PASS | {'sqrt_set_nats': '0.006929', 'sqrt_ci': ['0.003740', '0.017342'], 'off_ceiling_nats': '0.037979', 'per_block_k8_nats': '0.000461', 'lin_set_nats': '0.019083', 'lin_per_block_nats': '0.001270'} |
| old 0.05-nats gate -> ~8.06 (sqrt) / 7.86 (linear) pts | PASS | sqrt 8.059 lin 7.860 |
| 30x tighter: 0.05 / 0.00173 ~ 29 | PASS | 28.9x |
| best single block = drop[12] @ 0.00330 nats > both per-block thresholds | PASS | drop[12] 0.003303 |
| k~8 infeasible even before tax (8 cheapest singles sum >> budgets) | PASS | sum(8 cheapest singles)=0.03422 vs budgets 0.00693(sqrt)/0.01908(lin) |
| McNemar k2 all: (34,21) p=0.105 | PASS | (34,21) p=0.1048 |
| McNemar k6 all: (49,15) p=2.4e-5 | PASS | p=2.44e-05 |
| McNemar k8 all: (81,26) p=9.4e-8 | PASS | p=9.37e-08 |
| k8 mmlu -18.5 pts p~3e-6; k8 ifeval -14 pts p=0.0043 | PASS | mmlu p=3.02e-06 ifeval p=0.00434 |
| math500 k8 vs ref: delta 0.0, (6,6), p=1.0 | PASS |  |
| math500 k8 vs k6: +8 pts, (3,11), p=0.0574 | PASS | (3,11) p=0.0574 |
| per-task lin slopes mmlu 2.81 > ifeval 2.24 >> gsm8k 0.67 ~ math500 0.57 | PASS | {'gsm8k': 0.665, 'math500': 0.571, 'ifeval': 2.241, 'mmlu': 2.811} |
| math500 sqrt residuals +-4.4..4.8 pts | PASS | [0.0, 4.368, 4.688, -4.76] |
| frozen ids match bench ids | PASS |  |
| 57 MMLU subjects | PASS | 57 |
| k8 mmlu drop cluster CI ~[9.7, 27.4] vs iid ~[11.1, 25.7] | PASS | cluster [9.722, 27.369] iid [11.111, 25.743] |
| k4 KL_on 0.0177; predicted 4.8 (sqrt) vs 2.8 (lin) pts | PASS | KL 0.01770 sqrt 4.795 lin 2.783 |
| k2 measured 2.875 pts: passes 3.0-pt tolerance, fails 1.5-pt | PASS | 2.875 |

