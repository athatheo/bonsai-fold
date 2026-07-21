# c_thinking_compensation — thinking-length inflation and wall-clock cost

Paired analysis over 500 items x 4 configs (identical id sets asserted). Bootstrap: B=10000, seed=20260721. Geometric-mean per-item token ratios vs reference; McNemar exact for accuracy deltas; Wilson 95% CIs for truncation rates.

KL anchors (mean nats, read from kl_screen results): k2: on=0.0076/off=0.0330, k6: on=0.0311/off=0.2046, k8: on=0.0643/off=0.3915

## 1. gen_tokens by config x task

| config | scope | mean | median | p90 | geomean ratio vs ref [95% CI] |
|---|---|---|---|---|---|
| ref | gsm8k | 2939 | 1536 | 5422 | 1.000 (def.) |
| ref | math500 | 8223 | 5148 | 16384 | 1.000 (def.) |
| ref | ifeval | 3421 | 2347 | 5169 | 1.000 (def.) |
| ref | mmlu | 2755 | 1336 | 6373 | 1.000 (def.) |
| ref | overall | 4018 | 2056 | 14567 | 1.000 (def.) |
| k2 | gsm8k | 3368 | 1571 | 10004 | 1.059 [0.964, 1.171] |
| k2 | math500 | 8759 | 5668 | 16384 | 1.057 [0.979, 1.145] |
| k2 | ifeval | 3345 | 2405 | 5072 | 1.004 [0.882, 1.139] |
| k2 | mmlu | 2711 | 1202 | 6061 | 0.964 [0.870, 1.071] |
| k2 | overall | 4179 | 1954 | 16384 | 1.009 [0.955, 1.067] |
| k6 | gsm8k | 3235 | 1684 | 7557 | 1.083 [0.964, 1.221] |
| k6 | math500 | 9026 | 6636 | 16384 | 1.116 [1.022, 1.226] |
| k6 | ifeval | 4384 | 2566 | 16384 | 1.207 [1.091, 1.338] |
| k6 | mmlu | 3018 | 1319 | 9179 | 1.049 [0.923, 1.188] |
| k6 | overall | 4536 | 2070 | 16384 | 1.099 [1.034, 1.169] |
| k8 | gsm8k | 3657 | 1830 | 10432 | 1.221 [1.062, 1.413] |
| k8 | math500 | 9337 | 7920 | 16384 | 1.260 [1.119, 1.428] |
| k8 | ifeval | 5469 | 2750 | 16384 | 1.367 [1.159, 1.622] |
| k8 | mmlu | 3023 | 1456 | 7781 | 1.119 [0.989, 1.265] |
| k8 | overall | 4902 | 2232 | 16384 | 1.213 [1.130, 1.303] |

Monotone in k (k2<=k6<=k8, geomean ratio): gsm8k: YES, math500: YES, ifeval: YES, mmlu: YES, overall: YES

## 2. Inflation split by outcome (ratio = cfg/ref tokens, geomean [95% CI])

| config | category | n | geomean ratio [95% CI] | trunc cfg/ref | flag |
|---|---|---|---|---|---|
| k2 | both_correct | 383 | 1.005 [0.956, 1.058] | 0/0 |  |
| k2 | recovery_cfg_correct_ref_wrong | 21 | 0.408 [0.268, 0.598] | 0/12 | UNDERPOWERED (n<30) |
| k2 | regression_cfg_wrong_ref_correct | 34 | 1.923 [1.417, 2.650] | 18/0 |  |
| k2 | both_wrong | 62 | 0.987 [0.853, 1.146] | 35/33 |  |
| k2 | cfg_correct | 404 | 0.959 [0.907, 1.013] | — |  |
| k2 | cfg_wrong | 96 | 1.250 [1.070, 1.476] | — |  |
| k2 | ref_correct | 417 | 1.059 [1.002, 1.121] | — |  |
| k2 | ref_wrong | 83 | 0.789 [0.662, 0.936] | — |  |
| k6 | both_correct | 368 | 1.092 [1.034, 1.155] | 0/0 |  |
| k6 | recovery_cfg_correct_ref_wrong | 15 | 0.379 [0.212, 0.668] | 0/4 | UNDERPOWERED (n<30) |
| k6 | regression_cfg_wrong_ref_correct | 49 | 2.056 [1.503, 2.816] | 24/0 |  |
| k6 | both_wrong | 68 | 0.918 [0.797, 1.068] | 41/41 |  |
| k6 | cfg_correct | 383 | 1.048 [0.984, 1.111] | — |  |
| k6 | cfg_wrong | 117 | 1.287 [1.086, 1.531] | — |  |
| k6 | ref_correct | 417 | 1.176 [1.103, 1.255] | — |  |
| k6 | ref_wrong | 83 | 0.783 [0.658, 0.927] | — |  |
| k8 | both_correct | 336 | 1.142 [1.071, 1.218] | 0/0 |  |
| k8 | recovery_cfg_correct_ref_wrong | 26 | 0.545 [0.371, 0.801] | 0/14 | UNDERPOWERED (n<30) |
| k8 | regression_cfg_wrong_ref_correct | 81 | 2.297 [1.792, 2.929] | 33/0 |  |
| k8 | both_wrong | 57 | 1.012 [0.860, 1.203] | 33/31 |  |
| k8 | cfg_correct | 362 | 1.083 [1.011, 1.158] | — |  |
| k8 | cfg_wrong | 138 | 1.637 [1.374, 1.948] | — |  |
| k8 | ref_correct | 417 | 1.308 [1.213, 1.409] | — |  |
| k8 | ref_wrong | 83 | 0.834 [0.699, 0.998] | — |  |

Reading: recovery items are dominated by reference FLAILING (ref often hits the 16384 budget) while the folded model stops early and gets it right, so their ratio is < 1. Regression items show the folded model flailing (ratio ~2x, high cfg truncation). Extra thinking co-occurs with FAILURE, not recovery; the genuine compensation signal is the modest inflation on both-correct items.
| k8 | deep recovery (k8 correct, k6 & ref wrong) | 16 | 0.711 [0.424, 1.207] | UNDERPOWERED (n<30) |

### McNemar exact (accuracy vs reference)

| config | scope | ref-only b | cfg-only c | acc delta | p (exact) |
|---|---|---|---|---|---|
| k2 | overall | 34 | 21 | -0.026 | 0.1048 |
| k2 | gsm8k | 2 | 3 | +0.010 | 1.0000 |
| k2 | math500 | 8 | 2 | -0.060 | 0.1094 |
| k2 | ifeval | 8 | 3 | -0.050 | 0.2266 |
| k2 | mmlu | 16 | 13 | -0.015 | 0.7111 |
| k6 | overall | 49 | 15 | -0.068 | 0.0000 |
| k6 | gsm8k | 5 | 2 | -0.030 | 0.4531 |
| k6 | math500 | 8 | 0 | -0.080 | 0.0078 |
| k6 | ifeval | 9 | 2 | -0.070 | 0.0654 |
| k6 | mmlu | 27 | 11 | -0.080 | 0.0139 |
| k8 | overall | 81 | 26 | -0.110 | 0.0000 |
| k8 | gsm8k | 7 | 3 | -0.040 | 0.3438 |
| k8 | math500 | 6 | 6 | +0.000 | 1.0000 |
| k8 | ifeval | 18 | 4 | -0.140 | 0.0043 |
| k8 | mmlu | 50 | 13 | -0.185 | 0.0000 |

## 3. Truncation (finish_reason != 'stop'); gen_tokens censored at 16384

| config | gsm8k | math500 | ifeval | mmlu | overall |
|---|---|---|---|---|---|
| ref | 6/100 = 6.0% [2.8%, 12.5%] | 26/100 = 26.0% [18.4%, 35.4%] | 5/100 = 5.0% [2.2%, 11.2%] | 8/200 = 4.0% [2.0%, 7.7%] | 45/500 = 9.0% [6.8%, 11.8%] |
| k2 | 6/100 = 6.0% [2.8%, 12.5%] | 33/100 = 33.0% [24.6%, 42.7%] | 5/100 = 5.0% [2.2%, 11.2%] | 9/200 = 4.5% [2.4%, 8.3%] | 53/500 = 10.6% [8.2%, 13.6%] |
| k6 | 6/100 = 6.0% [2.8%, 12.5%] | 35/100 = 35.0% [26.4%, 44.7%] | 11/100 = 11.0% [6.3%, 18.6%] | 13/200 = 6.5% [3.8%, 10.8%] | 65/500 = 13.0% [10.3%, 16.2%] |
| k8 | 8/100 = 8.0% [4.1%, 15.0%] | 26/100 = 26.0% [18.4%, 35.4%] | 20/100 = 20.0% [13.3%, 28.9%] | 12/200 = 6.0% [3.5%, 10.2%] | 66/500 = 13.2% [10.5%, 16.4%] |

Geomean ratio vs ref excluding pairs truncated in either config:

| config | scope | n kept (excl.) | geomean ratio [95% CI] |
|---|---|---|---|
| k2 | gsm8k | 93 (7) | 1.039 [0.947, 1.143] |
| k2 | math500 | 65 (35) | 1.004 [0.916, 1.100] |
| k2 | ifeval | 90 (10) | 1.047 [0.963, 1.137] |
| k2 | mmlu | 185 (15) | 0.960 [0.875, 1.053] |
| k2 | overall | 433 (67) | 1.001 [0.952, 1.051] |
| k6 | gsm8k | 91 (9) | 1.070 [0.974, 1.179] |
| k6 | math500 | 65 (35) | 1.103 [0.979, 1.251] |
| k6 | ifeval | 88 (12) | 1.128 [1.056, 1.209] |
| k6 | mmlu | 184 (16) | 0.985 [0.877, 1.103] |
| k6 | overall | 428 (72) | 1.049 [0.989, 1.111] |
| k8 | gsm8k | 90 (10) | 1.173 [1.048, 1.312] |
| k8 | math500 | 68 (32) | 1.299 [1.117, 1.527] |
| k8 | ifeval | 77 (23) | 1.072 [0.984, 1.171] |
| k8 | mmlu | 183 (17) | 1.067 [0.952, 1.194] |
| k8 | overall | 418 (82) | 1.125 [1.055, 1.199] |

## 4/5. Wall-clock accounting (bandwidth-bound decode)

net wall-clock factor = token ratio / speedup upper bound; > 1.000 means the folded pack is NET SLOWER per item despite being smaller.

| config | bytes removed | speedup UB | scope | tok ratio (geo) | net (geo) [95% CI] | net (stop-only) | net (aggregate) |
|---|---|---|---|---|---|---|---|
| k2 | 2.6% | 1.027x | gsm8k | 1.059 | 1.032 [0.939, 1.140] | 1.012 | 1.116 |
| k2 | 2.6% | 1.027x | math500 | 1.057 | 1.029 [0.954, 1.115] | 0.978 | 1.038 |
| k2 | 2.6% | 1.027x | ifeval | 1.004 | 0.978 [0.859, 1.109] | 1.020 | 0.952 |
| k2 | 2.6% | 1.027x | mmlu | 0.964 | 0.939 [0.847, 1.043] | 0.935 | 0.959 |
| k2 | 2.6% | 1.027x | overall | 1.009 | 0.983 [0.930, 1.039] | 0.975 | 1.013 |
| k6 | 7.8% | 1.085x | gsm8k | 1.083 | 0.999 [0.889, 1.126] | 0.986 | 1.015 |
| k6 | 7.8% | 1.085x | math500 | 1.116 | 1.029 [0.942, 1.130] | 1.017 | 1.012 |
| k6 | 7.8% | 1.085x | ifeval | 1.207 | 1.113 [1.006, 1.234] | 1.040 | 1.182 |
| k6 | 7.8% | 1.085x | mmlu | 1.049 | 0.967 [0.851, 1.095] | 0.908 | 1.010 |
| k6 | 7.8% | 1.085x | overall | 1.099 | 1.013 [0.953, 1.078] | 0.967 | 1.041 |
| k8 | 10.4% | 1.116x | gsm8k | 1.221 | 1.094 [0.951, 1.266] | 1.051 | 1.115 |
| k8 | 10.4% | 1.116x | math500 | 1.260 | 1.129 [1.002, 1.279] | 1.164 | 1.017 |
| k8 | 10.4% | 1.116x | ifeval | 1.367 | 1.225 [1.039, 1.453] | 0.961 | 1.432 |
| k8 | 10.4% | 1.116x | mmlu | 1.119 | 1.002 [0.886, 1.133] | 0.956 | 0.983 |
| k8 | 10.4% | 1.116x | overall | 1.213 | 1.087 [1.012, 1.167] | 1.008 | 1.093 |

Prior-observation check (math500 mean tokens): ref 8223, k2 8759, k6 9026, k8 9337.

Caveat: speedup upper bound assumes decode throughput scales exactly with deployed bytes (perfect bandwidth-bound scaling) and ignores prefill; actual tok/s must be measured on this machine (timed decode of a fixed prompt set on each pack) to firm up net wall-clock. Truncated items censor gen_tokens at 16384, biasing inflation DOWN for configs that truncate more.
