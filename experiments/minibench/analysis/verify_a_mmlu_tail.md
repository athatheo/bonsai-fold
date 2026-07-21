# verify_a_mmlu_tail — independent re-derivation

Seed=12345 (different from original), 20000 permutations, stdlib exact tests, pairing by id, defensive dtype coercion.

**93 checks, 1 failed.**

| check | mine | theirs | ok |
|---|---|---|---|
| acc_ref | 0.805 | 0.805 | PASS |
| wilson_ref_lo | 0.745 | 0.745 | PASS |
| wilson_ref_hi | 0.854 | 0.854 | PASS |
| file_task_acc_ref | 0.805 | 0.805 | PASS |
| acc_k2 | 0.79 | 0.79 | PASS |
| wilson_k2_lo | 0.728 | 0.728 | PASS |
| wilson_k2_hi | 0.841 | 0.841 | PASS |
| file_task_acc_k2 | 0.79 | 0.79 | PASS |
| acc_k6 | 0.725 | 0.725 | PASS |
| wilson_k6_lo | 0.659 | 0.659 | PASS |
| wilson_k6_hi | 0.782 | 0.782 | PASS |
| file_task_acc_k6 | 0.725 | 0.725 | PASS |
| acc_k8 | 0.62 | 0.62 | PASS |
| wilson_k8_lo | 0.551 | 0.551 | PASS |
| wilson_k8_hi | 0.684 | 0.684 | PASS |
| file_task_acc_k8 | 0.62 | 0.62 | PASS |
| mcnemar_b | 50 | 50 | PASS |
| mcnemar_c | 13 | 13 | PASS |
| mcnemar_n_discordant | 63 | 63 | PASS |
| mcnemar_p | 3.015952084642337e-06 | 3e-06 | PASS |
| mcnemar_p_minlik_equals_doubling | 3.015952084642337e-06 | 3.015952084642337e-06 | PASS |
| delta_acc | -0.185 | -0.185 | PASS |
| loss_share_ref_correct | 31.1 | 31.1 | PASS |
| group_STEM_n | 56 | 56 | PASS |
| group_STEM_delta | -0.143 | -0.143 | PASS |
| group_STEM_p | 0.03857421875 | 0.039 | PASS |
| group_humanities_n | 49 | 49 | PASS |
| group_humanities_delta | -0.327 | -0.327 | PASS |
| group_humanities_p | 3.0517578125e-05 | 3.1e-05 | PASS |
| group_social_sciences_n | 44 | 44 | PASS |
| group_social_sciences_delta | -0.114 | -0.114 | PASS |
| group_social_sciences_p | 0.3017578125 | 0.3 | PASS |
| group_other_n | 51 | 51 | PASS |
| group_other_delta | -0.157 | -0.157 | PASS |
| group_other_p | 0.11531829833984375 | 0.115 | PASS |
| group_hum_b | 16 | 16 | PASS |
| group_hum_c | 0 | 0 | PASS |
| hum_delta | -0.327 | -0.327 | PASS |
| rest_delta | -0.139 | -0.139 | PASS |
| hum_rest_diff | -0.187 | -0.188 | **FAIL** |
| hum_rest_perm_p | 0.04429778511074446 | 0.045 | PASS |
| pos_k2_did | 0.01 | 0.01 | PASS |
| pos_k6_did | 0.0 | 0.0 | PASS |
| pos_k8_head | -0.15 | -0.15 | PASS |
| pos_k8_tail | -0.22 | -0.22 | PASS |
| pos_k8_did | -0.07 | -0.07 | PASS |
| pos_k8_perm_p | 0.4258287085645718 | 0.421 | PASS |
| head_STEM | 51 | 51 | PASS |
| head_humanities | 9 | 9 | PASS |
| tail_STEM | 5 | 5 | PASS |
| tail_humanities | 40 | 40 | PASS |
| comp_did | -0.058 | -0.058 | PASS |
| comp_residual | -0.012 | -0.012 | PASS |
| comp_share_pct | 83 | 83 | PASS |
| len_ref | 8 | 8 | PASS |
| acc_on_len_ref | 0.0 | 0.0 | PASS |
| zero_tokens_ref | 0 | 0 | PASS |
| len_k2 | 9 | 9 | PASS |
| acc_on_len_k2 | 0.0 | 0.0 | PASS |
| zero_tokens_k2 | 0 | 0 | PASS |
| len_k6 | 13 | 13 | PASS |
| acc_on_len_k6 | 0.0 | 0.0 | PASS |
| zero_tokens_k6 | 0 | 0 | PASS |
| len_k8 | 12 | 12 | PASS |
| acc_on_len_k8 | 0.0 | 0.0 | PASS |
| zero_tokens_k8 | 0 | 0 | PASS |
| len_k8_head | 4 | 4 | PASS |
| len_k8_tail | 8 | 8 | PASS |
| k8_only_n | 50 | 50 | PASS |
| k8_only_budget_hit | 6 | 6 | PASS |
| ref_only_n | 13 | 13 | PASS |
| ref_only_ref_truncated | 4 | 4 | PASS |
| trunc_upper_bound | 0.03 | 0.03 | PASS |
| trunc_net | 0.01 | 0.01 | PASS |
| worst_window_start | 141 | 141 | PASS |
| worst_window_delta | -0.45 | -0.45 | PASS |
| worst_window_first_subject | management | management | PASS |
| worst_window_has_nutrition | True | True | PASS |
| best_split_t | 143 | 143 | PASS |
| best_split_gap | 0.183 | 0.183 | PASS |
| split_on_boundary | 0 | 0 | PASS |
| frac_close_pct | 28 | 28 | PASS |
| n_boundaries | 57 | 57 | PASS |
| maxgap_p_step5 | 0.736013199340033 | 0.737 | PASS |
| maxgap_p_full_ge_step5_and_ns | True | True | PASS |
| quartiles_k8 | [-0.18, -0.12, -0.18, -0.26] | [-0.18, -0.12, -0.18, -0.26] | PASS |
| kl_onpolicy_k2 | 0.00755 | 0.00755 | PASS |
| kl_onpolicy_k6 | 0.03111 | 0.03111 | PASS |
| kl_onpolicy_k8 | 0.06426 | 0.06426 | PASS |
| kl_offpolicy_k2 | 0.03304 | 0.03304 | PASS |
| kl_offpolicy_k6 | 0.20461 | 0.20461 | PASS |
| kl_offpolicy_k8 | 0.39148 | 0.39148 | PASS |
| acc_monotone_decreasing_in_kl | True | True | PASS |

Notes:
- Raw dtypes: `correct` is bool and `gen_tokens` is int in all four result files; coercion paths untriggered.
- McNemar doubling and min-likelihood two-sided p agree exactly (symmetric null).
- Original max-gap null used a step-5 t-scan while the observed gap used step-1 (mildly anti-conservative). Full step-1 null gives p=0.82456 vs step-5 p=0.73601 — both decisively non-significant; conclusion unchanged.
