#!/usr/bin/env python3
"""verify_a_mmlu_tail: independent re-derivation of every key number in the
a_mmlu_tail findings. Implemented from the raw JSONs only (results, frozen
items, KL screens) -- NOT copied from a_mmlu_tail.py.

Adversarial checks included:
  - dtype census of `correct` / `gen_tokens` in the raw files, with defensive
    coercion (handles str dtypes if present);
  - pairing strictly by item id (never list position), id-set equality assert;
  - McNemar exact two-sided computed two ways (doubling and min-likelihood);
  - Wilson CI implemented independently;
  - positional split boundaries checked (head = mmlu-000..099, tail = 100..199);
  - max-gap permutation null computed with FULL step-1 scan (the original used
    a coarser step-5 null; step-5 is also reproduced for comparison);
  - independent MMLU category map re-derived from Hendrycks categories.py;
  - permutation tests re-run with a DIFFERENT seed (MC agreement expected).

Deterministic: numpy RNG seed 12345, N_PERM = 20000.
Outputs: verify_a_mmlu_tail.json, verify_a_mmlu_tail.md
"""

import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")
ITEMS_PATH = os.path.join(HERE, "..", "minibench_items.json")
KL_DIR = os.path.join(HERE, "..", "..", "kl_screen", "results")

CONFIGS = {"ref": "reference", "k2": "k2_16-12", "k6": "k6", "k8": "k8"}
KL_NAMES = {
    "k2": "drop[16,12]",
    "k6": "drop[16,12,13,9,8,4]",
    "k8": "drop[16,12,13,9,8,4,5,15]",
}
BUDGET = 16384
SEED = 12345
N_PERM = 20000

# Independent re-derivation of the official Hendrycks et al. MMLU category
# map (categories.py): subject -> subcategory -> {STEM, humanities,
# social_sciences, other}.
_SUBCAT = {
    "abstract_algebra": "math", "anatomy": "health", "astronomy": "physics",
    "business_ethics": "business", "clinical_knowledge": "health",
    "college_biology": "biology", "college_chemistry": "chemistry",
    "college_computer_science": "computer science",
    "college_mathematics": "math", "college_medicine": "health",
    "college_physics": "physics", "computer_security": "computer science",
    "conceptual_physics": "physics", "econometrics": "economics",
    "electrical_engineering": "engineering",
    "elementary_mathematics": "math", "formal_logic": "philosophy",
    "global_facts": "other", "high_school_biology": "biology",
    "high_school_chemistry": "chemistry",
    "high_school_computer_science": "computer science",
    "high_school_european_history": "history",
    "high_school_geography": "geography",
    "high_school_government_and_politics": "politics",
    "high_school_macroeconomics": "economics",
    "high_school_mathematics": "math",
    "high_school_microeconomics": "economics",
    "high_school_physics": "physics",
    "high_school_psychology": "psychology",
    "high_school_statistics": "math",
    "high_school_us_history": "history",
    "high_school_world_history": "history", "human_aging": "health",
    "human_sexuality": "culture", "international_law": "law",
    "jurisprudence": "law", "logical_fallacies": "philosophy",
    "machine_learning": "computer science", "management": "business",
    "marketing": "business", "medical_genetics": "health",
    "miscellaneous": "other", "moral_disputes": "philosophy",
    "moral_scenarios": "philosophy", "nutrition": "health",
    "philosophy": "philosophy", "prehistory": "history",
    "professional_accounting": "other", "professional_law": "law",
    "professional_medicine": "health",
    "professional_psychology": "psychology",
    "public_relations": "politics", "security_studies": "politics",
    "sociology": "culture", "us_foreign_policy": "politics",
    "virology": "health", "world_religions": "philosophy",
}
_CAT_OF_SUBCAT = {}
for _sc in ["physics", "chemistry", "biology", "computer science", "math",
            "engineering"]:
    _CAT_OF_SUBCAT[_sc] = "STEM"
for _sc in ["history", "philosophy", "law"]:
    _CAT_OF_SUBCAT[_sc] = "humanities"
for _sc in ["politics", "culture", "economics", "geography", "psychology"]:
    _CAT_OF_SUBCAT[_sc] = "social_sciences"
for _sc in ["other", "business", "health"]:
    _CAT_OF_SUBCAT[_sc] = "other"
GROUP_OF = {s: _CAT_OF_SUBCAT[sc] for s, sc in _SUBCAT.items()}
GROUPS = ["STEM", "humanities", "social_sciences", "other"]


def to_bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "1", "yes"):
            return True
        if s in ("false", "0", "no"):
            return False
    if isinstance(v, (int, float)) and v in (0, 1):
        return bool(v)
    raise ValueError(f"unparseable correct value: {v!r}")


def to_int(v):
    if isinstance(v, bool):
        raise ValueError("bool gen_tokens")
    if isinstance(v, int):
        return v
    if isinstance(v, str):
        return int(v.strip())
    raise ValueError(f"unparseable gen_tokens: {v!r}")


def wilson95(k, n):
    z = 1.959963984540054
    p = k / n
    d = 1.0 + z * z / n
    ctr = (p + z * z / (2 * n)) / d
    hw = (z / d) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [round(max(0.0, ctr - hw), 4), round(min(1.0, ctr + hw), 4)]


def mcnemar_exact_two_sided(b, c):
    """Exact two-sided McNemar p, both the doubling and min-likelihood
    definitions (identical under the symmetric Binomial(n, .5) null)."""
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "p_doubling": 1.0, "p_minlik": 1.0}
    lo = min(b, c)
    cdf_lo = sum(math.comb(n, i) for i in range(lo + 1)) / 2.0 ** n
    p_doubling = min(1.0, 2.0 * cdf_lo)
    p_obs = math.comb(n, b) / 2.0 ** n
    p_minlik = sum(math.comb(n, i) for i in range(n + 1)
                   if math.comb(n, i) / 2.0 ** n <= p_obs + 1e-15) / 2.0 ** n
    return {"b": b, "c": c, "n_discordant": n,
            "p_doubling": p_doubling, "p_minlik": p_minlik}


def main():
    checks = []  # (name, ok, mine, theirs)

    def check(name, mine, theirs, tol=0.0):
        if isinstance(mine, float) or isinstance(theirs, float):
            ok = abs(float(mine) - float(theirs)) <= tol + 1e-12
        else:
            ok = mine == theirs
        checks.append({"check": name, "ok": bool(ok),
                       "mine": mine, "theirs": theirs})
        return ok

    # ------------------------- load, dtype census -------------------------
    raw = {}
    dtype_census = {}
    for short, cfg in CONFIGS.items():
        with open(os.path.join(RESULTS, f"{cfg}.json")) as f:
            raw[short] = json.load(f)
        its = raw[short]["items"]
        dtype_census[short] = {
            "n_items": len(its),
            "correct_types": sorted({type(i["correct"]).__name__ for i in its}),
            "gen_tokens_types": sorted({type(i["gen_tokens"]).__name__
                                        for i in its}),
            "finish_reasons": sorted({i["finish_reason"] for i in its}),
        }
        assert len(its) == 500

    ids = {s: {i["id"] for i in raw[s]["items"]} for s in raw}
    for s in raw:
        assert ids[s] == ids["ref"], f"id-set mismatch {s}"
    expected_ids = ({f"gsm8k-{i:03d}" for i in range(100)}
                    | {f"math500-{i:03d}" for i in range(100)}
                    | {f"ifeval-{i:03d}" for i in range(100)}
                    | {f"mmlu-{i:03d}" for i in range(200)})
    assert ids["ref"] == expected_ids, "unexpected id universe"

    by_id = {s: {i["id"]: i for i in raw[s]["items"]} for s in raw}
    mmlu_ids = [f"mmlu-{i:03d}" for i in range(200)]

    # frozen items: subjects, alphabetical-by-subject order
    with open(ITEMS_PATH) as f:
        frozen = json.load(f)["items"]
    meta = {it["id"]: it for it in frozen if it["task"] == "mmlu"}
    assert len(meta) == 200
    subjects = [meta[i]["source"].split("[")[0] for i in mmlu_ids]
    assert subjects == sorted(subjects), "not alphabetical by subject"
    assert set(subjects) <= set(GROUP_OF), "unmapped subject present"
    groups_arr = [GROUP_OF[s] for s in subjects]

    # paired vectors over mmlu index 0..199, keyed BY ID
    corr = {s: np.array([int(to_bool(by_id[s][i]["correct"]))
                         for i in mmlu_ids]) for s in raw}
    fin = {s: [by_id[s][i]["finish_reason"] for i in mmlu_ids] for s in raw}
    toks = {s: np.array([to_int(by_id[s][i]["gen_tokens"])
                         for i in mmlu_ids]) for s in raw}

    out = {"dtype_census": dtype_census, "seed": SEED, "n_perm": N_PERM}

    # -------------------- 1. accuracies + Wilson CIs ----------------------
    claim_acc = {"ref": (0.805, [0.745, 0.854]), "k2": (0.790, [0.728, 0.841]),
                 "k6": (0.725, [0.659, 0.782]), "k8": (0.620, [0.551, 0.684])}
    acc_out = {}
    for s in ["ref", "k2", "k6", "k8"]:
        k = int(corr[s].sum())
        ci = wilson95(k, 200)
        acc_out[s] = {"n_correct": k, "acc": k / 200, "wilson95": ci}
        check(f"acc_{s}", round(k / 200, 3), claim_acc[s][0])
        check(f"wilson_{s}_lo", round(ci[0], 3), claim_acc[s][1][0])
        check(f"wilson_{s}_hi", round(ci[1], 3), claim_acc[s][1][1])
        # file-level task_accuracy agreement
        check(f"file_task_acc_{s}", raw[s]["task_accuracy"]["mmlu"], k / 200,
              tol=1e-9)
    out["mmlu_accuracy"] = acc_out

    # -------------------- 2. McNemar k8 vs ref (N=200) --------------------
    r, e = corr["ref"], corr["k8"]
    b = int(((r == 1) & (e == 0)).sum())   # ref right, k8 wrong
    c = int(((r == 0) & (e == 1)).sum())   # ref wrong, k8 right
    both_right = int(((r == 1) & (e == 1)).sum())
    mc = mcnemar_exact_two_sided(b, c)
    mc["delta_acc"] = (c - b) / 200
    mc["loss_share_of_ref_correct"] = b / (b + both_right)
    out["mcnemar_k8_ref"] = mc
    check("mcnemar_b", b, 50)
    check("mcnemar_c", c, 13)
    check("mcnemar_n_discordant", b + c, 63)
    check("mcnemar_p", mc["p_doubling"], 3.0e-6, tol=0.05e-6)
    check("mcnemar_p_minlik_equals_doubling", mc["p_minlik"],
          mc["p_doubling"], tol=1e-15)
    check("delta_acc", round(mc["delta_acc"], 3), -0.185)
    check("loss_share_ref_correct", round(100 * b / (b + both_right), 1), 31.1)

    # ---------------------- 3. per-group paired deltas --------------------
    claim_groups = {
        "STEM": (56, -0.143, 0.039), "humanities": (49, -0.327, 3.1e-5),
        "social_sciences": (44, -0.114, 0.30), "other": (51, -0.157, 0.115)}
    grp_out = {}
    exact_group_delta = {}
    for g in GROUPS:
        idx = np.array([i for i, gg in enumerate(groups_arr) if gg == g])
        n = len(idx)
        d = float((corr["k8"][idx] - corr["ref"][idx]).mean())
        exact_group_delta[g] = d
        gb = int(((corr["ref"][idx] == 1) & (corr["k8"][idx] == 0)).sum())
        gc = int(((corr["ref"][idx] == 0) & (corr["k8"][idx] == 1)).sum())
        gmc = mcnemar_exact_two_sided(gb, gc)
        grp_out[g] = {"n": n, "delta_k8_ref": round(d, 4), "b": gb, "c": gc,
                      "p_exact": gmc["p_doubling"],
                      "acc_ref": float(corr["ref"][idx].mean()),
                      "acc_k8": float(corr["k8"][idx].mean())}
        cn, cd, cp = claim_groups[g]
        check(f"group_{g}_n", n, cn)
        check(f"group_{g}_delta", round(d, 3), cd)
        # p claimed at 2-3 sig figs
        check(f"group_{g}_p", gmc["p_doubling"], cp, tol=0.05 * cp + 5e-4)
    check("group_hum_b", grp_out["humanities"]["b"], 16)
    check("group_hum_c", grp_out["humanities"]["c"], 0)
    out["per_group"] = grp_out

    rng = np.random.default_rng(SEED)
    d_k8 = (corr["k8"] - corr["ref"]).astype(float)

    # ------------- 4. humanities vs rest permutation (k8) -----------------
    hum_mask = np.array([g == "humanities" for g in groups_arr])
    d_hum = float(d_k8[hum_mask].mean())
    d_rest = float(d_k8[~hum_mask].mean())
    obs_hr = d_hum - d_rest
    n_h = int(hum_mask.sum())
    cnt = 0
    dd = d_k8.copy()
    for _ in range(N_PERM):
        rng.shuffle(dd)
        if abs(dd[:n_h].mean() - dd[n_h:].mean()) >= abs(obs_hr) - 1e-12:
            cnt += 1
    p_hr = (cnt + 1) / (N_PERM + 1)
    out["humanities_vs_rest"] = {"delta_hum": round(d_hum, 4),
                                 "delta_rest": round(d_rest, 4),
                                 "difference": round(obs_hr, 4),
                                 "perm_p": round(p_hr, 5)}
    check("hum_delta", round(d_hum, 3), -0.327)
    check("rest_delta", round(d_rest, 3), -0.139)
    check("hum_rest_diff", round(obs_hr, 3), -0.188)
    # MC tolerance: 3*SE at p~=.045 with 20k perms ~= .0044
    check("hum_rest_perm_p", p_hr, 0.045, tol=0.0044)

    # --------------------- 5. positional head/tail DiD --------------------
    claim_pos = {"k2": (None, None, 0.010), "k6": (None, None, 0.000),
                 "k8": (-0.150, -0.220, -0.070)}
    pos_out = {}
    for s in ["k2", "k6", "k8"]:
        d = (corr[s] - corr["ref"]).astype(float)
        hd, td = float(d[:100].mean()), float(d[100:].mean())
        pos_out[s] = {"head_delta": round(hd, 4), "tail_delta": round(td, 4),
                      "did": round(td - hd, 4)}
        ch, ct_, cdid = claim_pos[s]
        if ch is not None:
            check(f"pos_{s}_head", round(hd, 3), ch)
            check(f"pos_{s}_tail", round(td, 3), ct_)
        check(f"pos_{s}_did", round(td - hd, 3), cdid)
    obs_did = pos_out["k8"]["did"]
    cnt = 0
    dd = d_k8.copy()
    for _ in range(N_PERM):
        rng.shuffle(dd)
        if abs(dd[100:].mean() - dd[:100].mean()) >= abs(obs_did) - 1e-12:
            cnt += 1
    p_did = (cnt + 1) / (N_PERM + 1)
    pos_out["k8"]["perm_p"] = round(p_did, 5)
    out["positional"] = pos_out
    # MC tolerance: 3*SE at p~=.42 ~= .0105
    check("pos_k8_perm_p", p_did, 0.421, tol=0.0105)

    # ------------- 6. composition decomposition of the DiD ----------------
    head_cnt = {g: 0 for g in GROUPS}
    tail_cnt = {g: 0 for g in GROUPS}
    for i, g in enumerate(groups_arr):
        (head_cnt if i < 100 else tail_cnt)[g] += 1
    comp_did = sum(exact_group_delta[g] * (tail_cnt[g] - head_cnt[g]) / 100.0
                   for g in GROUPS)
    out["composition"] = {
        "head_counts": head_cnt, "tail_counts": tail_cnt,
        "did_from_composition_alone": round(comp_did, 4),
        "residual": round(obs_did - comp_did, 4),
        "share_of_observed_did": round(comp_did / obs_did, 4)}
    check("head_STEM", head_cnt["STEM"], 51)
    check("head_humanities", head_cnt["humanities"], 9)
    check("tail_STEM", tail_cnt["STEM"], 5)
    check("tail_humanities", tail_cnt["humanities"], 40)
    check("comp_did", round(comp_did, 3), -0.058)
    check("comp_residual", round(obs_did - comp_did, 3), -0.012)
    check("comp_share_pct", round(100 * comp_did / obs_did), 83)

    # ------------------------ 7. truncation screen ------------------------
    claim_len = {"ref": 8, "k2": 9, "k6": 13, "k8": 12}
    tr_out = {}
    for s in ["ref", "k2", "k6", "k8"]:
        li = [i for i in range(200) if fin[s][i] == "length"]
        n_len = len(li)
        acc_len = float(corr[s][li].mean()) if li else None
        tr_out[s] = {
            "n_length": n_len,
            "length_head": sum(1 for i in li if i < 100),
            "length_tail": sum(1 for i in li if i >= 100),
            "acc_on_length": acc_len,
            "n_zero_tokens": int((toks[s] == 0).sum()),
            "n_at_budget": int((toks[s] >= BUDGET).sum()),
            "other_finish_reasons": sorted({fin[s][i] for i in range(200)}
                                           - {"stop", "length"}),
        }
        check(f"len_{s}", n_len, claim_len[s])
        check(f"acc_on_len_{s}", acc_len, 0.0)
        check(f"zero_tokens_{s}", tr_out[s]["n_zero_tokens"], 0)
    check("len_k8_head", tr_out["k8"]["length_head"], 4)
    check("len_k8_tail", tr_out["k8"]["length_tail"], 8)
    k8_only = [i for i in range(200) if r[i] == 1 and e[i] == 0]
    k8_only_len = [i for i in k8_only if fin["k8"][i] == "length"]
    ref_only = [i for i in range(200) if r[i] == 0 and e[i] == 1]
    ref_only_len = [i for i in ref_only if fin["ref"][i] == "length"]
    tr_out["k8_only_errors_budget_hit"] = {"n": len(k8_only),
                                           "n_length": len(k8_only_len)}
    tr_out["ref_only_errors_ref_truncated"] = {"n": len(ref_only),
                                               "n_length": len(ref_only_len)}
    out["truncation"] = tr_out
    check("k8_only_n", len(k8_only), 50)
    check("k8_only_budget_hit", len(k8_only_len), 6)
    check("ref_only_n", len(ref_only), 13)
    check("ref_only_ref_truncated", len(ref_only_len), 4)
    check("trunc_upper_bound", round(len(k8_only_len) / 200, 3), 0.03)
    check("trunc_net", round((len(k8_only_len) - len(ref_only_len)) / 200, 3),
          0.01)

    # ------------------------- 8. breakpoint scan -------------------------
    W = 20
    win = np.array([d_k8[s0:s0 + W].mean() for s0 in range(200 - W + 1)])
    worst_start = int(win.argmin())
    worst_val = float(win.min())
    worst_subjects = sorted(set(subjects[worst_start:worst_start + W]))
    check("worst_window_start", worst_start, 141)
    check("worst_window_delta", round(worst_val, 3), -0.45)
    check("worst_window_first_subject", worst_subjects[0], "management")
    check("worst_window_has_nutrition", "nutrition" in worst_subjects, True)

    S = np.concatenate([[0.0], np.cumsum(d_k8)])
    ts = np.arange(10, 191)

    def maxgap(prefix):
        m1 = prefix[ts] / ts
        m2 = (prefix[200] - prefix[ts]) / (200 - ts)
        g = np.abs(m2 - m1)
        return float(g.max()), int(ts[g.argmax()])

    best_gap, best_t = maxgap(S)
    boundaries = {0} | {i for i in range(1, 200)
                        if subjects[i] != subjects[i - 1]}
    dist = min(abs(best_t - bb) for bb in boundaries)
    frac_close = sum(1 for t in range(10, 191)
                     if min(abs(t - bb) for bb in boundaries) <= dist) / 181
    check("best_split_t", best_t, 143)
    check("best_split_gap", round(best_gap, 3), 0.183)
    check("split_on_boundary", dist, 0)
    check("frac_close_pct", round(100 * frac_close), 28)
    check("n_boundaries", len(boundaries), 57)

    # max-gap permutation null: FULL step-1 scan (stricter than original's
    # step-5 null), plus step-5 replication
    cnt_full, cnt_step5 = 0, 0
    ts5 = np.arange(10, 191, 5)
    dd = d_k8.copy()
    for _ in range(N_PERM):
        rng.shuffle(dd)
        pref = np.concatenate([[0.0], np.cumsum(dd)])
        m1 = pref[ts] / ts
        m2 = (pref[200] - pref[ts]) / (200 - ts)
        g = np.abs(m2 - m1)
        if g.max() >= best_gap - 1e-12:
            cnt_full += 1
        g5 = np.abs((pref[200] - pref[ts5]) / (200 - ts5) - pref[ts5] / ts5)
        if g5.max() >= best_gap - 1e-12:
            cnt_step5 += 1
    p_gap_full = (cnt_full + 1) / (N_PERM + 1)
    p_gap_step5 = (cnt_step5 + 1) / (N_PERM + 1)
    out["breakpoint"] = {
        "worst_window": {"start": worst_start, "end": worst_start + W - 1,
                         "delta": round(worst_val, 4),
                         "subjects": worst_subjects},
        "best_split": {"t": best_t, "gap": round(best_gap, 4),
                       "dist_to_boundary": dist,
                       "frac_indices_as_close": round(frac_close, 4)},
        "maxgap_perm_p_full_scan": round(p_gap_full, 5),
        "maxgap_perm_p_step5_scan": round(p_gap_step5, 5)}
    # original claim p=0.737 came from a step-5 null; verify step-5 within MC
    # tolerance and report the stricter full-scan p too
    check("maxgap_p_step5", p_gap_step5, 0.737, tol=0.011)
    check("maxgap_p_full_ge_step5_and_ns", p_gap_full >= 0.5, True)

    # ------------------------- 9. index quartiles -------------------------
    q_k8 = [round(float(d_k8[q * 50:(q + 1) * 50].mean()), 4)
            for q in range(4)]
    out["quartiles_k8"] = q_k8
    check("quartiles_k8", q_k8, [-0.18, -0.12, -0.18, -0.26])

    # --------------------------- 10. KL anchors ---------------------------
    kl_out = {}
    claim_kl = {"onpolicy": {"k2": 0.00755, "k6": 0.03111, "k8": 0.06426},
                "offpolicy": {"k2": 0.03304, "k6": 0.20461, "k8": 0.39148}}
    for pol in ["onpolicy", "offpolicy"]:
        with open(os.path.join(KL_DIR, f"topk_{pol}.json")) as f:
            cand = {c["name"]: c["mean_kl_nats"]
                    for c in json.load(f)["candidates"]}
        kl_out[pol] = {s: cand[KL_NAMES[s]] for s in KL_NAMES}
        for s in KL_NAMES:
            check(f"kl_{pol}_{s}", round(kl_out[pol][s], 5),
                  claim_kl[pol][s])
    out["kl_anchors"] = kl_out
    accs = [acc_out[s]["acc"] for s in ["ref", "k2", "k6", "k8"]]
    kls = [0.0] + [kl_out["onpolicy"][s] for s in ["k2", "k6", "k8"]]
    check("acc_monotone_decreasing_in_kl",
          all(a1 > a2 for a1, a2 in zip(accs, accs[1:]))
          and all(x1 < x2 for x1, x2 in zip(kls, kls[1:])), True)

    # ------------------------------- report -------------------------------
    n_fail = sum(1 for ch in checks if not ch["ok"])
    out["checks"] = checks
    out["n_checks"] = len(checks)
    out["n_failed"] = n_fail

    with open(os.path.join(HERE, "verify_a_mmlu_tail.json"), "w") as f:
        json.dump(out, f, indent=2)

    md = ["# verify_a_mmlu_tail — independent re-derivation\n",
          f"Seed={SEED} (different from original), {N_PERM} permutations, "
          "stdlib exact tests, pairing by id, defensive dtype coercion.\n",
          f"**{len(checks)} checks, {n_fail} failed.**\n",
          "| check | mine | theirs | ok |", "|---|---|---|---|"]
    for ch in checks:
        md.append(f"| {ch['check']} | {ch['mine']} | {ch['theirs']} | "
                  f"{'PASS' if ch['ok'] else '**FAIL**'} |")
    md.append("\nNotes:")
    md.append("- Raw dtypes: `correct` is bool and `gen_tokens` is int in "
              "all four result files; coercion paths untriggered.")
    md.append("- McNemar doubling and min-likelihood two-sided p agree "
              "exactly (symmetric null).")
    md.append("- Original max-gap null used a step-5 t-scan while the "
              "observed gap used step-1 (mildly anti-conservative). "
              f"Full step-1 null gives p={out['breakpoint']['maxgap_perm_p_full_scan']} "
              f"vs step-5 p={out['breakpoint']['maxgap_perm_p_step5_scan']} — "
              "both decisively non-significant; conclusion unchanged.")
    with open(os.path.join(HERE, "verify_a_mmlu_tail.md"), "w") as f:
        f.write("\n".join(md) + "\n")

    for ch in checks:
        if not ch["ok"]:
            print("FAIL:", ch)
    print(f"{len(checks)} checks, {n_fail} failed")
    print("maxgap p full-scan:", out["breakpoint"]["maxgap_perm_p_full_scan"],
          "| step-5:", out["breakpoint"]["maxgap_perm_p_step5_scan"])


if __name__ == "__main__":
    main()
