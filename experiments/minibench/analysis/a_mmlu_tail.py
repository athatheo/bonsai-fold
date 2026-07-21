#!/usr/bin/env python3
"""a_mmlu_tail: explain k8's MMLU drop (.620 vs reference .805, N=200).

Pure-JSON, CPU-only, deterministic (seed=0). No model loading.

Analyses (k8 vs reference primary; k2/k6 for trend):
  1. Per-subject paired accuracy deltas + MMLU-category group aggregation.
  2. Positional head(0-99) vs tail(100-199) paired difference-in-differences,
     permutation test (fixed seed).
  3. McNemar exact (two-sided binomial on discordant pairs) k8 vs ref on MMLU.
  4. Truncation/anomaly screen: finish_reason census, gen_tokens distribution,
     budget hits head vs tail, truncation-attributable k8-only errors.
  5. Breakpoint scan: sliding-window (w=20) paired delta over MMLU index;
     best two-segment breakpoint and alignment with subject boundaries.

Outputs: a_mmlu_tail.json (machine-readable), a_mmlu_tail.md (summary).
"""

import json
import math
import os
import random
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")
ITEMS_PATH = os.path.join(HERE, "..", "minibench_items.json")
KL_DIR = os.path.join(HERE, "..", "..", "kl_screen", "results")

CONFIGS = ["reference", "k2_16-12", "k6", "k8"]
SHORT = {"reference": "ref", "k2_16-12": "k2", "k6": "k6", "k8": "k8"}
KL_NAME = {
    "k2_16-12": "drop[16,12]",
    "k6": "drop[16,12,13,9,8,4]",
    "k8": "drop[16,12,13,9,8,4,5,15]",
}
MAX_TOKENS = 16384
SEED = 0
N_PERM = 20000

# Official MMLU category mapping (Hendrycks et al. categories.py):
# subject -> subcategory -> one of 4 categories.
SUBJECT_GROUP = {
    "abstract_algebra": "STEM", "anatomy": "other", "astronomy": "STEM",
    "business_ethics": "other", "clinical_knowledge": "other",
    "college_biology": "STEM", "college_chemistry": "STEM",
    "college_computer_science": "STEM", "college_mathematics": "STEM",
    "college_medicine": "other", "college_physics": "STEM",
    "computer_security": "STEM", "conceptual_physics": "STEM",
    "econometrics": "social_sciences", "electrical_engineering": "STEM",
    "elementary_mathematics": "STEM", "formal_logic": "humanities",
    "global_facts": "other", "high_school_biology": "STEM",
    "high_school_chemistry": "STEM", "high_school_computer_science": "STEM",
    "high_school_european_history": "humanities",
    "high_school_geography": "social_sciences",
    "high_school_government_and_politics": "social_sciences",
    "high_school_macroeconomics": "social_sciences",
    "high_school_mathematics": "STEM",
    "high_school_microeconomics": "social_sciences",
    "high_school_physics": "STEM", "high_school_psychology": "social_sciences",
    "high_school_statistics": "STEM", "high_school_us_history": "humanities",
    "high_school_world_history": "humanities", "human_aging": "other",
    "human_sexuality": "social_sciences", "international_law": "humanities",
    "jurisprudence": "humanities", "logical_fallacies": "humanities",
    "machine_learning": "STEM", "management": "other", "marketing": "other",
    "medical_genetics": "other", "miscellaneous": "other",
    "moral_disputes": "humanities", "moral_scenarios": "humanities",
    "nutrition": "other", "philosophy": "humanities", "prehistory": "humanities",
    "professional_accounting": "other", "professional_law": "humanities",
    "professional_medicine": "other", "professional_psychology": "social_sciences",
    "public_relations": "social_sciences", "security_studies": "social_sciences",
    "sociology": "social_sciences", "us_foreign_policy": "social_sciences",
    "virology": "other", "world_religions": "humanities",
}


# ----------------------------- stats helpers ------------------------------

def wilson_ci(k, n, z=1.959963984540054):
    """Wilson score 95% CI for a binomial proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))


def binom_cdf(k, n):
    """P(X <= k), X ~ Binomial(n, 0.5). Exact via math.comb."""
    if k < 0:
        return 0.0
    total = sum(math.comb(n, i) for i in range(0, k + 1))
    return total / (2 ** n)


def mcnemar_exact(b, c):
    """Two-sided exact McNemar: binomial(b+c, 0.5) doubling method.

    b = config A correct & config B wrong; c = A wrong & B correct.
    Returns dict with discordant counts and exact two-sided p.
    """
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "n_discordant": 0, "p_exact_two_sided": 1.0}
    k = min(b, c)
    p = min(1.0, 2.0 * binom_cdf(k, n))
    return {"b": b, "c": c, "n_discordant": n, "p_exact_two_sided": p}


def summarize_tokens(vals):
    s = sorted(vals)
    n = len(s)

    def q(p):
        return s[min(n - 1, int(p * n))]

    return {
        "n": n,
        "mean": round(statistics.fmean(s), 1),
        "median": s[n // 2],
        "p90": q(0.90),
        "min": s[0],
        "max": s[-1],
    }


# ------------------------------- load data --------------------------------

def load():
    runs = {}
    for cfg in CONFIGS:
        with open(os.path.join(RESULTS, f"{cfg}.json")) as f:
            runs[cfg] = json.load(f)

    id_sets = {cfg: {it["id"] for it in runs[cfg]["items"]} for cfg in CONFIGS}
    ref_ids = id_sets["reference"]
    for cfg in CONFIGS:
        assert id_sets[cfg] == ref_ids, f"id set mismatch: {cfg}"
        assert len(runs[cfg]["items"]) == 500, f"expected 500 items in {cfg}"
        for it in runs[cfg]["items"]:
            assert isinstance(it["correct"], bool), (cfg, it["id"], it["correct"])
            assert isinstance(it["gen_tokens"], int), (cfg, it["id"])

    by_id = {cfg: {it["id"]: it for it in runs[cfg]["items"]} for cfg in CONFIGS}

    with open(ITEMS_PATH) as f:
        frozen = json.load(f)["items"]
    mmlu_meta = [it for it in frozen if it["task"] == "mmlu"]
    assert len(mmlu_meta) == 200
    # frozen order == id order (mmlu-000..199); assert and parse subjects
    mmlu_ids = [it["id"] for it in mmlu_meta]
    assert mmlu_ids == [f"mmlu-{i:03d}" for i in range(200)]
    subjects = [it["source"].split("[")[0] for it in mmlu_meta]
    assert subjects == sorted(subjects), "MMLU items not alphabetical by subject"
    for s in subjects:
        assert s in SUBJECT_GROUP, f"unmapped subject {s}"

    kl = {}
    for pol in ["onpolicy", "offpolicy"]:
        with open(os.path.join(KL_DIR, f"topk_{pol}.json")) as f:
            d = json.load(f)
        kl[pol] = {c["name"]: c["mean_kl_nats"] for c in d["candidates"]}
    return runs, by_id, mmlu_ids, subjects, kl


def main():
    runs, by_id, mmlu_ids, subjects, kl = load()

    # correctness vectors over mmlu index 0..199, per config
    corr = {
        SHORT[cfg]: [int(by_id[cfg][i]["correct"]) for i in mmlu_ids]
        for cfg in CONFIGS
    }
    fin = {
        SHORT[cfg]: [by_id[cfg][i]["finish_reason"] for i in mmlu_ids]
        for cfg in CONFIGS
    }
    toks = {
        SHORT[cfg]: [by_id[cfg][i]["gen_tokens"] for i in mmlu_ids]
        for cfg in CONFIGS
    }

    out = {"meta": {
        "task": "a_mmlu_tail",
        "n_mmlu": 200,
        "seed": SEED,
        "n_permutations": N_PERM,
        "kl_anchors_mean_kl_nats": {
            "onpolicy": {SHORT[c]: kl["onpolicy"][KL_NAME[c]] for c in KL_NAME},
            "offpolicy": {SHORT[c]: kl["offpolicy"][KL_NAME[c]] for c in KL_NAME},
        },
        "mmlu_accuracy": {},
    }}
    for c in ["ref", "k2", "k6", "k8"]:
        k = sum(corr[c])
        lo, hi = wilson_ci(k, 200)
        out["meta"]["mmlu_accuracy"][c] = {
            "acc": k / 200, "n_correct": k,
            "wilson95": [round(lo, 4), round(hi, 4)],
        }

    # ---------- 1. per-subject paired deltas + group aggregation ----------
    subj_rows = []
    uniq_subjects = sorted(set(subjects))
    for s in uniq_subjects:
        idx = [i for i, x in enumerate(subjects) if x == s]
        n = len(idx)
        row = {"subject": s, "group": SUBJECT_GROUP[s], "n": n,
               "index_range": [idx[0], idx[-1]]}
        for c in ["ref", "k2", "k6", "k8"]:
            row[f"acc_{c}"] = round(sum(corr[c][i] for i in idx) / n, 4)
        row["delta_k8_ref"] = round(row["acc_k8"] - row["acc_ref"], 4)
        row["underpowered"] = n < 20  # all subjects are; keep the flag honest
        subj_rows.append(row)
    subj_rows.sort(key=lambda r: r["delta_k8_ref"])
    out["per_subject"] = subj_rows

    groups = ["STEM", "humanities", "social_sciences", "other"]
    group_rows = []
    for g in groups:
        idx = [i for i, s in enumerate(subjects) if SUBJECT_GROUP[s] == g]
        n = len(idx)
        row = {"group": g, "n": n}
        for c in ["ref", "k2", "k6", "k8"]:
            k = sum(corr[c][i] for i in idx)
            lo, hi = wilson_ci(k, n)
            row[f"acc_{c}"] = round(k / n, 4)
            row[f"wilson95_{c}"] = [round(lo, 4), round(hi, 4)]
        b = sum(1 for i in idx if corr["ref"][i] and not corr["k8"][i])
        cc = sum(1 for i in idx if not corr["ref"][i] and corr["k8"][i])
        mc = mcnemar_exact(b, cc)
        row["delta_k8_ref"] = round(row["acc_k8"] - row["acc_ref"], 4)
        row["mcnemar_k8_ref"] = mc
        row["underpowered"] = n < 60
        group_rows.append(row)
    out["per_group"] = group_rows

    # ---------- 2. positional head/tail paired diff-in-diffs ----------
    d_k8 = [corr["k8"][i] - corr["ref"][i] for i in range(200)]  # in {-1,0,1}
    head, tail = list(range(100)), list(range(100, 200))
    positional = {}
    for cname in ["k2", "k6", "k8"]:
        d = [corr[cname][i] - corr["ref"][i] for i in range(200)]
        hd = statistics.fmean(d[i] for i in head)
        td = statistics.fmean(d[i] for i in tail)
        positional[cname] = {
            "head_acc_ref": round(statistics.fmean(corr["ref"][i] for i in head), 4),
            "tail_acc_ref": round(statistics.fmean(corr["ref"][i] for i in tail), 4),
            "head_acc": round(statistics.fmean(corr[cname][i] for i in head), 4),
            "tail_acc": round(statistics.fmean(corr[cname][i] for i in tail), 4),
            "head_paired_delta": round(hd, 4),
            "tail_paired_delta": round(td, 4),
            "diff_in_diffs_tail_minus_head": round(td - hd, 4),
        }
    # permutation test on k8 DiD: shuffle per-item paired deltas across positions
    rng = random.Random(SEED)
    obs = positional["k8"]["diff_in_diffs_tail_minus_head"]
    d = list(d_k8)
    count = 0
    for _ in range(N_PERM):
        rng.shuffle(d)
        did = statistics.fmean(d[100:]) - statistics.fmean(d[:100])
        if abs(did) >= abs(obs) - 1e-12:
            count += 1
    p_perm = (count + 1) / (N_PERM + 1)
    positional["k8"]["permutation_p_two_sided"] = round(p_perm, 5)
    # head/tail subject-group composition (they differ by construction)
    comp = {}
    for half, idx in [("head", head), ("tail", tail)]:
        cnt = {}
        for i in idx:
            g = SUBJECT_GROUP[subjects[i]]
            cnt[g] = cnt.get(g, 0) + 1
        comp[half] = cnt
    positional["head_tail_group_composition"] = comp
    out["positional"] = positional

    # ---------- 3. McNemar exact k8 vs ref on all 200 MMLU items ----------
    b = sum(1 for i in range(200) if corr["ref"][i] and not corr["k8"][i])
    c = sum(1 for i in range(200) if not corr["ref"][i] and corr["k8"][i])
    both_wrong = sum(1 for i in range(200) if not corr["ref"][i] and not corr["k8"][i])
    both_right = sum(1 for i in range(200) if corr["ref"][i] and corr["k8"][i])
    mc = mcnemar_exact(b, c)
    mc.update({
        "both_right": both_right, "both_wrong": both_wrong,
        "net_flips": b - c,
        "delta_acc": round((c - b) / 200, 4),
        "flip_rate_ref_correct_to_k8_wrong": round(b / max(1, b + both_right), 4),
    })
    out["mcnemar_k8_vs_ref_mmlu"] = mc

    # ---------- 4. truncation / anomaly screen ----------
    trunc = {}
    for cname in ["ref", "k2", "k6", "k8"]:
        n_len = sum(1 for f in fin[cname] if f == "length")
        n_len_head = sum(1 for i in head if fin[cname][i] == "length")
        n_len_tail = sum(1 for i in tail if fin[cname][i] == "length")
        len_idx = [i for i in range(200) if fin[cname][i] == "length"]
        acc_len = (statistics.fmean(corr[cname][i] for i in len_idx)
                   if len_idx else None)
        weird = [f for f in fin[cname] if f not in ("stop", "length")]
        trunc[cname] = {
            "finish_reason_counts": {
                "stop": 200 - n_len - len(weird), "length": n_len,
                "other": len(weird),
            },
            "length_head": n_len_head, "length_tail": n_len_tail,
            "acc_on_length_items": (round(acc_len, 4)
                                    if acc_len is not None else None),
            "gen_tokens": summarize_tokens(toks[cname]),
            "n_gen_tokens_zero": sum(1 for t in toks[cname] if t == 0),
            "n_gen_tokens_at_budget": sum(1 for t in toks[cname]
                                          if t >= MAX_TOKENS),
        }
    # k8-only errors attributable to truncation
    k8_only_wrong = [i for i in range(200)
                     if corr["ref"][i] and not corr["k8"][i]]
    k8_only_wrong_len = [i for i in k8_only_wrong if fin["k8"][i] == "length"]
    trunc["k8_only_errors"] = {
        "n": len(k8_only_wrong),
        "n_length_truncated_in_k8": len(k8_only_wrong_len),
        "indices_length_truncated": k8_only_wrong_len,
        "n_in_tail": sum(1 for i in k8_only_wrong if i >= 100),
    }
    # note: raw answer text is not stored in results; empty-answer check is
    # limited to gen_tokens==0 / finish_reason proxies.
    trunc["note"] = ("results JSON stores no answer text; missing/empty-answer "
                     "screen uses gen_tokens==0 and finish_reason proxies only")
    out["truncation_screen"] = trunc

    # ---------- 5. breakpoint scan ----------
    W = 20
    windows = []
    for s0 in range(0, 200 - W + 1):
        wd = statistics.fmean(d_k8[s0:s0 + W])
        windows.append({"start": s0, "end": s0 + W - 1,
                        "paired_delta": round(wd, 4)})
    worst = min(windows, key=lambda w: w["paired_delta"])
    subject_boundaries = [0] + [i for i in range(1, 200)
                                if subjects[i] != subjects[i - 1]]
    # best two-segment split of the paired-delta sequence (max |mean diff|)
    best_t, best_gap = None, -1.0
    for t in range(10, 191):
        m1 = statistics.fmean(d_k8[:t])
        m2 = statistics.fmean(d_k8[t:])
        if abs(m2 - m1) > best_gap:
            best_gap, best_t = abs(m2 - m1), t
    m1 = statistics.fmean(d_k8[:best_t])
    m2 = statistics.fmean(d_k8[best_t:])
    dist_to_boundary = min(abs(best_t - bnd) for bnd in subject_boundaries)
    # null: how unusual is being that close to a boundary? (57 boundaries/200)
    close_frac = statistics.fmean(
        1 if min(abs(t - bnd) for bnd in subject_boundaries) <= dist_to_boundary
        else 0
        for t in range(10, 191))
    # permutation significance of the best split gap (same shuffle null)
    rng2 = random.Random(SEED + 1)
    d2 = list(d_k8)
    cnt_gap = 0
    for _ in range(N_PERM):
        rng2.shuffle(d2)
        g = 0.0
        # coarse scan (step 5) is enough for a null of max-gap magnitude;
        # keep exact scan for observed value above
        for t in range(10, 191, 5):
            gg = abs(statistics.fmean(d2[t:]) - statistics.fmean(d2[:t]))
            if gg > g:
                g = gg
        if g >= best_gap - 1e-12:
            cnt_gap += 1
    p_gap = (cnt_gap + 1) / (N_PERM + 1)
    out["breakpoint_scan"] = {
        "window_w": W,
        "worst_window": {
            **worst,
            "subjects_in_window": sorted(set(subjects[worst["start"]:
                                                      worst["end"] + 1])),
        },
        "best_windows_sorted": sorted(windows,
                                      key=lambda w: w["paired_delta"])[:5],
        "n_subject_boundaries": len(subject_boundaries),
        "best_two_segment_split": {
            "t": best_t,
            "mean_paired_delta_before": round(m1, 4),
            "mean_paired_delta_after": round(m2, 4),
            "gap": round(best_gap, 4),
            "subject_at_t": subjects[best_t],
            "subject_before_t": subjects[best_t - 1],
            "distance_to_nearest_subject_boundary": dist_to_boundary,
            "fraction_of_indices_at_least_this_close_to_a_boundary":
                round(close_frac, 4),
            "max_gap_permutation_p": round(p_gap, 5),
        },
    }

    # also: trend of paired delta by index-quartile for k2/k6/k8
    quart = {}
    for cname in ["k2", "k6", "k8"]:
        d = [corr[cname][i] - corr["ref"][i] for i in range(200)]
        quart[cname] = [round(statistics.fmean(d[q * 50:(q + 1) * 50]), 4)
                        for q in range(4)]
    out["paired_delta_by_index_quartile"] = quart

    # ---------- 6. decomposition: composition vs subject-specific ----------
    # (a) How much of the observed tail-head DiD is explained purely by the
    # head/tail group composition, given each group's overall paired delta?
    group_delta = {r["group"]: r["acc_k8"] - r["acc_ref"]
                   for r in out["per_group"]}
    comp_did = sum(
        group_delta[g] * ((comp["tail"].get(g, 0) - comp["head"].get(g, 0))
                          / 100.0)
        for g in groups)
    out["composition_decomposition"] = {
        "observed_did_tail_minus_head": obs,
        "did_expected_from_group_composition_alone": round(comp_did, 4),
        "residual_positional_did": round(obs - comp_did, 4),
    }
    # (b) humanities vs rest: is k8's damage concentrated there beyond chance?
    hum_idx = [i for i in range(200)
               if SUBJECT_GROUP[subjects[i]] == "humanities"]
    rest_idx = [i for i in range(200) if i not in set(hum_idx)]
    d_hum = statistics.fmean(d_k8[i] for i in hum_idx)
    d_rest = statistics.fmean(d_k8[i] for i in rest_idx)
    obs_hr = d_hum - d_rest
    rng3 = random.Random(SEED + 2)
    d3 = list(d_k8)
    n_h = len(hum_idx)
    cnt_hr = 0
    for _ in range(N_PERM):
        rng3.shuffle(d3)
        hr = statistics.fmean(d3[:n_h]) - statistics.fmean(d3[n_h:])
        if abs(hr) >= abs(obs_hr) - 1e-12:
            cnt_hr += 1
    p_hr = (cnt_hr + 1) / (N_PERM + 1)
    # c-side sanity: how many of the 13 ref-wrong/k8-right items were
    # ref budget truncations?
    c_side = [i for i in range(200) if not corr["ref"][i] and corr["k8"][i]]
    out["humanities_vs_rest_k8"] = {
        "n_humanities": n_h,
        "paired_delta_humanities": round(d_hum, 4),
        "paired_delta_rest": round(d_rest, 4),
        "difference": round(obs_hr, 4),
        "permutation_p_two_sided": round(p_hr, 5),
        "humanities_in_tail": comp["tail"].get("humanities", 0),
        "humanities_in_head": comp["head"].get("humanities", 0),
    }
    out["truncation_screen"]["ref_only_errors_recovered_by_k8"] = {
        "n": len(c_side),
        "n_ref_length_truncated": sum(1 for i in c_side
                                      if fin["ref"][i] == "length"),
    }

    # ------------------------------ write out ------------------------------
    json_path = os.path.join(HERE, "a_mmlu_tail.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)

    md = []
    md.append("# a_mmlu_tail — why did k8's MMLU fall to .620?\n")
    md.append("k8 = drop[16,12,13,9,8,4,5,15]; paired vs reference on the "
              "frozen 200-item MMLU slice (alphabetical-by-subject order). "
              f"Seed={SEED}, {N_PERM} permutations, exact stdlib tests "
              "(no scipy).\n")
    acc = out["meta"]["mmlu_accuracy"]
    md.append("## MMLU accuracy (Wilson 95% CI)\n")
    md.append("| config | acc | 95% CI | on-policy KL (nats) |")
    md.append("|---|---|---|---|")
    klmap = {"ref": 0.0, **out["meta"]["kl_anchors_mean_kl_nats"]["onpolicy"]}
    for c in ["ref", "k2", "k6", "k8"]:
        a = acc[c]
        md.append(f"| {c} | {a['acc']:.3f} | [{a['wilson95'][0]:.3f}, "
                  f"{a['wilson95'][1]:.3f}] | {klmap[c]:.4f} |")
    mcn = out["mcnemar_k8_vs_ref_mmlu"]
    md.append("\n## McNemar exact, k8 vs ref (MMLU, N=200)\n")
    md.append(f"- both right {mcn['both_right']}, both wrong "
              f"{mcn['both_wrong']}, ref-right/k8-wrong b={mcn['b']}, "
              f"ref-wrong/k8-right c={mcn['c']} (net {mcn['net_flips']} "
              f"flips, delta={mcn['delta_acc']:+.3f})")
    md.append(f"- exact two-sided p = {mcn['p_exact_two_sided']:.2e} — "
              "the drop is not chance.")
    md.append(f"- {mcn['flip_rate_ref_correct_to_k8_wrong']:.1%} of items ref "
              "got right are lost by k8.")
    md.append("\n## Subject groups (paired; official MMLU categories)\n")
    md.append("| group | n | ref | k2 | k6 | k8 | Δ(k8−ref) | McNemar p | "
              "power |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for r in out["per_group"]:
        md.append(f"| {r['group']} | {r['n']} | {r['acc_ref']:.3f} | "
                  f"{r['acc_k2']:.3f} | {r['acc_k6']:.3f} | "
                  f"{r['acc_k8']:.3f} | {r['delta_k8_ref']:+.3f} | "
                  f"{r['mcnemar_k8_ref']['p_exact_two_sided']:.3f} | "
                  f"{'LOW' if r['underpowered'] else 'ok'} |")
    md.append("\nAll 57 per-subject cells have n<=9 — individually "
              "uninterpretable; see JSON `per_subject` for the full table. "
              "Worst subjects by paired delta (n in parens): " + ", ".join(
                  f"{r['subject']} {r['delta_k8_ref']:+.2f} (n={r['n']})"
                  for r in out["per_subject"][:6]) + ".\n")
    md.append("## Positional head (0–99) vs tail (100–199), paired\n")
    md.append("| config | head Δ | tail Δ | DiD (tail−head) |")
    md.append("|---|---|---|---|")
    for c in ["k2", "k6", "k8"]:
        p = out["positional"][c]
        md.append(f"| {c} | {p['head_paired_delta']:+.3f} | "
                  f"{p['tail_paired_delta']:+.3f} | "
                  f"{p['diff_in_diffs_tail_minus_head']:+.3f} |")
    pk8 = out["positional"]["k8"]
    md.append(f"\nk8 DiD permutation p (two-sided, {N_PERM} shuffles) = "
              f"{pk8['permutation_p_two_sided']:.3f}.")
    md.append(f"Head/tail composition differs by construction: "
              f"{out['positional']['head_tail_group_composition']}.\n")
    md.append("## Truncation / anomaly screen (MMLU)\n")
    md.append("| config | stop | length | len head/tail | acc on length "
              "items | gen_tokens mean/med/p90 | zero-token |")
    md.append("|---|---|---|---|---|---|---|")
    for c in ["ref", "k2", "k6", "k8"]:
        t = out["truncation_screen"][c]
        g = t["gen_tokens"]
        md.append(f"| {c} | {t['finish_reason_counts']['stop']} | "
                  f"{t['finish_reason_counts']['length']} | "
                  f"{t['length_head']}/{t['length_tail']} | "
                  f"{t['acc_on_length_items']} | "
                  f"{g['mean']}/{g['median']}/{g['p90']} | "
                  f"{t['n_gen_tokens_zero']} |")
    ko = out["truncation_screen"]["k8_only_errors"]
    md.append(f"\nk8-only errors: {ko['n']}; of those, "
              f"{ko['n_length_truncated_in_k8']} hit the 16384 budget in k8; "
              f"{ko['n_in_tail']} are in the tail half.\n")
    bp = out["breakpoint_scan"]
    md.append("## Breakpoint scan (w=20 sliding paired delta, k8−ref)\n")
    ww = bp["worst_window"]
    md.append(f"- Worst window: items {ww['start']}–{ww['end']} "
              f"(paired Δ={ww['paired_delta']:+.3f}); subjects: "
              f"{', '.join(ww['subjects_in_window'])}.")
    b2 = bp["best_two_segment_split"]
    md.append(f"- Best two-segment split at t={b2['t']} "
              f"(…{b2['subject_before_t']} | {b2['subject_at_t']}…): "
              f"Δ={b2['mean_paired_delta_before']:+.3f} before vs "
              f"{b2['mean_paired_delta_after']:+.3f} after "
              f"(gap {b2['gap']:.3f}, max-gap permutation p="
              f"{b2['max_gap_permutation_p']:.3f}).")
    md.append(f"- Split sits {b2['distance_to_nearest_subject_boundary']} "
              f"items from a subject boundary; "
              f"{b2['fraction_of_indices_at_least_this_close_to_a_boundary']:.0%}"
              " of all candidate indices are at least that close to one of "
              f"the {bp['n_subject_boundaries']} boundaries, so alignment is "
              "uninformative.")
    md.append(f"- Paired delta by index quartile (k2/k6/k8): "
              f"{out['paired_delta_by_index_quartile']}.\n")
    md.append("## Decomposition: composition vs position vs subject damage\n")
    cd = out["composition_decomposition"]
    md.append(f"- Observed tail−head DiD {cd['observed_did_tail_minus_head']:+.3f}; "
              f"expected from head/tail subject-group composition alone "
              f"{cd['did_expected_from_group_composition_alone']:+.3f}; "
              f"residual positional component "
              f"{cd['residual_positional_did']:+.3f}.")
    hr = out["humanities_vs_rest_k8"]
    md.append(f"- Humanities (n={hr['n_humanities']}, "
              f"{hr['humanities_in_tail']}/49 in the tail): paired Δ "
              f"{hr['paired_delta_humanities']:+.3f} vs rest "
              f"{hr['paired_delta_rest']:+.3f}; difference "
              f"{hr['difference']:+.3f}, permutation p = "
              f"{hr['permutation_p_two_sided']:.4f}.")
    ro = out["truncation_screen"]["ref_only_errors_recovered_by_k8"]
    md.append(f"- Of the {ro['n']} ref-wrong/k8-right items, "
              f"{ro['n_ref_length_truncated']} were ref budget truncations.\n")
    md_path = os.path.join(HERE, "a_mmlu_tail.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md) + "\n")
    print("wrote", json_path)
    print("wrote", md_path)


if __name__ == "__main__":
    main()
