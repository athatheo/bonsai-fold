#!/usr/bin/env python
"""d_kl_fit: fit the KL -> benchmark-damage mapping from the 4 mini-bench anchors
and recalibrate the accept gate (replacing the provisional 0.05-nats on-policy gate).

CPU-only pure-JSON analysis. Deterministic (seed 20260721). stdlib + numpy only.

Inputs (read-only):
  experiments/minibench/results/{reference,k2_16-12,k6,k8}.json
  experiments/kl_screen/results/topk_{onpolicy,offpolicy}.json
  experiments/kl_screen/results/single_block_onpolicy.json
  experiments/minibench/minibench_items.json

Outputs:
  experiments/minibench/analysis/d_kl_fit.json
  experiments/minibench/analysis/d_kl_fit.md
"""
import json
import math
import os

import numpy as np

ROOT = "/Users/atheocharis/repos/bonsai-fold"
RES = os.path.join(ROOT, "experiments/minibench/results")
KLD = os.path.join(ROOT, "experiments/kl_screen/results")
OUT = os.path.join(ROOT, "experiments/minibench/analysis")

TASKS = ["gsm8k", "math500", "ifeval", "mmlu"]
CONFIGS = ["reference", "k2", "k6", "k8"]
CONFIG_FILES = {
    "reference": "reference.json",
    "k2": "k2_16-12.json",
    "k6": "k6.json",
    "k8": "k8.json",
}
SET_NAMES = {  # candidate names in topk files
    "k2": "drop[16,12]",
    "k4": "drop[16,12,13,9]",
    "k6": "drop[16,12,13,9,8,4]",
    "k8": "drop[16,12,13,9,8,4,5,15]",
}
SET_BLOCKS = {
    "k2": [16, 12],
    "k4": [16, 12, 13, 9],
    "k6": [16, 12, 13, 9, 8, 4],
    "k8": [16, 12, 13, 9, 8, 4, 5, 15],
}
BOOT_B = 10000
BOOT_SEED = 20260721
Z = 1.959963984540054  # 97.5% normal quantile


def as_bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "1", "yes"):
            return True
        if s in ("false", "0", "no"):
            return False
        raise ValueError(f"unparseable correct value: {v!r}")
    if isinstance(v, (int, float)):
        return bool(v)
    raise ValueError(f"unparseable correct value: {v!r}")


def wilson_ci(k, n, z=Z):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (center - half, center + half)


def mcnemar_exact(b, c):
    """Two-sided exact McNemar: b = n(A right, B wrong), c = n(A wrong, B right).
    Binomial(n=b+c, p=0.5) two-sided p via stdlib math.comb."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def load_results():
    data = {}
    for cfg, fname in CONFIG_FILES.items():
        with open(os.path.join(RES, fname)) as f:
            data[cfg] = json.load(f)
    # identical id sets across all four configs, in matched order after sorting
    id_sets = {cfg: sorted(i["id"] for i in d["items"]) for cfg, d in data.items()}
    ref_ids = id_sets["reference"]
    assert len(ref_ids) == 500 and len(set(ref_ids)) == 500
    for cfg in CONFIGS:
        assert id_sets[cfg] == ref_ids, f"id set mismatch: {cfg}"
    # frozen items file matches too
    with open(os.path.join(ROOT, "experiments/minibench/minibench_items.json")) as f:
        frozen = json.load(f)["items"]
    assert sorted(i["id"] for i in frozen) == ref_ids, "frozen item ids mismatch"
    mmlu_subject = {
        i["id"]: i["source"].split("[")[0] for i in frozen if i["task"] == "mmlu"
    }
    return data, ref_ids, mmlu_subject


def fit_forms(x, y):
    """x: KL nats (4,), y: macro drop in points (4,). Returns dict of fits."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    sx = np.sqrt(x)
    out = {}

    def summarize(pred, params, form):
        resid = y - pred
        ss_res = float(np.sum(resid ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        return {
            "form": form,
            "params": params,
            "pred_points": [round(float(v), 4) for v in pred],
            "residuals_points": [round(float(r), 4) for r in resid],
            "rmse_points": round(math.sqrt(ss_res / len(y)), 4),
            "r2_centered": round(1 - ss_res / ss_tot, 5),
        }

    b0 = float(np.sum(x * y) / np.sum(x * x))
    out["linear_origin"] = summarize(b0 * x, {"slope_pts_per_nat": round(b0, 3)},
                                     "drop = b*KL")
    A = np.vstack([np.ones_like(x), x]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    out["linear_intercept"] = summarize(
        A @ coef,
        {"intercept_pts": round(float(coef[0]), 4),
         "slope_pts_per_nat": round(float(coef[1]), 3)},
        "drop = a + b*KL")
    c0 = float(np.sum(sx * y) / np.sum(x))
    out["sqrt_origin"] = summarize(c0 * sx, {"coef_pts_per_sqrtnat": round(c0, 3)},
                                   "drop = c*sqrt(KL)")
    As = np.vstack([np.ones_like(sx), sx]).T
    coefs, *_ = np.linalg.lstsq(As, y, rcond=None)
    out["sqrt_intercept"] = summarize(
        As @ coefs,
        {"intercept_pts": round(float(coefs[0]), 4),
         "coef_pts_per_sqrtnat": round(float(coefs[1]), 3)},
        "drop = a + c*sqrt(KL)")
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    data, ref_ids, mmlu_subject = load_results()
    id_index = {i: n for n, i in enumerate(ref_ids)}
    n_items = len(ref_ids)

    # ---- aligned arrays ----
    task_of = np.empty(n_items, dtype=object)
    correct = {}   # cfg -> bool array aligned to ref_ids
    gentok = {}
    finlen = {}
    for cfg in CONFIGS:
        arr = np.zeros(n_items, dtype=bool)
        gt = np.zeros(n_items, dtype=np.int64)
        fl = np.zeros(n_items, dtype=bool)
        for it in data[cfg]["items"]:
            j = id_index[it["id"]]
            arr[j] = as_bool(it["correct"])
            gt[j] = int(it["gen_tokens"])
            fl[j] = (it["finish_reason"] == "length")
            task_of[j] = it["task"]
        correct[cfg] = arr
        gentok[cfg] = gt
        finlen[cfg] = fl
    task_masks = {t: (task_of == t) for t in TASKS}
    for t in TASKS:
        assert task_masks[t].sum() == (200 if t == "mmlu" else 100)

    # ---- recompute accuracies; cross-check stored values ----
    acc = {}
    for cfg in CONFIGS:
        ta = {t: float(correct[cfg][task_masks[t]].mean()) for t in TASKS}
        macro = float(np.mean([ta[t] for t in TASKS]))
        stored_ta = data[cfg]["task_accuracy"]
        for t in TASKS:
            assert abs(ta[t] - stored_ta[t]) < 1e-9, (cfg, t, ta[t], stored_ta[t])
        assert abs(macro - data[cfg]["macro_avg"]) < 1e-9
        acc[cfg] = {"task": ta, "macro": macro}

    # ---- KL anchors (actual values from files) ----
    def read_topk(fname):
        with open(os.path.join(KLD, fname)) as f:
            d = json.load(f)
        return {c["name"]: float(c["mean_kl_nats"]) for c in d["candidates"]}

    kl_on = read_topk("topk_onpolicy.json")
    kl_off = read_topk("topk_offpolicy.json")
    with open(os.path.join(KLD, "single_block_onpolicy.json")) as f:
        singles_on = {c["name"]: float(c["mean_kl_nats"])
                      for c in json.load(f)["candidates"]}

    kl_anchor = {"reference": 0.0}
    kl_anchor_off = {"reference": 0.0}
    for cfg in ("k2", "k6", "k8"):
        kl_anchor[cfg] = kl_on[SET_NAMES[cfg]]
        kl_anchor_off[cfg] = kl_off[SET_NAMES[cfg]]

    x = np.array([kl_anchor[c] for c in CONFIGS])           # on-policy KL, nats
    y_macro = np.array([(acc["reference"]["macro"] - acc[c]["macro"]) * 100
                        for c in CONFIGS])                  # drop, points

    # ---- off/on ratio and interaction tax from measured data ----
    ratios = {}
    for k in ("k2", "k4", "k6", "k8"):
        ratios[k] = kl_off[SET_NAMES[k]] / kl_on[SET_NAMES[k]]
    ratio_mean = float(np.mean(list(ratios.values())))
    ratio_pooled = (sum(kl_off[SET_NAMES[k]] for k in SET_NAMES)
                    / sum(kl_on[SET_NAMES[k]] for k in SET_NAMES))

    tax = {}
    for k, blocks in SET_BLOCKS.items():
        s = sum(singles_on[f"drop[{b}]"] for b in blocks)
        tax[k] = {"set_kl_on": kl_on[SET_NAMES[k]], "sum_singles_on": s,
                  "tax": kl_on[SET_NAMES[k]] / s}

    # ---- paired McNemar exact tests (each folded config vs reference) ----
    mcnemar = {}
    ref_c = correct["reference"]
    for cfg in ("k2", "k6", "k8"):
        cc = correct[cfg]
        entry = {}
        for scope, mask in [("all", np.ones(n_items, bool))] + list(task_masks.items()):
            b = int(np.sum(ref_c[mask] & ~cc[mask]))   # ref right, cfg wrong
            c = int(np.sum(~ref_c[mask] & cc[mask]))   # ref wrong, cfg right
            n = int(mask.sum())
            entry[scope] = {
                "n": n,
                "delta_acc_points": round((cc[mask].mean() - ref_c[mask].mean()) * 100, 3),
                "discordant_ref_right_cfg_wrong": b,
                "discordant_ref_wrong_cfg_right": c,
                "p_exact_two_sided": float(f"{mcnemar_exact(b, c):.3g}"),
            }
        mcnemar[f"{cfg}_vs_reference"] = entry
    # math500 non-monotonicity checks
    for a, bcfg in [("k6", "k8"), ("reference", "k8")]:
        m = task_masks["math500"]
        ca, cb = correct[a][m], correct[bcfg][m]
        bb = int(np.sum(ca & ~cb))
        cc_ = int(np.sum(~ca & cb))
        mcnemar[f"math500_{bcfg}_vs_{a}"] = {
            "n": int(m.sum()),
            "delta_acc_points": round((cb.mean() - ca.mean()) * 100, 3),
            "discordant_a_right_b_wrong": bb,
            "discordant_a_wrong_b_right": cc_,
            "p_exact_two_sided": float(f"{mcnemar_exact(bb, cc_):.3g}"),
        }

    # ---- Wilson CIs per task per config ----
    wilson = {}
    for cfg in CONFIGS:
        wilson[cfg] = {}
        for t in TASKS:
            m = task_masks[t]
            k = int(correct[cfg][m].sum())
            n = int(m.sum())
            lo, hi = wilson_ci(k, n)
            wilson[cfg][t] = {"acc": round(k / n, 4), "n": n,
                              "wilson95": [round(lo, 4), round(hi, 4)]}

    # ---- fits: macro and per-task ----
    fits_macro = fit_forms(x, y_macro)
    fits_task = {}
    for t in TASKS:
        yt = np.array([(acc["reference"]["task"][t] - acc[c]["task"][t]) * 100
                       for c in CONFIGS])
        fits_task[t] = {"drops_points": [round(float(v), 3) for v in yt],
                        **fit_forms(x, yt)}

    # ---- paired bootstrap over items (B=10000, seed 20260721) ----
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, n_items, size=(BOOT_B, n_items))
    task_code = np.zeros(n_items, dtype=np.int8)
    for ti, t in enumerate(TASKS):
        task_code[task_masks[t]] = ti
    tk = task_code[idx]                                # (B, 500)
    boot_macro = {}
    for cfg in CONFIGS:
        cr = correct[cfg][idx]                         # (B, 500) same resample all cfgs
        taccs = np.empty((BOOT_B, 4))
        for ti in range(4):
            m = (tk == ti)
            cnt = m.sum(axis=1)
            assert cnt.min() > 0                       # ~impossible to fail
            taccs[:, ti] = (cr & m).sum(axis=1) / cnt
        boot_macro[cfg] = taccs.mean(axis=1)
    Y = np.stack([(boot_macro["reference"] - boot_macro[c]) * 100
                  for c in CONFIGS], axis=1)           # (B, 4) drops in points
    sx = np.sqrt(x)
    b_lin = Y @ x / float(x @ x)                       # pts per nat
    c_sqrt = Y @ sx / float(np.sum(x))                 # pts per sqrt-nat (sum sx^2 == sum x)

    def pct(a, q):
        return float(np.percentile(a, q))

    boot = {
        "B": BOOT_B, "seed": BOOT_SEED,
        "linear_origin_slope_pts_per_0.01nats": {
            "point": round(fits_macro["linear_origin"]["params"]["slope_pts_per_nat"] / 100, 4),
            "ci95": [round(pct(b_lin, 2.5) / 100, 4), round(pct(b_lin, 97.5) / 100, 4)],
        },
        "sqrt_origin_coef_pts_per_sqrtnat": {
            "point": fits_macro["sqrt_origin"]["params"]["coef_pts_per_sqrtnat"],
            "ci95": [round(pct(c_sqrt, 2.5), 3), round(pct(c_sqrt, 97.5), 3)],
        },
        "macro_drop_points_ci95": {
            c: [round(pct(Y[:, i], 2.5), 3), round(pct(Y[:, i], 97.5), 3)]
            for i, c in enumerate(CONFIGS) if c != "reference"
        },
    }

    # budgets per replicate (invert each form)
    budgets = {}
    for pts in (1.5, 3.0):
        sq = (pts / c_sqrt) ** 2
        li = pts / b_lin
        c_pt = fits_macro["sqrt_origin"]["params"]["coef_pts_per_sqrtnat"]
        b_pt = fits_macro["linear_origin"]["params"]["slope_pts_per_nat"]
        budgets[f"{pts}pts"] = {
            "sqrt_origin_nats": {"point": round((pts / c_pt) ** 2, 6),
                                 "ci95": [round(pct(sq, 2.5), 6), round(pct(sq, 97.5), 6)]},
            "linear_origin_nats": {"point": round(pts / b_pt, 6),
                                   "ci95": [round(pct(li, 2.5), 6), round(pct(li, 97.5), 6)]},
        }

    # ---- MMLU subject-cluster bootstrap sensitivity (57 subjects) ----
    subjects = sorted(set(mmlu_subject.values()))
    subj_items = {s: np.array([id_index[i] for i in ref_ids
                               if task_of[id_index[i]] == "mmlu"
                               and mmlu_subject[i] == s]) for s in subjects}
    rng2 = np.random.default_rng(BOOT_SEED + 1)
    sidx = rng2.integers(0, len(subjects), size=(BOOT_B, len(subjects)))
    k8_drop_mmlu = np.empty(BOOT_B)
    ref_arr, k8_arr = correct["reference"], correct["k8"]
    subj_ref = np.array([ref_arr[subj_items[s]].sum() for s in subjects], float)
    subj_k8 = np.array([k8_arr[subj_items[s]].sum() for s in subjects], float)
    subj_n = np.array([len(subj_items[s]) for s in subjects], float)
    ns = subj_n[sidx].sum(axis=1)
    k8_drop_mmlu = (subj_ref[sidx].sum(axis=1) - subj_k8[sidx].sum(axis=1)) / ns * 100
    # iid item bootstrap CI for the same quantity, from the main bootstrap
    m_mmlu = (tk == 3)
    iid_drop = ((correct["reference"][idx] & m_mmlu).sum(1)
                - (correct["k8"][idx] & m_mmlu).sum(1)) / m_mmlu.sum(1) * 100
    cluster_sens = {
        "quantity": "k8 vs reference MMLU drop (points)",
        "point": round((acc["reference"]["task"]["mmlu"] - acc["k8"]["task"]["mmlu"]) * 100, 3),
        "iid_item_bootstrap_ci95": [round(pct(iid_drop, 2.5), 3), round(pct(iid_drop, 97.5), 3)],
        "subject_cluster_bootstrap_ci95": [round(pct(k8_drop_mmlu, 2.5), 3),
                                           round(pct(k8_drop_mmlu, 97.5), 3)],
        "n_subjects": len(subjects),
    }

    # ---- gate recalibration ----
    tax_k8 = tax["k8"]["tax"]
    min_single_on = min(singles_on.values())
    gate = {
        "provisional_gate_nats_onpolicy": 0.05,
        "provisional_gate_predicted_drop_points": {
            "sqrt_origin": round(fits_macro["sqrt_origin"]["params"]["coef_pts_per_sqrtnat"]
                                 * math.sqrt(0.05), 3),
            "linear_origin": round(fits_macro["linear_origin"]["params"]["slope_pts_per_nat"]
                                   * 0.05, 3),
        },
        "off_over_on_ratio": {"per_set": {k: round(v, 4) for k, v in ratios.items()},
                              "mean": round(ratio_mean, 4),
                              "pooled": round(ratio_pooled, 4)},
        "interaction_tax": {k: round(v["tax"], 4) for k, v in tax.items()},
        "recommended_form": "sqrt_origin (conservative near zero; see md)",
        "budgets": {},
        "min_measured_single_block_kl_on": round(min_single_on, 6),
    }
    for pts in (1.5, 3.0):
        bset = budgets[f"{pts}pts"]["sqrt_origin_nats"]["point"]
        bset_lin = budgets[f"{pts}pts"]["linear_origin_nats"]["point"]
        gate["budgets"][f"{pts}pts"] = {
            "a_onpolicy_set_budget_nats": bset,
            "a_ci95": budgets[f"{pts}pts"]["sqrt_origin_nats"]["ci95"],
            "b_offpolicy_companion_ceiling_nats": round(bset * ratio_mean, 6),
            "c_single_block_screen_for_k8_nats": round(bset / (tax_k8 * 8), 8),
            "linear_sensitivity_set_budget_nats": bset_lin,
            "linear_sensitivity_single_block_k8_nats": round(bset_lin / (tax_k8 * 8), 8),
            "k8_feasible_with_current_inventory": bool(bset / (tax_k8 * 8) >= min_single_on),
        }

    # ---- context: truncation and thinking-length ----
    gen_ctx = {}
    for cfg in CONFIGS:
        gen_ctx[cfg] = {
            "n_finish_length": int(finlen[cfg].sum()),
            "mean_gen_tokens": round(float(gentok[cfg].mean()), 1),
        }

    result = {
        "meta": {
            "analysis": "d_kl_fit",
            "date": "2026-07-21",
            "seed": BOOT_SEED,
            "bootstrap_B": BOOT_B,
            "note": "KL x-values held fixed in bootstrap (probe-set noise not propagated)",
        },
        "anchors": {
            c: {
                "kl_onpolicy_nats": round(kl_anchor[c], 6),
                "kl_offpolicy_nats": round(kl_anchor_off[c], 6),
                "task_accuracy": {t: round(acc[c]["task"][t], 4) for t in TASKS},
                "macro": round(acc[c]["macro"], 5),
                "macro_drop_points": round(float(y_macro[i]), 3),
            } for i, c in enumerate(CONFIGS)
        },
        "fits_macro": fits_macro,
        "fits_per_task": fits_task,
        "bootstrap": boot,
        "budgets_inverted": budgets,
        "mcnemar_exact": mcnemar,
        "wilson_ci_per_task": wilson,
        "mmlu_cluster_sensitivity": cluster_sens,
        "gate_recalibration": gate,
        "generation_context": gen_ctx,
        "kl_raw": {"topk_onpolicy": kl_on, "topk_offpolicy": kl_off,
                   "singles_onpolicy": singles_on},
    }
    with open(os.path.join(OUT, "d_kl_fit.json"), "w") as f:
        json.dump(result, f, indent=2)

    # ---- markdown summary ----
    L = []
    L.append("# d_kl_fit — KL→damage mapping and gate recalibration (2026-07-21)\n")
    L.append("Anchors (on-policy set KL from topk_onpolicy.json; accuracies recomputed "
             "from items, matched stored values exactly; N=500 paired items).\n")
    L.append("| config | KL_on (nats) | KL_off (nats) | GSM8K | MATH500 | IFEval | MMLU | macro | drop (pts) |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for i, c in enumerate(CONFIGS):
        a = result["anchors"][c]
        t = a["task_accuracy"]
        L.append(f"| {c} | {a['kl_onpolicy_nats']:.5f} | {a['kl_offpolicy_nats']:.5f} | "
                 f"{t['gsm8k']:.3f} | {t['math500']:.3f} | {t['ifeval']:.3f} | {t['mmlu']:.3f} | "
                 f"{a['macro']:.4f} | {a['macro_drop_points']:.2f} |")
    L.append("\n## Fits: macro drop (points) vs on-policy KL (4 points — form underdetermined)\n")
    L.append("| form | params | residuals @ (ref,k2,k6,k8) pts | RMSE | R2 |")
    L.append("|---|---|---|---|---|")
    for name, ft in fits_macro.items():
        L.append(f"| {ft['form']} | {ft['params']} | {ft['residuals_points']} | "
                 f"{ft['rmse_points']} | {ft['r2_centered']} |")
    L.append("\nRecommended: **sqrt_origin** (drop = c·sqrt(KL)). It is the only 1-parameter "
             "form whose residuals are <0.3 pts at every anchor (linear-origin misses k2 by "
             "+1.7 pts); the anchors are clearly concave (marginal pts/nat fall 380→209→142 "
             "across k2/k6/k8). Near KL→0 the sqrt form over-predicts damage — the "
             "conservative direction for an accept gate. Linear-origin kept as sensitivity.\n")
    bl = boot["linear_origin_slope_pts_per_0.01nats"]
    bs = boot["sqrt_origin_coef_pts_per_sqrtnat"]
    L.append(f"Paired item bootstrap (B={BOOT_B}, seed {BOOT_SEED}): linear-origin slope "
             f"= {bl['point']} pts per 0.01 nats, 95% CI [{bl['ci95'][0]}, {bl['ci95'][1]}]; "
             f"sqrt coef = {bs['point']} pts/sqrt-nat, 95% CI [{bs['ci95'][0]}, {bs['ci95'][1]}]. "
             "KL values held fixed (probe noise not propagated).\n")
    L.append("Macro-drop 95% CIs (paired bootstrap, points): "
             + ", ".join(f"{c}: {v}" for c, v in boot["macro_drop_points_ci95"].items()) + "\n")
    L.append("## Per-task fits (drops in points at k2/k6/k8)\n")
    L.append("| task | drops (k2,k6,k8) | sqrt coef (pts/sqrt-nat) | lin slope (pts/0.01nats) | sqrt RMSE (pts) | note |")
    L.append("|---|---|---|---|---|---|")
    notes = {"gsm8k": "shallow; k2 slightly above ref",
             "math500": "NON-MONOTONE (k8 = ref); fit is descriptive only, high residual",
             "ifeval": "steep, clean monotone staircase",
             "mmlu": "steepest; drives k8 damage"}
    for t in TASKS:
        ft = fits_task[t]
        L.append(f"| {t} | {ft['drops_points'][1:]} | "
                 f"{ft['sqrt_origin']['params']['coef_pts_per_sqrtnat']} | "
                 f"{round(ft['linear_origin']['params']['slope_pts_per_nat']/100, 3)} | "
                 f"{ft['sqrt_origin']['rmse_points']} | {notes[t]} |")
    L.append("\n## Paired McNemar exact tests (vs reference)\n")
    L.append("| comparison | scope | Δacc (pts) | discordant (ref+/cfg-, ref-/cfg+) | p (exact, 2-sided) |")
    L.append("|---|---|---|---|---|")
    for key, entry in mcnemar.items():
        if key.startswith("math500_"):
            e = entry
            L.append(f"| {key} | math500 | {e['delta_acc_points']} | "
                     f"({e['discordant_a_right_b_wrong']}, {e['discordant_a_wrong_b_right']}) | "
                     f"{e['p_exact_two_sided']} |")
        else:
            for scope, e in entry.items():
                L.append(f"| {key} | {scope} | {e['delta_acc_points']} | "
                         f"({e['discordant_ref_right_cfg_wrong']}, {e['discordant_ref_wrong_cfg_right']}) | "
                         f"{e['p_exact_two_sided']} |")
    L.append("\n## Interaction tax and off/on ratio (measured)\n")
    L.append("| set | KL_on | Σ singles_on | tax | KL_off/KL_on |")
    L.append("|---|---|---|---|---|")
    for k in ("k2", "k4", "k6", "k8"):
        tv = tax[k]
        L.append(f"| {k} | {tv['set_kl_on']:.5f} | {tv['sum_singles_on']:.5f} | "
                 f"{tv['tax']:.3f} | {ratios[k]:.3f} |")
    L.append(f"\nOff/on ratio: mean {ratio_mean:.3f}, pooled {ratio_pooled:.3f}, "
             f"range {min(ratios.values()):.2f}–{max(ratios.values()):.2f} "
             "(k-dependent: rises with k).\n")
    L.append("## Gate recalibration (replaces provisional 0.05-nats on-policy screen)\n")
    L.append(f"The old 0.05-nats gate maps to a predicted macro drop of "
             f"~{gate['provisional_gate_predicted_drop_points']['sqrt_origin']} pts (sqrt) / "
             f"{gate['provisional_gate_predicted_drop_points']['linear_origin']} pts (linear) "
             "— roughly 5x too lax for a 1.5-pt Gate B.\n")
    L.append("| Gate B budget | (a) on-policy set budget (nats) [95% CI] | (b) off-policy ceiling (nats) | (c) k8 single-block screen (nats) | linear-fit sensitivity (set / single) | k8 feasible? |")
    L.append("|---|---|---|---|---|---|")
    for pts in ("1.5pts", "3.0pts"):
        g = gate["budgets"][pts]
        L.append(f"| {pts.replace('pts',' pts')} | {g['a_onpolicy_set_budget_nats']:.5f} "
                 f"[{g['a_ci95'][0]:.5f}, {g['a_ci95'][1]:.5f}] | "
                 f"{g['b_offpolicy_companion_ceiling_nats']:.5f} | "
                 f"{g['c_single_block_screen_for_k8_nats']:.6f} | "
                 f"{g['linear_sensitivity_set_budget_nats']:.5f} / "
                 f"{g['linear_sensitivity_single_block_k8_nats']:.6f} | "
                 f"{'YES' if g['k8_feasible_with_current_inventory'] else 'NO'} |")
    L.append(f"\nSingle-block screen uses budget / (tax(8)·8) with measured tax(8) = "
             f"{tax_k8:.3f}. Best measured single block is drop[12] at "
             f"{min_single_on:.5f} nats — above BOTH per-block thresholds, so **no k~8 set "
             "from the current inventory can pass Gate B at 1.5 or even 3.0 points**; "
             "beyond-k8 greedy selection under Gate B is ruled out by this calibration. "
             "At 3.0 pts, k2 passes directly on measured accuracy (2.88 pts); k4 (KL_on "
             "0.0177) is where the two fit forms disagree most (predicted 4.8 vs 2.8 pts) — "
             "benchmarking k4 would discriminate the forms.\n")
    L.append("## MMLU clustering sensitivity\n")
    cs = cluster_sens
    L.append(f"k8 MMLU drop = {cs['point']} pts; iid-item bootstrap CI "
             f"{cs['iid_item_bootstrap_ci95']} vs subject-cluster bootstrap CI "
             f"{cs['subject_cluster_bootstrap_ci95']} ({cs['n_subjects']} subjects). "
             "Cluster CI is the honest one for MMLU given subject-clustered item order.\n")
    L.append("## Caveats\n")
    L.append("- Only 4 anchors (3 non-zero): functional form is underdetermined; sqrt vs "
             "linear differ up to ~2 pts in the unmeasured k3–k5 region and near zero.")
    L.append("- KL x-values are point estimates from 100 probes; bootstrap propagates only "
             "bench-item noise, so budget CIs are too narrow.")
    L.append("- Per-task fits use N=100 (200 for MMLU); Wilson CIs are ±7–9 pts — underpowered.")
    L.append("- math500 is non-monotone (k8 = reference): the sqrt fit reports it as high "
             "residual; do not use the math500 fit predictively.")
    L.append("- Off/on ratio is k-dependent (4.4→6.6); the mean (~5.5x) ceiling is a guard, "
             "not a law.")
    L.append("- Budgets extrapolate BELOW the smallest measured anchor (k2 = 0.0076 nats): "
             "no direct accuracy measurement exists in the accept region.")
    with open(os.path.join(OUT, "d_kl_fit.md"), "w") as f:
        f.write("\n".join(L) + "\n")

    print(json.dumps({
        "anchors_kl_on": {c: result["anchors"][c]["kl_onpolicy_nats"] for c in CONFIGS},
        "macro_drops_pts": {c: result["anchors"][c]["macro_drop_points"] for c in CONFIGS},
        "fits": {k: v["params"] | {"rmse": v["rmse_points"], "r2": v["r2_centered"]}
                 for k, v in fits_macro.items()},
        "boot_slope_pts_per_0.01nats": boot["linear_origin_slope_pts_per_0.01nats"],
        "boot_sqrt_coef": boot["sqrt_origin_coef_pts_per_sqrtnat"],
        "gate": gate["budgets"],
        "tax": gate["interaction_tax"],
        "ratio": gate["off_over_on_ratio"],
        "cluster_sens": cluster_sens,
    }, indent=2))


if __name__ == "__main__":
    main()
