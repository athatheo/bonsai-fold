#!/usr/bin/env python
"""verify_d_kl_fit: independent re-derivation of every key number in d_kl_fit.

Written from the raw JSONs only (results/*.json, kl_screen/results/*.json,
minibench_items.json). Does NOT import or reuse d_kl_fit.py.
Deterministic: bootstrap seed 20260721 (item bootstrap), 20260722 (subject cluster).
CPU-only, stdlib + numpy.

Outputs: verify_d_kl_fit.json, verify_d_kl_fit.md
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
CFG_FILES = {"reference": "reference.json", "k2": "k2_16-12.json",
             "k6": "k6.json", "k8": "k8.json"}
CONFIGS = ["reference", "k2", "k6", "k8"]
SETS = {"k2": [16, 12], "k4": [16, 12, 13, 9],
        "k6": [16, 12, 13, 9, 8, 4], "k8": [16, 12, 13, 9, 8, 4, 5, 15]}
B = 10000
SEED = 20260721


def to_bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "1", "yes"):
            return True
        if s in ("false", "0", "no"):
            return False
        raise ValueError(v)
    if isinstance(v, (int, float)) and v in (0, 1):
        return bool(v)
    raise ValueError(v)


def mcnemar_exact_2sided(b, c):
    """Two-sided exact McNemar via binomial(b+c, 0.5) on min tail, stdlib only."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def wilson(k, n, z=1.959963984540054):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def main():
    checks = []  # (name, ok, detail)

    def check(name, ok, detail=""):
        checks.append({"check": name, "ok": bool(ok), "detail": str(detail)})

    # ---------- load raw bench results, pair by id ----------
    raw = {c: json.load(open(os.path.join(RES, f))) for c, f in CFG_FILES.items()}
    by_id = {}
    for c in CONFIGS:
        d = {}
        for it in raw[c]["items"]:
            assert it["id"] not in d, f"dup id {it['id']} in {c}"
            d[it["id"]] = it
        by_id[c] = d
    ids = sorted(by_id["reference"])
    check("500 unique ids in reference", len(ids) == 500, len(ids))
    same = all(sorted(by_id[c]) == ids for c in CONFIGS)
    check("identical id sets across 4 configs", same)

    task_of = {i: by_id["reference"][i]["task"] for i in ids}
    for c in CONFIGS:  # task labels agree across configs
        assert all(by_id[c][i]["task"] == task_of[i] for i in ids)
    n_task = {t: sum(1 for i in ids if task_of[i] == t) for t in TASKS}
    check("task Ns 100/100/100/200",
          n_task == {"gsm8k": 100, "math500": 100, "ifeval": 100, "mmlu": 200}, n_task)

    corr = {c: np.array([to_bool(by_id[c][i]["correct"]) for i in ids]) for c in CONFIGS}
    tmask = {t: np.array([task_of[i] == t for i in ids]) for t in TASKS}

    # ---------- accuracies / macro drops ----------
    acc = {c: {t: float(corr[c][tmask[t]].mean()) for t in TASKS} for c in CONFIGS}
    macro = {c: float(np.mean([acc[c][t] for t in TASKS])) for c in CONFIGS}
    for c in CONFIGS:
        for t in TASKS:
            check(f"stored task_accuracy matches recomputed: {c}/{t}",
                  abs(acc[c][t] - raw[c]["task_accuracy"][t]) < 1e-12,
                  f"recomputed {acc[c][t]:.6f} stored {raw[c]['task_accuracy'][t]}")
        check(f"stored macro_avg matches recomputed: {c}",
              abs(macro[c] - raw[c]["macro_avg"]) < 1e-12,
              f"recomputed {macro[c]:.6f} stored {raw[c]['macro_avg']}")
    drops = {c: (macro["reference"] - macro[c]) * 100 for c in CONFIGS}
    check("macro drops 0/2.875/6.5/9.125",
          max(abs(drops["reference"] - 0), abs(drops["k2"] - 2.875),
              abs(drops["k6"] - 6.5), abs(drops["k8"] - 9.125)) < 1e-9,
          {c: round(drops[c], 4) for c in CONFIGS})

    # ---------- KL anchors from files ----------
    def name(blocks):
        return "drop[" + ",".join(str(b) for b in blocks) + "]"

    kl_on = {c["name"]: float(c["mean_kl_nats"])
             for c in json.load(open(os.path.join(KLD, "topk_onpolicy.json")))["candidates"]}
    kl_off = {c["name"]: float(c["mean_kl_nats"])
              for c in json.load(open(os.path.join(KLD, "topk_offpolicy.json")))["candidates"]}
    sing_on = {c["name"]: float(c["mean_kl_nats"])
               for c in json.load(open(os.path.join(KLD, "single_block_onpolicy.json")))["candidates"]}

    anch_on = {"reference": 0.0, "k2": kl_on[name(SETS["k2"])],
               "k6": kl_on[name(SETS["k6"])], "k8": kl_on[name(SETS["k8"])]}
    anch_off = {"reference": 0.0, "k2": kl_off[name(SETS["k2"])],
                "k6": kl_off[name(SETS["k6"])], "k8": kl_off[name(SETS["k8"])]}
    claimed_on = {"reference": 0.0, "k2": 0.007555, "k6": 0.031111, "k8": 0.064262}
    claimed_off = {"k2": 0.033044, "k6": 0.204610, "k8": 0.391483}
    check("claimed KL_on anchors",
          all(abs(anch_on[c] - claimed_on[c]) < 5e-7 for c in CONFIGS),
          {c: f"{anch_on[c]:.6f}" for c in CONFIGS})
    check("claimed KL_off anchors",
          all(abs(anch_off[c] - claimed_off[c]) < 5e-7 for c in ("k2", "k6", "k8")),
          {c: f"{anch_off[c]:.6f}" for c in ("k2", "k6", "k8")})

    x = np.array([anch_on[c] for c in CONFIGS])
    y = np.array([drops[c] for c in CONFIGS])
    sx = np.sqrt(x)

    # ---------- fits (independent implementation) ----------
    def r2_rmse(pred):
        res = y - pred
        rmse = float(np.sqrt(np.mean(res ** 2)))
        r2 = float(1 - np.sum(res ** 2) / np.sum((y - y.mean()) ** 2))
        return res, rmse, r2

    # linear through origin: minimize sum (y - b x)^2 -> b = <x,y>/<x,x>
    b_lin = float(x @ y / (x @ x))
    res_lin, rmse_lin, r2_lin = r2_rmse(b_lin * x)
    # sqrt through origin: y = c sqrt(x) -> c = <sx,y>/<sx,sx>
    c_sqrt = float(sx @ y / (sx @ sx))
    res_sq, rmse_sq, r2_sq = r2_rmse(c_sqrt * sx)
    # linear with intercept (closed form OLS)
    xm, ym = x.mean(), y.mean()
    b1 = float(np.sum((x - xm) * (y - ym)) / np.sum((x - xm) ** 2))
    a1 = float(ym - b1 * xm)
    res_li, rmse_li, r2_li = r2_rmse(a1 + b1 * x)

    check("sqrt coef ~36.04, RMSE ~0.15, R2 ~0.998",
          abs(c_sqrt - 36.041) < 0.01 and abs(rmse_sq - 0.147) < 0.01
          and abs(r2_sq - 0.998) < 0.001,
          f"c={c_sqrt:.4f} rmse={rmse_sq:.4f} r2={r2_sq:.5f}")
    check("sqrt residuals <0.3 pts at all anchors",
          np.max(np.abs(res_sq)) < 0.3, [round(float(r), 4) for r in res_sq])
    check("linear-origin slope ~157.2, RMSE ~1.26, R2 ~0.867",
          abs(b_lin - 157.207) < 0.05 and abs(rmse_lin - 1.264) < 0.005
          and abs(r2_lin - 0.867) < 0.002,
          f"b={b_lin:.3f} rmse={rmse_lin:.4f} r2={r2_lin:.5f}")
    check("linear-origin misses k2 by +1.7 pts",
          abs(res_lin[1] - 1.687) < 0.01, f"resid k2 = {res_lin[1]:+.4f}")
    check("linear+intercept: intercept ~1.19, R2 ~0.925",
          abs(a1 - 1.1925) < 0.005 and abs(r2_li - 0.925) < 0.001,
          f"a={a1:.4f} b={b1:.3f} r2={r2_li:.5f}")
    secants = [drops[c] / anch_on[c] for c in ("k2", "k6", "k8")]
    check("concavity: drop/KL falls ~380 -> ~209 -> ~142 pts/nat",
          abs(secants[0] - 380.5) < 1 and abs(secants[1] - 208.9) < 1
          and abs(secants[2] - 142.0) < 1,
          [round(s, 1) for s in secants])

    # ---------- paired item bootstrap ----------
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, 500, size=(B, 500))
    tcode = np.zeros(500, dtype=np.int8)
    for ti, t in enumerate(TASKS):
        tcode[tmask[t]] = ti
    tk = tcode[idx]
    boot_macro = {}
    for c in CONFIGS:
        cr = corr[c][idx]
        ta = np.empty((B, 4))
        for ti in range(4):
            m = tk == ti
            ta[:, ti] = (cr & m).sum(1) / m.sum(1)
        boot_macro[c] = ta.mean(1)
    Yb = np.stack([(boot_macro["reference"] - boot_macro[c]) * 100 for c in CONFIGS], 1)
    bl_b = Yb @ x / float(x @ x)          # linear slope per replicate (pts/nat)
    cs_b = Yb @ sx / float(sx @ sx)       # sqrt coef per replicate

    def ci(a):
        return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]

    ci_lin = ci(bl_b / 100)  # pts per 0.01 nats
    ci_sq = ci(cs_b)
    ci_drop = {c: ci(Yb[:, i]) for i, c in enumerate(CONFIGS) if c != "reference"}
    check("boot linear slope 1.57 pts/0.01nats, CI ~[1.00, 2.14]",
          abs(b_lin / 100 - 1.572) < 0.005 and abs(ci_lin[0] - 1.00) < 0.03
          and abs(ci_lin[1] - 2.14) < 0.03,
          f"point {b_lin/100:.4f} CI [{ci_lin[0]:.4f}, {ci_lin[1]:.4f}]")
    check("boot sqrt coef CI ~[22.8, 49.1]",
          abs(ci_sq[0] - 22.8) < 0.5 and abs(ci_sq[1] - 49.1) < 0.5,
          f"CI [{ci_sq[0]:.3f}, {ci_sq[1]:.3f}]")
    claimed_drop_ci = {"k2": [0.117, 5.673], "k6": [3.725, 9.38], "k8": [5.391, 12.794]}
    ok = all(abs(ci_drop[c][j] - claimed_drop_ci[c][j]) < 0.15
             for c in claimed_drop_ci for j in (0, 1))
    check("macro-drop CIs k2/k6/k8", ok,
          {c: [round(v, 3) for v in ci_drop[c]] for c in ci_drop})

    # ---------- gate budgets ----------
    ratios = {k: kl_off[name(SETS[k])] / kl_on[name(SETS[k])] for k in SETS}
    ratio_mean = float(np.mean(list(ratios.values())))
    ratio_pooled = (sum(kl_off[name(SETS[k])] for k in SETS)
                    / sum(kl_on[name(SETS[k])] for k in SETS))
    tax = {k: kl_on[name(SETS[k])] / sum(sing_on[name([b])] for b in SETS[k])
           for k in SETS}
    check("tax(2..8) = 1.000/1.175/1.334/1.878",
          abs(tax["k2"] - 1.000) < 0.001 and abs(tax["k4"] - 1.175) < 0.001
          and abs(tax["k6"] - 1.334) < 0.001 and abs(tax["k8"] - 1.878) < 0.001,
          {k: round(v, 4) for k, v in tax.items()})
    check("off/on ratios 4.37/4.88/6.58/6.09 mean 5.48 pooled 5.93",
          abs(ratios["k2"] - 4.374) < 0.005 and abs(ratios["k4"] - 4.882) < 0.005
          and abs(ratios["k6"] - 6.577) < 0.005 and abs(ratios["k8"] - 6.092) < 0.005
          and abs(ratio_mean - 5.481) < 0.005 and abs(ratio_pooled - 5.932) < 0.005,
          {**{k: round(v, 3) for k, v in ratios.items()},
           "mean": round(ratio_mean, 4), "pooled": round(ratio_pooled, 4)})

    budgets = {}
    for pts in (1.5, 3.0):
        bset = (pts / c_sqrt) ** 2
        bset_ci = ci((pts / cs_b) ** 2)
        blin = pts / b_lin
        budgets[pts] = {
            "sqrt_set_nats": bset, "sqrt_ci": bset_ci,
            "off_ceiling_nats": bset * ratio_mean,
            "per_block_k8_nats": bset / (tax["k8"] * 8),
            "lin_set_nats": blin, "lin_per_block_nats": blin / (tax["k8"] * 8),
        }
    g15, g30 = budgets[1.5], budgets[3.0]
    check("1.5pt budget 0.00173 CI [0.00093,0.00434], off 0.00949, blk 1.15e-4",
          abs(g15["sqrt_set_nats"] - 0.00173) < 2e-5
          and abs(g15["sqrt_ci"][0] - 0.00093) < 5e-5
          and abs(g15["sqrt_ci"][1] - 0.00434) < 8e-5
          and abs(g15["off_ceiling_nats"] - 0.00949) < 5e-5
          and abs(g15["per_block_k8_nats"] - 1.15e-4) < 2e-6,
          {k: (f"{v:.6f}" if not isinstance(v, list) else [f"{u:.6f}" for u in v])
           for k, v in g15.items()})
    check("3.0pt budget 0.00693 CI [0.00374,0.01734], off 0.0380, blk 4.61e-4; lin 0.0191/1.27e-3",
          abs(g30["sqrt_set_nats"] - 0.00693) < 5e-5
          and abs(g30["sqrt_ci"][0] - 0.00374) < 1.5e-4
          and abs(g30["sqrt_ci"][1] - 0.01734) < 3e-4
          and abs(g30["off_ceiling_nats"] - 0.0380) < 3e-4
          and abs(g30["per_block_k8_nats"] - 4.61e-4) < 5e-6
          and abs(g30["lin_set_nats"] - 0.0191) < 1e-4
          and abs(g30["lin_per_block_nats"] - 1.27e-3) < 1e-5,
          {k: (f"{v:.6f}" if not isinstance(v, list) else [f"{u:.6f}" for u in v])
           for k, v in g30.items()})
    check("old 0.05-nats gate -> ~8.06 (sqrt) / 7.86 (linear) pts",
          abs(c_sqrt * math.sqrt(0.05) - 8.059) < 0.01
          and abs(b_lin * 0.05 - 7.860) < 0.01,
          f"sqrt {c_sqrt*math.sqrt(0.05):.3f} lin {b_lin*0.05:.3f}")
    check("30x tighter: 0.05 / 0.00173 ~ 29",
          25 < 0.05 / g15["sqrt_set_nats"] < 32, f"{0.05/g15['sqrt_set_nats']:.1f}x")
    min_single = min(sing_on.values())
    min_single_name = min(sing_on, key=sing_on.get)
    check("best single block = drop[12] @ 0.00330 nats > both per-block thresholds",
          min_single_name == "drop[12]" and abs(min_single - 0.00330) < 1e-5
          and min_single > g15["per_block_k8_nats"]
          and min_single > g30["per_block_k8_nats"],
          f"{min_single_name} {min_single:.6f}")
    # stronger feasibility: even untaxed, 8 cheapest measured singles exceed budgets
    cheapest8 = sum(sorted(sing_on.values())[:8])
    check("k~8 infeasible even before tax (8 cheapest singles sum >> budgets)",
          cheapest8 > g30["sqrt_set_nats"] and cheapest8 > g30["lin_set_nats"],
          f"sum(8 cheapest singles)={cheapest8:.5f} vs budgets "
          f"{g30['sqrt_set_nats']:.5f}(sqrt)/{g30['lin_set_nats']:.5f}(lin)")

    # ---------- McNemar exact ----------
    mcn = {}
    for c in ("k2", "k6", "k8"):
        for scope, m in [("all", np.ones(500, bool))] + [(t, tmask[t]) for t in TASKS]:
            b = int((corr["reference"][m] & ~corr[c][m]).sum())
            cc = int((~corr["reference"][m] & corr[c][m]).sum())
            mcn[(c, scope)] = {
                "b_ref_right_cfg_wrong": b, "c_ref_wrong_cfg_right": cc,
                "delta_pts": float((corr[c][m].mean() - corr["reference"][m].mean()) * 100),
                "p": mcnemar_exact_2sided(b, cc)}
    m = tmask["math500"]
    b86 = int((corr["k6"][m] & ~corr["k8"][m]).sum())
    c86 = int((~corr["k6"][m] & corr["k8"][m]).sum())
    p86 = mcnemar_exact_2sided(b86, c86)
    check("McNemar k2 all: (34,21) p=0.105",
          mcn[("k2", "all")]["b_ref_right_cfg_wrong"] == 34
          and mcn[("k2", "all")]["c_ref_wrong_cfg_right"] == 21
          and abs(mcn[("k2", "all")]["p"] - 0.105) < 0.001,
          f"({mcn[('k2','all')]['b_ref_right_cfg_wrong']},"
          f"{mcn[('k2','all')]['c_ref_wrong_cfg_right']}) p={mcn[('k2','all')]['p']:.4f}")
    check("McNemar k6 all: (49,15) p=2.4e-5",
          mcn[("k6", "all")]["b_ref_right_cfg_wrong"] == 49
          and mcn[("k6", "all")]["c_ref_wrong_cfg_right"] == 15
          and abs(mcn[("k6", "all")]["p"] - 2.44e-5) < 2e-6,
          f"p={mcn[('k6','all')]['p']:.3g}")
    check("McNemar k8 all: (81,26) p=9.4e-8",
          mcn[("k8", "all")]["b_ref_right_cfg_wrong"] == 81
          and mcn[("k8", "all")]["c_ref_wrong_cfg_right"] == 26
          and abs(mcn[("k8", "all")]["p"] - 9.37e-8) < 5e-9,
          f"p={mcn[('k8','all')]['p']:.3g}")
    check("k8 mmlu -18.5 pts p~3e-6; k8 ifeval -14 pts p=0.0043",
          abs(mcn[("k8", "mmlu")]["delta_pts"] + 18.5) < 1e-9
          and abs(mcn[("k8", "mmlu")]["p"] - 3.02e-6) < 3e-7
          and abs(mcn[("k8", "ifeval")]["delta_pts"] + 14.0) < 1e-9
          and abs(mcn[("k8", "ifeval")]["p"] - 0.00434) < 5e-5,
          f"mmlu p={mcn[('k8','mmlu')]['p']:.3g} ifeval p={mcn[('k8','ifeval')]['p']:.3g}")
    check("math500 k8 vs ref: delta 0.0, (6,6), p=1.0",
          mcn[("k8", "math500")]["b_ref_right_cfg_wrong"] == 6
          and mcn[("k8", "math500")]["c_ref_wrong_cfg_right"] == 6
          and mcn[("k8", "math500")]["delta_pts"] == 0.0
          and mcn[("k8", "math500")]["p"] == 1.0, "")
    check("math500 k8 vs k6: +8 pts, (3,11), p=0.0574",
          b86 == 3 and c86 == 11 and abs(p86 - 0.0574) < 0.0005,
          f"({b86},{c86}) p={p86:.4f}")

    # ---------- per-task fits ----------
    per_task = {}
    for t in TASKS:
        yt = np.array([(acc["reference"][t] - acc[c][t]) * 100 for c in CONFIGS])
        ct = float(sx @ yt / (sx @ sx))
        bt = float(x @ yt / (x @ x))
        rest = yt - ct * sx
        per_task[t] = {"drops": [float(v) for v in yt], "sqrt_coef": ct,
                       "lin_slope_per_0.01": bt / 100,
                       "sqrt_resid": [float(r) for r in rest],
                       "sqrt_rmse": float(np.sqrt(np.mean(rest ** 2)))}
    check("per-task lin slopes mmlu 2.81 > ifeval 2.24 >> gsm8k 0.67 ~ math500 0.57",
          abs(per_task["mmlu"]["lin_slope_per_0.01"] - 2.811) < 0.005
          and abs(per_task["ifeval"]["lin_slope_per_0.01"] - 2.241) < 0.005
          and abs(per_task["gsm8k"]["lin_slope_per_0.01"] - 0.665) < 0.005
          and abs(per_task["math500"]["lin_slope_per_0.01"] - 0.571) < 0.005,
          {t: round(per_task[t]["lin_slope_per_0.01"], 3) for t in TASKS})
    m5r = per_task["math500"]["sqrt_resid"]
    check("math500 sqrt residuals +-4.4..4.8 pts",
          4.3 < abs(m5r[1]) < 4.5 and 4.6 < abs(m5r[2]) < 4.8 and 4.6 < abs(m5r[3]) < 4.9,
          [round(r, 3) for r in m5r])

    # ---------- MMLU subject-cluster bootstrap ----------
    frozen = json.load(open(os.path.join(ROOT, "experiments/minibench/minibench_items.json")))["items"]
    check("frozen ids match bench ids", sorted(i["id"] for i in frozen) == ids)
    subj = {i["id"]: i["source"].split("[")[0] for i in frozen if i["task"] == "mmlu"}
    subjects = sorted(set(subj.values()))
    check("57 MMLU subjects", len(subjects) == 57, len(subjects))
    pos = {i: n for n, i in enumerate(ids)}
    sitems = [np.array([pos[i] for i in ids if task_of[i] == "mmlu" and subj[i] == s])
              for s in subjects]
    sref = np.array([corr["reference"][a].sum() for a in sitems], float)
    sk8 = np.array([corr["k8"][a].sum() for a in sitems], float)
    sn = np.array([len(a) for a in sitems], float)
    rng2 = np.random.default_rng(SEED + 1)
    sidx = rng2.integers(0, 57, size=(B, 57))
    clus = (sref[sidx].sum(1) - sk8[sidx].sum(1)) / sn[sidx].sum(1) * 100
    ci_clus = ci(clus)
    mm = tmask["mmlu"]
    iid = ((corr["reference"][idx] & (tk == 3)).sum(1)
           - (corr["k8"][idx] & (tk == 3)).sum(1)) / (tk == 3).sum(1) * 100
    ci_iid = ci(iid)
    check("k8 mmlu drop cluster CI ~[9.7, 27.4] vs iid ~[11.1, 25.7]",
          abs(ci_clus[0] - 9.72) < 0.6 and abs(ci_clus[1] - 27.37) < 0.6
          and abs(ci_iid[0] - 11.11) < 0.3 and abs(ci_iid[1] - 25.74) < 0.3,
          f"cluster [{ci_clus[0]:.3f}, {ci_clus[1]:.3f}] iid [{ci_iid[0]:.3f}, {ci_iid[1]:.3f}]")

    # ---------- k4 discrimination claim ----------
    kl4 = kl_on[name(SETS["k4"])]
    pred_sq4 = c_sqrt * math.sqrt(kl4)
    pred_li4 = b_lin * kl4
    check("k4 KL_on 0.0177; predicted 4.8 (sqrt) vs 2.8 (lin) pts",
          abs(kl4 - 0.0177) < 5e-4 and abs(pred_sq4 - 4.8) < 0.05
          and abs(pred_li4 - 2.8) < 0.05,
          f"KL {kl4:.5f} sqrt {pred_sq4:.3f} lin {pred_li4:.3f}")
    # k2 gate status: measured drop passes 3.0 but not 1.5; KL vs budgets
    check("k2 measured 2.875 pts: passes 3.0-pt tolerance, fails 1.5-pt",
          drops["k2"] < 3.0 and drops["k2"] > 1.5, f"{drops['k2']:.3f}")

    # Wilson CI half-widths for per-task caveat (context only)
    wil = {t: wilson(int(corr['k8'][tmask[t]].sum()), n_task[t]) for t in TASKS}

    n_ok = sum(1 for c in checks if c["ok"])
    out = {
        "meta": {"analysis": "verify_d_kl_fit", "date": "2026-07-21",
                 "seed_item_bootstrap": SEED, "seed_cluster_bootstrap": SEED + 1,
                 "B": B},
        "n_checks": len(checks), "n_ok": n_ok,
        "all_ok": n_ok == len(checks),
        "checks": checks,
        "rederived": {
            "kl_on_anchors": {c: anch_on[c] for c in CONFIGS},
            "kl_off_anchors": {c: anch_off[c] for c in CONFIGS},
            "macro_drops_pts": {c: drops[c] for c in CONFIGS},
            "fits": {
                "sqrt_origin": {"coef": c_sqrt, "rmse": rmse_sq, "r2": r2_sq,
                                "residuals": [float(r) for r in res_sq]},
                "linear_origin": {"slope": b_lin, "rmse": rmse_lin, "r2": r2_lin,
                                  "residuals": [float(r) for r in res_lin]},
                "linear_intercept": {"intercept": a1, "slope": b1, "r2": r2_li},
            },
            "bootstrap": {"lin_slope_per_0.01nats_ci": ci_lin,
                          "sqrt_coef_ci": ci_sq,
                          "macro_drop_ci": ci_drop},
            "budgets": {str(p): budgets[p] for p in budgets},
            "tax": tax, "off_on_ratios": ratios,
            "ratio_mean": ratio_mean, "ratio_pooled": ratio_pooled,
            "min_single_block": {min_single_name: min_single},
            "mcnemar": {f"{k[0]}|{k[1]}": v for k, v in mcn.items()},
            "math500_k8_vs_k6": {"b": b86, "c": c86, "p": p86},
            "per_task": per_task,
            "mmlu_cluster_ci": ci_clus, "mmlu_iid_ci": ci_iid,
            "k4_pred": {"kl_on": kl4, "sqrt_pts": pred_sq4, "lin_pts": pred_li4},
            "wilson_k8_per_task": {t: list(wil[t]) for t in TASKS},
        },
    }
    with open(os.path.join(OUT, "verify_d_kl_fit.json"), "w") as f:
        json.dump(out, f, indent=2)

    lines = ["# verify_d_kl_fit — independent re-derivation of d_kl_fit (2026-07-21)",
             "",
             f"Checks passed: {n_ok}/{len(checks)}", "",
             "| check | ok | re-derived value |", "|---|---|---|"]
    for c in checks:
        lines.append(f"| {c['check']} | {'PASS' if c['ok'] else 'FAIL'} | {c['detail']} |")
    lines.append("")
    with open(os.path.join(OUT, "verify_d_kl_fit.md"), "w") as f:
        f.write("\n".join(lines) + "\n")

    print(json.dumps({"n_checks": len(checks), "n_ok": n_ok,
                      "failures": [c for c in checks if not c["ok"]]}, indent=2))


if __name__ == "__main__":
    main()
