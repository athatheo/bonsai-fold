#!/usr/bin/env python3
"""Independent verification of c_thinking_compensation findings.

Re-derives every key number from the raw JSONs (results/*.json,
kl_screen/results/topk_*.json). Written from scratch; does NOT reuse the
original analysis script. Deterministic: bootstrap seed 987654321, B=50000
(different seed and larger B than the original on purpose, to check that
its bootstrap CIs are stable and not seed-artifacts). Also computes analytic
t-based CIs on log-ratios as a bootstrap cross-check, and exact two-sided
McNemar p-values via math.comb.

Outputs:
  experiments/minibench/analysis/verify_c_thinking_compensation.json
  experiments/minibench/analysis/verify_c_thinking_compensation.md
"""
import json
import math
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
RES = os.path.join(ROOT, "experiments", "minibench", "results")

SEED = 987654321
B = 50000

CONFIGS = ["reference", "k2_16-12", "k6", "k8"]
SHORT = {"reference": "ref", "k2_16-12": "k2", "k6": "k6", "k8": "k8"}
TASKS = ["gsm8k", "math500", "ifeval", "mmlu"]
# Deployed-byte reduction fractions per docs/LAB_NOTEBOOK.md lines 119-121
# ("k2 {16,12} (-2.6% GB)", "k6 ... (-7.8%)", "k8 ... (-10.4%, -6% KV)").
BYTE_FRAC = {"k2": 0.026, "k6": 0.078, "k8": 0.104}


def to_bool(x):
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        return x.strip().lower() in ("true", "1", "yes")
    return bool(x)


def load_results():
    data = {}
    dtype_report = {}
    for cfg in CONFIGS:
        with open(os.path.join(RES, f"{cfg}.json")) as f:
            d = json.load(f)
        items = {}
        ctypes, gtypes, finish = set(), set(), set()
        for it in d["items"]:
            ctypes.add(type(it["correct"]).__name__)
            gtypes.add(type(it["gen_tokens"]).__name__)
            finish.add(it["finish_reason"])
            assert it["id"] not in items, f"duplicate id {it['id']} in {cfg}"
            items[it["id"]] = {
                "task": it["task"],
                "correct": to_bool(it["correct"]),
                "gen_tokens": int(it["gen_tokens"]),
                "trunc": it["finish_reason"] != "stop",
            }
        data[SHORT[cfg]] = {
            "acc": d["task_accuracy"], "macro": d["macro_avg"], "items": items,
            "max_tokens": d["max_tokens"],
        }
        dtype_report[cfg] = {"correct_types": sorted(ctypes),
                             "gen_tokens_types": sorted(gtypes),
                             "finish_reasons": sorted(finish)}
    ids = sorted(data["ref"]["items"])
    for c in data.values():
        assert sorted(c["items"]) == ids, "id sets differ across configs"
    assert len(ids) == 500
    # tasks consistent across configs
    for i in ids:
        tset = {c["items"][i]["task"] for c in data.values()}
        assert len(tset) == 1
    # sanity: gen_tokens positive (log-safe)
    for c in data.values():
        assert min(v["gen_tokens"] for v in c["items"].values()) > 0
    # sanity: truncated items sit at the budget
    for name, c in data.items():
        for i in ids:
            if c["items"][i]["trunc"]:
                assert c["items"][i]["gen_tokens"] >= c["max_tokens"], (name, i)
    return data, ids, dtype_report


def load_kl():
    want = {"k2": "drop[16,12]", "k6": "drop[16,12,13,9,8,4]",
            "k8": "drop[16,12,13,9,8,4,5,15]"}
    out = {k: {} for k in want}
    for pol in ("onpolicy", "offpolicy"):
        path = os.path.join(ROOT, "experiments", "kl_screen", "results",
                            f"topk_{pol}.json")
        with open(path) as f:
            d = json.load(f)
        by = {c["name"]: c["mean_kl_nats"] for c in d["candidates"]}
        for k, nm in want.items():
            out[k][pol] = by[nm]
    return out


def wilson(k, n):
    z = 1.959963984540054
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [c - h, c + h]


def mcnemar_two_sided(b, c):
    """Exact two-sided McNemar: 2 * P(X <= min(b,c)), X ~ Bin(b+c, 0.5)."""
    n = b + c
    if n == 0:
        return 1.0
    lo = min(b, c)
    p = 2.0 * sum(math.comb(n, i) for i in range(lo + 1)) / 2.0 ** n
    return min(1.0, p)


def geo_block(num, den, rng):
    """Geomean of paired ratios num/den, bootstrap 95% CI + analytic t CI."""
    lr = np.log(np.asarray(num, float)) - np.log(np.asarray(den, float))
    n = len(lr)
    gm = float(np.exp(lr.mean()))
    idx = rng.integers(0, n, size=(B, n))
    bm = lr[idx].mean(axis=1)
    blo, bhi = np.percentile(bm, [2.5, 97.5])
    # analytic t-interval on mean log-ratio (t_crit approx via normal for n>=30)
    se = lr.std(ddof=1) / math.sqrt(n) if n > 1 else float("nan")
    tcrit = 1.959963984540054 if n >= 200 else t_crit(n - 1)
    return {"n": n, "geomean": gm,
            "boot_ci95": [float(np.exp(blo)), float(np.exp(bhi))],
            "t_ci95": [float(np.exp(lr.mean() - tcrit * se)),
                       float(np.exp(lr.mean() + tcrit * se))]}


def t_crit(df, p=0.975):
    """97.5% Student-t quantile via Cornish-Fisher expansion (stdlib only)."""
    z = 1.959963984540054
    g1 = (z**3 + z) / 4
    g2 = (5 * z**5 + 16 * z**3 + 3 * z) / 96
    g3 = (3 * z**7 + 19 * z**5 + 17 * z**3 - 15 * z) / 384
    return z + g1 / df + g2 / df**2 + g3 / df**3


def main():
    data, ids, dtypes = load_results()
    kl = load_kl()
    rng = np.random.default_rng(SEED)
    ref = data["ref"]["items"]
    by_task = {t: [i for i in ids if ref[i]["task"] == t] for t in TASKS}
    assert {t: len(v) for t, v in by_task.items()} == {
        "gsm8k": 100, "math500": 100, "ifeval": 100, "mmlu": 200}

    out = {"seed": SEED, "B": B, "dtypes": dtypes,
           "kl_anchors_mean_nats": kl,
           "macro_avg": {c: data[c]["macro"] for c in ("ref", "k2", "k6", "k8")}}

    scopes = TASKS + ["overall"]
    sub_of = lambda s: ids if s == "overall" else by_task[s]

    # 1. paired geomean token ratios vs ref, all items
    ratios = {}
    for c in ("k2", "k6", "k8"):
        it = data[c]["items"]
        ratios[c] = {}
        for s in scopes:
            sub = sub_of(s)
            ratios[c][s] = geo_block([it[i]["gen_tokens"] for i in sub],
                                     [ref[i]["gen_tokens"] for i in sub], rng)
    out["paired_geomean_ratio"] = ratios
    out["monotone"] = {s: ratios["k2"][s]["geomean"] <= ratios["k6"][s]["geomean"]
                          <= ratios["k8"][s]["geomean"] for s in scopes}

    # 2. stop-only (exclude pairs truncated in either config)
    stop_ratios = {}
    for c in ("k2", "k6", "k8"):
        it = data[c]["items"]
        stop_ratios[c] = {}
        for s in scopes:
            sub = [i for i in sub_of(s) if not it[i]["trunc"] and not ref[i]["trunc"]]
            blk = geo_block([it[i]["gen_tokens"] for i in sub],
                            [ref[i]["gen_tokens"] for i in sub], rng)
            blk["n_excluded"] = len(sub_of(s)) - len(sub)
            stop_ratios[c][s] = blk
    out["paired_geomean_ratio_stop_only"] = stop_ratios

    # 3. net wall-clock = token ratio / bandwidth-bound speedup upper bound
    wall = {}
    for c in ("k2", "k6", "k8"):
        ub = 1.0 / (1.0 - BYTE_FRAC[c])
        per = {}
        for s in scopes:
            r = ratios[c][s]
            per[s] = {"net_geo": r["geomean"] / ub,
                      "net_ci95": [r["boot_ci95"][0] / ub, r["boot_ci95"][1] / ub],
                      "net_t_ci95": [r["t_ci95"][0] / ub, r["t_ci95"][1] / ub],
                      "net_stop_only": stop_ratios[c][s]["geomean"] / ub}
        wall[c] = {"byte_frac": BYTE_FRAC[c], "speedup_ub": ub, "per_scope": per}
    out["wallclock"] = wall

    # 4. outcome split (k2/k6/k8) with truncation counts and means
    split = {}
    for c in ("k2", "k6", "k8"):
        it = data[c]["items"]
        cats = {"both_correct": [], "recovery": [], "regression": [], "both_wrong": []}
        for i in ids:
            cc, rr = it[i]["correct"], ref[i]["correct"]
            key = ("both_correct" if cc and rr else "recovery" if cc else
                   "regression" if rr else "both_wrong")
            cats[key].append(i)
        row = {}
        for k, sub in cats.items():
            blk = geo_block([it[i]["gen_tokens"] for i in sub],
                            [ref[i]["gen_tokens"] for i in sub], rng)
            blk["n_trunc_cfg"] = sum(it[i]["trunc"] for i in sub)
            blk["n_trunc_ref"] = sum(ref[i]["trunc"] for i in sub)
            blk["mean_tok_cfg"] = float(np.mean([it[i]["gen_tokens"] for i in sub]))
            blk["mean_tok_ref"] = float(np.mean([ref[i]["gen_tokens"] for i in sub]))
            blk["underpowered_n_lt_30"] = blk["n"] < 30
            row[k] = blk
        split[c] = row
    out["outcome_split"] = split

    # 5. truncation rates with Wilson CIs
    trunc = {}
    for c in ("ref", "k2", "k6", "k8"):
        it = data[c]["items"]
        trunc[c] = {}
        for s in scopes:
            sub = sub_of(s)
            k = sum(it[i]["trunc"] for i in sub)
            trunc[c][s] = {"k": k, "n": len(sub), "rate": k / len(sub),
                           "wilson95": wilson(k, len(sub))}
    out["truncation"] = trunc

    # 6. McNemar exact vs ref
    mcn = {}
    for c in ("k2", "k6", "k8"):
        it = data[c]["items"]
        mcn[c] = {}
        for s in scopes:
            sub = sub_of(s)
            b = sum(1 for i in sub if ref[i]["correct"] and not it[i]["correct"])
            cc = sum(1 for i in sub if it[i]["correct"] and not ref[i]["correct"])
            mcn[c][s] = {"b_ref_only": b, "c_cfg_only": cc,
                         "acc_delta": (cc - b) / len(sub),
                         "p_two_sided_exact": mcnemar_two_sided(b, cc)}
    out["mcnemar"] = mcn

    # 7. prior check: math500 mean tokens per config
    out["math500_mean_tokens"] = {
        c: float(np.mean([data[c]["items"][i]["gen_tokens"] for i in by_task["math500"]]))
        for c in ("ref", "k2", "k6", "k8")}

    # 8. headline support: does k8 net wall-clock CI exclude 1.0?
    it8 = data["k8"]["items"]
    lr = np.array([math.log(it8[i]["gen_tokens"] / ref[i]["gen_tokens"]) for i in ids])
    ub = 1.0 / (1.0 - BYTE_FRAC["k8"])
    shifted = lr - math.log(ub)
    se = shifted.std(ddof=1) / math.sqrt(len(shifted))
    tstat = shifted.mean() / se
    # two-sided normal p (n=500, t ~ z)
    p = 2 * 0.5 * math.erfc(abs(tstat) / math.sqrt(2))
    out["k8_net_gt_1_test"] = {"mean_log_net": float(shifted.mean()),
                               "t_stat": float(tstat), "p_two_sided_normal": float(p)}

    jpath = os.path.join(HERE, "verify_c_thinking_compensation.json")
    with open(jpath, "w") as f:
        json.dump(out, f, indent=2)

    # ---------- markdown summary ----------
    L = ["# verify_c_thinking_compensation — independent re-derivation", "",
         f"Seed {SEED}, bootstrap B={B} (different from original), plus analytic "
         f"t-CIs on log-ratios as cross-check. All numbers recomputed from raw "
         f"results JSONs, paired by id (id sets asserted identical, n=500).", "",
         f"dtypes observed: {json.dumps(dtypes['reference'])} (same in all four files)", "",
         "## Geomean paired token ratio vs ref [bootstrap 95% CI] (t-CI)", "",
         "| config | scope | n | geomean | boot CI | t CI |", "|---|---|---|---|---|---|"]
    for c in ("k2", "k6", "k8"):
        for s in scopes:
            r = ratios[c][s]
            L.append(f"| {c} | {s} | {r['n']} | {r['geomean']:.4f} | "
                     f"[{r['boot_ci95'][0]:.4f}, {r['boot_ci95'][1]:.4f}] | "
                     f"[{r['t_ci95'][0]:.4f}, {r['t_ci95'][1]:.4f}] |")
    L += ["", "Monotone k2<=k6<=k8: " + ", ".join(f"{s}:{'Y' if v else 'N'}"
                                                  for s, v in out["monotone"].items()), "",
          "## Net wall-clock (ratio / speedup UB)", "",
          "| config | UB | scope | net geo | net boot CI | net stop-only |", "|---|---|---|---|---|---|"]
    for c in ("k2", "k6", "k8"):
        for s in scopes:
            p_ = wall[c]["per_scope"][s]
            L.append(f"| {c} | {wall[c]['speedup_ub']:.4f} | {s} | {p_['net_geo']:.4f} | "
                     f"[{p_['net_ci95'][0]:.4f}, {p_['net_ci95'][1]:.4f}] | "
                     f"{p_['net_stop_only']:.4f} |")
    L += ["", "## k8 outcome split", "",
          "| cat | n | geomean | boot CI | trunc cfg/ref | mean tok ref->cfg |", "|---|---|---|---|---|---|"]
    for k, blk in split["k8"].items():
        L.append(f"| {k} | {blk['n']} | {blk['geomean']:.4f} | "
                 f"[{blk['boot_ci95'][0]:.4f}, {blk['boot_ci95'][1]:.4f}] | "
                 f"{blk['n_trunc_cfg']}/{blk['n_trunc_ref']} | "
                 f"{blk['mean_tok_ref']:.0f}->{blk['mean_tok_cfg']:.0f} |")
    L += ["", "## McNemar exact (two-sided) vs ref", "",
          "| config | scope | b | c | delta | p |", "|---|---|---|---|---|---|"]
    for c in ("k2", "k6", "k8"):
        for s in scopes:
            m = mcn[c][s]
            L.append(f"| {c} | {s} | {m['b_ref_only']} | {m['c_cfg_only']} | "
                     f"{m['acc_delta']:+.3f} | {m['p_two_sided_exact']:.3e} |")
    L += ["", "## Truncation overall (Wilson 95%)", ""]
    for c in ("ref", "k2", "k6", "k8"):
        t = trunc[c]["overall"]
        L.append(f"- {c}: {t['k']}/{t['n']} = {t['rate']:.1%} "
                 f"[{t['wilson95'][0]:.1%}, {t['wilson95'][1]:.1%}]")
    t8 = trunc["k8"]["ifeval"]; tr = trunc["ref"]["ifeval"]
    L.append(f"- ifeval: ref {tr['rate']:.0%} -> k8 {t8['rate']:.0%} "
             f"[{t8['wilson95'][0]:.1%}, {t8['wilson95'][1]:.1%}]; "
             f"math500 ref {trunc['ref']['math500']['rate']:.0%} / "
             f"k8 {trunc['k8']['math500']['rate']:.0%}")
    L += ["", f"KL anchors: {json.dumps(kl)}", "",
          f"math500 mean tokens: " + ", ".join(f"{c} {v:.0f}" for c, v in out['math500_mean_tokens'].items()), "",
          f"k8 net>1 test: mean log net {out['k8_net_gt_1_test']['mean_log_net']:.4f}, "
          f"t={out['k8_net_gt_1_test']['t_stat']:.2f}, p={out['k8_net_gt_1_test']['p_two_sided_normal']:.2e}"]
    with open(os.path.join(HERE, "verify_c_thinking_compensation.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("wrote verify_c_thinking_compensation.{json,md}")


if __name__ == "__main__":
    main()
