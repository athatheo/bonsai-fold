#!/usr/bin/env python3
"""c_thinking_compensation: quantify thinking-length compensation across all
minibench items and its wall-clock consequence.

Deterministic (seed 20260721). CPU-only, pure-JSON. stdlib + numpy only.

Outputs:
  experiments/minibench/analysis/c_thinking_compensation.json
  experiments/minibench/analysis/c_thinking_compensation.md
"""
import json
import math
import os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RES = os.path.join(ROOT, "experiments", "minibench", "results")
OUT = os.path.join(ROOT, "experiments", "minibench", "analysis")

SEED = 20260721
B = 10000

CONFIGS = ["reference", "k2_16-12", "k6", "k8"]
SHORT = {"reference": "ref", "k2_16-12": "k2", "k6": "k6", "k8": "k8"}
DROPPED = {"reference": [], "k2_16-12": [16, 12], "k6": [16, 12, 13, 9, 8, 4],
           "k8": [16, 12, 13, 9, 8, 4, 5, 15]}
# Deployed-byte reductions given by the experiment design (fraction of pack bytes removed).
BYTE_FRAC = {"k2_16-12": 0.026, "k6": 0.078, "k8": 0.104}
TASKS = ["gsm8k", "math500", "ifeval", "mmlu"]


# ---------------------------------------------------------------- loading
def load():
    data = {}
    for cfg in CONFIGS:
        with open(os.path.join(RES, f"{cfg}.json")) as f:
            d = json.load(f)
        items = {}
        for it in d["items"]:
            # normalize dtypes defensively
            correct = it["correct"]
            if isinstance(correct, str):
                correct = correct.strip().lower() in ("true", "1", "yes")
            items[it["id"]] = {
                "task": it["task"],
                "correct": bool(correct),
                "gen_tokens": int(it["gen_tokens"]),
                "truncated": it["finish_reason"] != "stop",
            }
        data[cfg] = {"meta": {k: d[k] for k in ("pack", "drop", "max_tokens", "seed")},
                     "acc": d["task_accuracy"], "macro": d["macro_avg"], "items": items}
    ids = sorted(data["reference"]["items"].keys())
    for cfg in CONFIGS:
        assert sorted(data[cfg]["items"].keys()) == ids, f"id mismatch in {cfg}"
    assert len(ids) == 500
    return data, ids


def load_kl():
    name_of = {"k2_16-12": "drop[16,12]", "k6": "drop[16,12,13,9,8,4]",
               "k8": "drop[16,12,13,9,8,4,5,15]"}
    out = {"reference": {"onpolicy": 0.0, "offpolicy": 0.0}}
    for pol in ("onpolicy", "offpolicy"):
        with open(os.path.join(ROOT, "experiments", "kl_screen", "results",
                               f"topk_{pol}.json")) as f:
            d = json.load(f)
        by_name = {c["name"]: c["mean_kl_nats"] for c in d["candidates"]}
        for cfg, nm in name_of.items():
            out.setdefault(cfg, {})[pol] = by_name[nm]
    return out


# ---------------------------------------------------------------- stats helpers
def wilson_ci(k, n, z=1.959963984540054):
    """Wilson 95% CI for a binomial proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def mcnemar_exact(b, c):
    """Two-sided exact McNemar test on discordant pair counts (b, c).
    p = two-sided binomial(min(b,c); n=b+c, 0.5) via math.comb."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / 2.0 ** n
    return min(1.0, 2.0 * tail)


def geomean(x):
    x = np.asarray(x, dtype=float)
    return float(np.exp(np.mean(np.log(x)))) if len(x) else float("nan")


def boot_geomean_ci(ratios, rng, b=B):
    """95% percentile bootstrap CI for the geometric-mean of per-item ratios."""
    r = np.log(np.asarray(ratios, dtype=float))
    n = len(r)
    if n == 0:
        return (float("nan"), float("nan"))
    idx = rng.integers(0, n, size=(b, n))
    means = r[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return (float(np.exp(lo)), float(np.exp(hi)))


def tok_stats(vals):
    v = np.asarray(vals, dtype=float)
    return {"n": int(len(v)), "mean": float(v.mean()), "median": float(np.median(v)),
            "p90": float(np.percentile(v, 90))}


# ---------------------------------------------------------------- analyses
def main():
    data, ids = load()
    kl = load_kl()
    rng = np.random.default_rng(SEED)

    ref = data["reference"]["items"]
    task_of = {i: ref[i]["task"] for i in ids}
    ids_by_task = {t: [i for i in ids if task_of[i] == t] for t in TASKS}

    results = {"seed": SEED, "bootstrap_B": B,
               "kl_anchors_mean_nats": {SHORT[c]: kl[c] for c in CONFIGS},
               "task_accuracy": {SHORT[c]: data[c]["acc"] for c in CONFIGS},
               "macro_avg": {SHORT[c]: data[c]["macro"] for c in CONFIGS}}

    # ---- 1. gen_tokens stats + paired ratios vs reference -------------------
    sec1 = {}
    for cfg in CONFIGS:
        it = data[cfg]["items"]
        per_task = {t: tok_stats([it[i]["gen_tokens"] for i in ids_by_task[t]]) for t in TASKS}
        per_task["overall"] = tok_stats([it[i]["gen_tokens"] for i in ids])
        sec1[SHORT[cfg]] = per_task
    results["token_stats"] = sec1

    def ratio_block(cfg, subset_ids):
        it = data[cfg]["items"]
        ratios = [it[i]["gen_tokens"] / ref[i]["gen_tokens"] for i in subset_ids]
        gm = geomean(ratios)
        lo, hi = boot_geomean_ci(ratios, rng)
        return {"n": len(subset_ids), "geomean_ratio": gm, "ci95": [lo, hi],
                "pct_inflation": (gm - 1) * 100}

    sec1r = {}
    for cfg in CONFIGS[1:]:
        blocks = {t: ratio_block(cfg, ids_by_task[t]) for t in TASKS}
        blocks["overall"] = ratio_block(cfg, ids)
        sec1r[SHORT[cfg]] = blocks
    results["paired_ratio_vs_ref"] = sec1r

    mono = {}
    for scope in TASKS + ["overall"]:
        seq = [sec1r[c][scope]["geomean_ratio"] for c in ("k2", "k6", "k8")]
        mono[scope] = {"geomean_ratios_k2_k6_k8": seq,
                       "monotone_increasing": bool(seq[0] <= seq[1] <= seq[2])}
    results["monotonicity"] = mono

    # ---- 2. split by outcome -------------------------------------------------
    sec2 = {}
    for cfg in CONFIGS[1:]:
        it = data[cfg]["items"]
        cats = {"both_correct": [], "recovery_cfg_correct_ref_wrong": [],
                "regression_cfg_wrong_ref_correct": [], "both_wrong": []}
        for i in ids:
            c, r = it[i]["correct"], ref[i]["correct"]
            key = ("both_correct" if c and r else
                   "recovery_cfg_correct_ref_wrong" if c and not r else
                   "regression_cfg_wrong_ref_correct" if not c and r else "both_wrong")
            cats[key].append(i)
        cat_out = {}
        for k, sub in cats.items():
            if sub:
                blk = ratio_block(cfg, sub)
                blk["mean_tokens_cfg"] = float(np.mean([it[i]["gen_tokens"] for i in sub]))
                blk["mean_tokens_ref"] = float(np.mean([ref[i]["gen_tokens"] for i in sub]))
                blk["n_trunc_cfg"] = sum(1 for i in sub if it[i]["truncated"])
                blk["n_trunc_ref"] = sum(1 for i in sub if ref[i]["truncated"])
                blk["underpowered"] = len(sub) < 30
            else:
                blk = {"n": 0}
            cat_out[k] = blk
        # simple splits
        for label, pred in [("cfg_correct", lambda i: it[i]["correct"]),
                            ("cfg_wrong", lambda i: not it[i]["correct"]),
                            ("ref_correct", lambda i: ref[i]["correct"]),
                            ("ref_wrong", lambda i: not ref[i]["correct"])]:
            sub = [i for i in ids if pred(i)]
            cat_out[label] = ratio_block(cfg, sub)
        # McNemar exact vs reference (overall + per task)
        mcn = {}
        for scope, sub in [("overall", ids)] + [(t, ids_by_task[t]) for t in TASKS]:
            b = sum(1 for i in sub if ref[i]["correct"] and not it[i]["correct"])
            c = sum(1 for i in sub if not ref[i]["correct"] and it[i]["correct"])
            mcn[scope] = {"ref_only_correct_b": b, "cfg_only_correct_c": c,
                          "acc_delta": (c - b) / len(sub),
                          "p_mcnemar_exact": mcnemar_exact(b, c)}
        cat_out["mcnemar_vs_ref"] = mcn
        sec2[SHORT[cfg]] = cat_out
    # k8 vs k6 recovery: correct at k8 but wrong at BOTH k6 and ref
    it8, it6 = data["k8"]["items"], data["k6"]["items"]
    deep_rec = [i for i in ids if it8[i]["correct"] and not it6[i]["correct"]
                and not ref[i]["correct"]]
    blk = ratio_block("k8", deep_rec) if deep_rec else {"n": 0}
    blk["note"] = "k8 correct, k6 AND ref wrong; ratio is k8/ref tokens"
    blk["underpowered"] = len(deep_rec) < 30
    sec2["k8_deep_recovery_vs_k6_and_ref"] = blk
    results["outcome_split"] = sec2

    # ---- 3. truncation -------------------------------------------------------
    sec3 = {}
    for cfg in CONFIGS:
        it = data[cfg]["items"]
        row = {}
        for scope, sub in [(t, ids_by_task[t]) for t in TASKS] + [("overall", ids)]:
            k = sum(1 for i in sub if it[i]["truncated"])
            lo, hi = wilson_ci(k, len(sub))
            row[scope] = {"n_trunc": k, "n": len(sub), "rate": k / len(sub),
                          "wilson95": [lo, hi]}
        sec3[SHORT[cfg]] = row
    results["truncation"] = sec3

    # inflation excluding pairs truncated in either config (censored gen_tokens)
    sec3b = {}
    for cfg in CONFIGS[1:]:
        it = data[cfg]["items"]
        blocks = {}
        for scope, sub in [(t, ids_by_task[t]) for t in TASKS] + [("overall", ids)]:
            clean = [i for i in sub if not it[i]["truncated"] and not ref[i]["truncated"]]
            blk = ratio_block(cfg, clean)
            blk["n_excluded"] = len(sub) - len(clean)
            blocks[scope] = blk
        sec3b[SHORT[cfg]] = blocks
    results["paired_ratio_vs_ref_stop_only"] = sec3b

    # ---- 4/5. wall-clock accounting -----------------------------------------
    sec4 = {}
    for cfg in CONFIGS[1:]:
        frac = BYTE_FRAC[cfg]
        speedup = 1.0 / (1.0 - frac)  # bandwidth-bound decode upper bound
        it = data[cfg]["items"]
        per = {}
        for scope in TASKS + ["overall"]:
            sub = ids_by_task.get(scope, ids)
            gm = sec1r[SHORT[cfg]][scope]["geomean_ratio"]
            gm_lo, gm_hi = sec1r[SHORT[cfg]][scope]["ci95"]
            gm_stop = sec3b[SHORT[cfg]][scope]["geomean_ratio"]
            agg = (sum(it[i]["gen_tokens"] for i in sub) /
                   sum(ref[i]["gen_tokens"] for i in sub))
            per[scope] = {
                "tokens_geomean_ratio": gm,
                "net_wallclock_geomean": gm / speedup,
                "net_wallclock_geomean_ci95": [gm_lo / speedup, gm_hi / speedup],
                "tokens_geomean_ratio_stop_only": gm_stop,
                "net_wallclock_geomean_stop_only": gm_stop / speedup,
                "tokens_aggregate_ratio": agg,   # total-tokens ratio = batch wall-clock proxy
                "net_wallclock_aggregate": agg / speedup,
            }
        sec4[SHORT[cfg]] = {"byte_fraction_removed": frac,
                            "speedup_upper_bound": speedup, "per_scope": per}
    results["wallclock"] = sec4

    # prior-observation check (math subset means)
    results["prior_check_math500_mean_tokens"] = {
        SHORT[c]: float(np.mean([data[c]["items"][i]["gen_tokens"]
                                 for i in ids_by_task["math500"]])) for c in CONFIGS}

    with open(os.path.join(OUT, "c_thinking_compensation.json"), "w") as f:
        json.dump(results, f, indent=2)

    write_md(results)
    print("wrote c_thinking_compensation.{json,md}")


# ---------------------------------------------------------------- markdown
def write_md(r):
    L = []
    A = L.append
    A("# c_thinking_compensation — thinking-length inflation and wall-clock cost")
    A("")
    A(f"Paired analysis over 500 items x 4 configs (identical id sets asserted). "
      f"Bootstrap: B={r['bootstrap_B']}, seed={r['seed']}. Geometric-mean per-item "
      f"token ratios vs reference; McNemar exact for accuracy deltas; Wilson 95% CIs "
      f"for truncation rates.")
    A("")
    A("KL anchors (mean nats, read from kl_screen results): " + ", ".join(
        f"{c}: on={r['kl_anchors_mean_nats'][c]['onpolicy']:.4f}/"
        f"off={r['kl_anchors_mean_nats'][c]['offpolicy']:.4f}"
        for c in ("k2", "k6", "k8")))
    A("")
    A("## 1. gen_tokens by config x task")
    A("")
    A("| config | scope | mean | median | p90 | geomean ratio vs ref [95% CI] |")
    A("|---|---|---|---|---|---|")
    for c in ("ref", "k2", "k6", "k8"):
        for scope in ("gsm8k", "math500", "ifeval", "mmlu", "overall"):
            s = r["token_stats"][c][scope]
            if c == "ref":
                ratio = "1.000 (def.)"
            else:
                b = r["paired_ratio_vs_ref"][c][scope]
                ratio = f"{b['geomean_ratio']:.3f} [{b['ci95'][0]:.3f}, {b['ci95'][1]:.3f}]"
            A(f"| {c} | {scope} | {s['mean']:.0f} | {s['median']:.0f} | {s['p90']:.0f} | {ratio} |")
    A("")
    A("Monotone in k (k2<=k6<=k8, geomean ratio): " + ", ".join(
        f"{sc}: {'YES' if r['monotonicity'][sc]['monotone_increasing'] else 'NO'}"
        for sc in r["monotonicity"]))
    A("")
    A("## 2. Inflation split by outcome (ratio = cfg/ref tokens, geomean [95% CI])")
    A("")
    A("| config | category | n | geomean ratio [95% CI] | trunc cfg/ref | flag |")
    A("|---|---|---|---|---|---|")
    for c in ("k2", "k6", "k8"):
        o = r["outcome_split"][c]
        for cat in ("both_correct", "recovery_cfg_correct_ref_wrong",
                    "regression_cfg_wrong_ref_correct", "both_wrong",
                    "cfg_correct", "cfg_wrong", "ref_correct", "ref_wrong"):
            b = o[cat]
            if b.get("n", 0) == 0:
                A(f"| {c} | {cat} | 0 | — | — | empty |")
                continue
            flag = "UNDERPOWERED (n<30)" if b.get("underpowered") else ""
            trunc = (f"{b['n_trunc_cfg']}/{b['n_trunc_ref']}"
                     if "n_trunc_cfg" in b else "—")
            A(f"| {c} | {cat} | {b['n']} | {b['geomean_ratio']:.3f} "
              f"[{b['ci95'][0]:.3f}, {b['ci95'][1]:.3f}] | {trunc} | {flag} |")
    A("")
    A("Reading: recovery items are dominated by reference FLAILING (ref often hits "
      "the 16384 budget) while the folded model stops early and gets it right, so "
      "their ratio is < 1. Regression items show the folded model flailing (ratio ~2x, "
      "high cfg truncation). Extra thinking co-occurs with FAILURE, not recovery; the "
      "genuine compensation signal is the modest inflation on both-correct items.")
    d = r["outcome_split"]["k8_deep_recovery_vs_k6_and_ref"]
    if d.get("n", 0) > 0:
        A(f"| k8 | deep recovery (k8 correct, k6 & ref wrong) | {d['n']} | "
          f"{d['geomean_ratio']:.3f} [{d['ci95'][0]:.3f}, {d['ci95'][1]:.3f}] | "
          f"{'UNDERPOWERED (n<30)' if d.get('underpowered') else ''} |")
    A("")
    A("### McNemar exact (accuracy vs reference)")
    A("")
    A("| config | scope | ref-only b | cfg-only c | acc delta | p (exact) |")
    A("|---|---|---|---|---|---|")
    for c in ("k2", "k6", "k8"):
        for scope, m in r["outcome_split"][c]["mcnemar_vs_ref"].items():
            A(f"| {c} | {scope} | {m['ref_only_correct_b']} | {m['cfg_only_correct_c']} | "
              f"{m['acc_delta']:+.3f} | {m['p_mcnemar_exact']:.4f} |")
    A("")
    A("## 3. Truncation (finish_reason != 'stop'); gen_tokens censored at 16384")
    A("")
    A("| config | " + " | ".join(("gsm8k", "math500", "ifeval", "mmlu", "overall")) + " |")
    A("|---|---|---|---|---|---|")
    for c in ("ref", "k2", "k6", "k8"):
        cells = []
        for scope in ("gsm8k", "math500", "ifeval", "mmlu", "overall"):
            t = r["truncation"][c][scope]
            cells.append(f"{t['n_trunc']}/{t['n']} = {t['rate']:.1%} "
                         f"[{t['wilson95'][0]:.1%}, {t['wilson95'][1]:.1%}]")
        A(f"| {c} | " + " | ".join(cells) + " |")
    A("")
    A("Geomean ratio vs ref excluding pairs truncated in either config:")
    A("")
    A("| config | scope | n kept (excl.) | geomean ratio [95% CI] |")
    A("|---|---|---|---|")
    for c in ("k2", "k6", "k8"):
        for scope in ("gsm8k", "math500", "ifeval", "mmlu", "overall"):
            b = r["paired_ratio_vs_ref_stop_only"][c][scope]
            A(f"| {c} | {scope} | {b['n']} ({b['n_excluded']}) | "
              f"{b['geomean_ratio']:.3f} [{b['ci95'][0]:.3f}, {b['ci95'][1]:.3f}] |")
    A("")
    A("## 4/5. Wall-clock accounting (bandwidth-bound decode)")
    A("")
    A("net wall-clock factor = token ratio / speedup upper bound; > 1.000 means the "
      "folded pack is NET SLOWER per item despite being smaller.")
    A("")
    A("| config | bytes removed | speedup UB | scope | tok ratio (geo) | net (geo) [95% CI] | net (stop-only) | net (aggregate) |")
    A("|---|---|---|---|---|---|---|---|")
    for c in ("k2", "k6", "k8"):
        w = r["wallclock"][c]
        for scope in ("gsm8k", "math500", "ifeval", "mmlu", "overall"):
            p = w["per_scope"][scope]
            A(f"| {c} | {w['byte_fraction_removed']:.1%} | {w['speedup_upper_bound']:.3f}x "
              f"| {scope} | {p['tokens_geomean_ratio']:.3f} | "
              f"{p['net_wallclock_geomean']:.3f} "
              f"[{p['net_wallclock_geomean_ci95'][0]:.3f}, {p['net_wallclock_geomean_ci95'][1]:.3f}] | "
              f"{p['net_wallclock_geomean_stop_only']:.3f} | {p['net_wallclock_aggregate']:.3f} |")
    A("")
    pm = r["prior_check_math500_mean_tokens"]
    A(f"Prior-observation check (math500 mean tokens): ref {pm['ref']:.0f}, "
      f"k2 {pm['k2']:.0f}, k6 {pm['k6']:.0f}, k8 {pm['k8']:.0f}.")
    A("")
    A("Caveat: speedup upper bound assumes decode throughput scales exactly with "
      "deployed bytes (perfect bandwidth-bound scaling) and ignores prefill; actual "
      "tok/s must be measured on this machine (timed decode of a fixed prompt set on "
      "each pack) to firm up net wall-clock. Truncated items censor gen_tokens at "
      "16384, biasing inflation DOWN for configs that truncate more.")
    with open(os.path.join(OUT, "c_thinking_compensation.md"), "w") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
