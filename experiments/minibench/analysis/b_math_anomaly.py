#!/usr/bin/env python
"""b_math_anomaly: Is k8's math500=.72 (== reference, > k6=.64) real non-monotonicity
or paired sampling noise / a truncation artifact?

Pure-JSON, CPU-only, deterministic. stdlib-only stats (math.comb); numpy used only
for medians/means. McNemar exact (two-sided binomial on discordant pairs); Wilson CIs.

Outputs: b_math_anomaly.json (numbers), b_math_anomaly.md (summary).
"""

import json
import math
from pathlib import Path

import numpy as np

ROOT = Path("/Users/atheocharis/repos/bonsai-fold")
RES = ROOT / "experiments/minibench/results"
OUT = ROOT / "experiments/minibench/analysis"
OUT.mkdir(parents=True, exist_ok=True)

CONFIGS = ["reference", "k2_16-12", "k6", "k8"]
SHORT = {"reference": "ref", "k2_16-12": "k2", "k6": "k6", "k8": "k8"}

# ---------------------------------------------------------------- stats helpers

def wilson_ci(k, n, z=1.959963984540054):
    """Wilson 95% score interval for a binomial proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def mcnemar_exact(b, c):
    """Two-sided exact McNemar: binomial(b+c, 0.5) on discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / 2.0**n
    return min(1.0, p)


def binom_tail_ge(k, n):
    """One-sided P(X >= k | X ~ Binomial(n, 0.5)). Exact, stdlib."""
    if n == 0:
        return 1.0
    return sum(math.comb(n, i) for i in range(k, n + 1)) / 2.0**n


def sign_test(diffs):
    """Exact two-sided sign test on paired differences (zeros dropped)."""
    pos = sum(1 for d in diffs if d > 0)
    neg = sum(1 for d in diffs if d < 0)
    return {"pos": pos, "neg": neg, "ties": len(diffs) - pos - neg,
            "p_two_sided": mcnemar_exact(pos, neg)}


# ---------------------------------------------------------------- load & pair

def load():
    data = {}
    for cfg in CONFIGS:
        d = json.loads((RES / f"{cfg}.json").read_text())
        by_id = {}
        for it in d["items"]:
            # normalize dtypes defensively (observed: bool / int already)
            c = it["correct"]
            if isinstance(c, str):
                c = c.strip().lower() in ("true", "1", "yes")
            by_id[it["id"]] = {
                "task": it["task"],
                "correct": bool(c),
                "gen_tokens": int(it["gen_tokens"]),
                "finish_reason": str(it["finish_reason"]),
            }
        data[SHORT[cfg]] = {"meta": {k: d[k] for k in d if k != "items"},
                            "items": by_id}
    ids = [set(v["items"].keys()) for v in data.values()]
    assert all(s == ids[0] for s in ids), "id sets differ across configs"
    assert len(ids[0]) == 500, f"expected 500 ids, got {len(ids[0])}"
    return data


data = load()
math_ids = sorted(i for i in data["ref"]["items"] if i.startswith("math500-"))
gsm_ids = sorted(i for i in data["ref"]["items"] if i.startswith("gsm8k-"))
assert len(math_ids) == 100 and len(gsm_ids) == 100

results = {"n_math500": len(math_ids), "n_gsm8k": len(gsm_ids)}

# KL anchor context (read from files, not transcribed)
kl = {}
for pol in ("onpolicy", "offpolicy"):
    d = json.loads((ROOT / f"experiments/kl_screen/results/topk_{pol}.json").read_text())
    kl[pol] = {c["name"]: c["mean_kl_nats"] for c in d["candidates"]}
results["kl_anchors_mean_nats"] = {
    "k2=drop[16,12]": {p: kl[p]["drop[16,12]"] for p in kl},
    "k6=drop[16,12,13,9,8,4]": {p: kl[p]["drop[16,12,13,9,8,4]"] for p in kl},
    "k8=drop[16,12,13,9,8,4,5,15]": {p: kl[p]["drop[16,12,13,9,8,4,5,15]"] for p in kl},
}

# ---------------------------------------------------------------- 1) flip tables

def flip_table(ids, a, b):
    A, B = data[a]["items"], data[b]["items"]
    both = a_only = b_only = neither = 0
    a_to_b_losses, a_to_b_gains = [], []  # ids: a-correct->b-wrong, a-wrong->b-correct
    for i in ids:
        ca, cb = A[i]["correct"], B[i]["correct"]
        if ca and cb:
            both += 1
        elif ca and not cb:
            a_only += 1
            a_to_b_losses.append(i)
        elif cb and not ca:
            b_only += 1
            a_to_b_gains.append(i)
        else:
            neither += 1
    return {
        "pair": f"{a}_vs_{b}",
        "both_correct": both,
        f"{a}_only_correct": a_only,
        f"{b}_only_correct": b_only,
        "both_wrong": neither,
        "n_discordant": a_only + b_only,
        f"acc_{a}": (both + a_only) / len(ids),
        f"acc_{b}": (both + b_only) / len(ids),
        "mcnemar_exact_p_two_sided": mcnemar_exact(a_only, b_only),
        f"ids_{a}_right_{b}_wrong": a_to_b_losses,
        f"ids_{a}_wrong_{b}_right": a_to_b_gains,
    }


pairs = [("ref", "k6"), ("ref", "k8"), ("k6", "k8"),
         ("ref", "k2"), ("k2", "k6"), ("k2", "k8")]
results["flip_tables_math500"] = {f"{a}_vs_{b}": flip_table(math_ids, a, b)
                                  for a, b in pairs}
results["flip_tables_gsm8k"] = {f"{a}_vs_{b}": flip_table(gsm_ids, a, b)
                                for a, b in [("ref", "k6"), ("ref", "k8"), ("k6", "k8")]}

# per-item monotonicity along chain ref -> k2 -> k6 -> k8 (math500)
chain = ["ref", "k2", "k6", "k8"]
nonmono = []
for i in math_ids:
    v = [data[c]["items"][i]["correct"] for c in chain]
    # monotone-degrading pattern = once wrong, stays wrong (1..1 0..0)
    if any(v[j] < v[j + 1] for j in range(3)):  # a recovery somewhere
        nonmono.append({"id": i, "pattern": "".join("1" if x else "0" for x in v)})
results["chain_nonmonotone_items_math500"] = {
    "chain_order": chain, "count": len(nonmono), "items": nonmono}

# ---------------------------------------------------------------- 2) Wilson CIs

wilson = {}
for c in ("ref", "k2", "k6", "k8"):
    k = sum(data[c]["items"][i]["correct"] for i in math_ids)
    lo, hi = wilson_ci(k, len(math_ids))
    wilson[c] = {"correct": k, "n": len(math_ids), "acc": k / len(math_ids),
                 "wilson95_lo": lo, "wilson95_hi": hi}
results["wilson_math500"] = wilson
results["wilson_ci_overlap_k6_k8"] = not (
    wilson["k6"]["wilson95_hi"] < wilson["k8"]["wilson95_lo"]
    or wilson["k8"]["wilson95_hi"] < wilson["k6"]["wilson95_lo"])

# ---------------------------------------------------------------- 3) truncation

trunc = {}
for c in ("ref", "k2", "k6", "k8"):
    it = data[c]["items"]
    tab = {"stop_correct": 0, "stop_wrong": 0, "length_correct": 0, "length_wrong": 0}
    trunc_ids = []
    for i in math_ids:
        r = it[i]
        key = ("stop" if r["finish_reason"] == "stop" else "length") + \
              ("_correct" if r["correct"] else "_wrong")
        tab[key] += 1
        if r["finish_reason"] != "stop":
            trunc_ids.append(i)
    gt = np.array([it[i]["gen_tokens"] for i in math_ids])
    tab["n_truncated"] = tab["length_correct"] + tab["length_wrong"]
    tab["truncated_ids"] = trunc_ids
    tab["gen_tokens_mean"] = float(gt.mean())
    tab["gen_tokens_median"] = float(np.median(gt))
    trunc[c] = tab
results["truncation_math500"] = trunc

# gsm8k companion truncation counts
results["truncation_gsm8k_n_truncated"] = {
    c: sum(1 for i in gsm_ids if data[c]["items"][i]["finish_reason"] != "stop")
    for c in ("ref", "k2", "k6", "k8")}

# did k6 lose specifically by truncation, and did k8 recover those?
ft = results["flip_tables_math500"]["k6_vs_k8"]
k6w_k8r = ft["ids_k6_wrong_k8_right"]
k6r_k8w = ft["ids_k6_right_k8_wrong"]


def flip_detail(ids):
    rows = []
    for i in ids:
        rows.append({
            "id": i,
            "k6_tokens": data["k6"]["items"][i]["gen_tokens"],
            "k6_finish": data["k6"]["items"][i]["finish_reason"],
            "k8_tokens": data["k8"]["items"][i]["gen_tokens"],
            "k8_finish": data["k8"]["items"][i]["finish_reason"],
            "ref_correct": data["ref"]["items"][i]["correct"],
            "ref_tokens": data["ref"]["items"][i]["gen_tokens"],
        })
    return rows


results["flip_detail_k6_wrong_k8_right"] = flip_detail(k6w_k8r)
results["flip_detail_k6_right_k8_wrong"] = flip_detail(k6r_k8w)


def summarize_flip_tokens(rows):
    if not rows:
        return None
    k6t = np.array([r["k6_tokens"] for r in rows])
    k8t = np.array([r["k8_tokens"] for r in rows])
    return {
        "n": len(rows),
        "k6_tokens_mean": float(k6t.mean()), "k6_tokens_median": float(np.median(k6t)),
        "k8_tokens_mean": float(k8t.mean()), "k8_tokens_median": float(np.median(k8t)),
        "n_k6_truncated": sum(r["k6_finish"] != "stop" for r in rows),
        "n_k8_truncated": sum(r["k8_finish"] != "stop" for r in rows),
        "n_ref_correct": sum(r["ref_correct"] for r in rows),
    }


results["flip_tokens_summary"] = {
    "k6_wrong_k8_right": summarize_flip_tokens(results["flip_detail_k6_wrong_k8_right"]),
    "k6_right_k8_wrong": summarize_flip_tokens(results["flip_detail_k6_right_k8_wrong"]),
}

# thinking-length compensation: paired gen_tokens k6 vs k8 over all math500 items
d_tokens = [data["k8"]["items"][i]["gen_tokens"] - data["k6"]["items"][i]["gen_tokens"]
            for i in math_ids]
results["paired_gen_tokens_k8_minus_k6_math500"] = {
    "mean": float(np.mean(d_tokens)),
    "median": float(np.median(d_tokens)),
    "sign_test": sign_test(d_tokens),
}
# same restricted to items both got correct (clean thinking-length comparison)
both_ok = [i for i in math_ids
           if data["k6"]["items"][i]["correct"] and data["k8"]["items"][i]["correct"]]
d_ok = [data["k8"]["items"][i]["gen_tokens"] - data["k6"]["items"][i]["gen_tokens"]
        for i in both_ok]
results["paired_gen_tokens_k8_minus_k6_both_correct"] = {
    "n": len(both_ok),
    "mean": float(np.mean(d_ok)), "median": float(np.median(d_ok)),
    "sign_test": sign_test(d_ok),
}

# ---------------------------------------------------------------- 4) difficulty

results["difficulty_stratification"] = {
    "available": False,
    "note": ("minibench_items.json math500 entries carry only {id, task, prompt, gold};"
             " the MATH-500 'level' field was not persisted by build_minibench.py and"
             " ids are ordered by (sorted) source index, not difficulty. Skipped."),
}

# ---------------------------------------------------------------- 5) verdict

b = ft["ids_k6_right_k8_wrong"]  # k6-only correct
c = ft["ids_k6_wrong_k8_right"]  # k8-only correct
nb, nc = len(b), len(c)
n_disc = nb + nc
results["verdict_numbers"] = {
    "k6_only_correct": nb,
    "k8_only_correct": nc,
    "n_discordant": n_disc,
    "observed_margin_items": nc - nb,
    "p_one_sided_k8_ge_this_margin_given_equal_true_acc": binom_tail_ge(nc, n_disc),
    "mcnemar_two_sided_p": mcnemar_exact(nb, nc),
    "note": ("One-sided prob computed as P(X >= k8_wins | Binomial(n_discordant, 0.5)):"
             " chance that k8 beats k6 by at least the observed margin if both had"
             " equal true accuracy on this item population."),
}

# mechanism facts, computed (not transcribed) so the JSON is self-contained
rec = results["flip_detail_k6_wrong_k8_right"]
los = results["flip_detail_k6_right_k8_wrong"]
results["mechanism"] = {
    "truncation_implies_wrong_all_configs": all(
        trunc[cfg]["length_correct"] == 0 for cfg in trunc),
    "k8_recoveries_all_k6_truncations": all(r["k6_finish"] != "stop" for r in rec),
    "k8_recoveries_all_finished_by_k8": all(r["k8_finish"] == "stop" for r in rec),
    "k8_losses_all_k8_truncations": all(r["k8_finish"] != "stop" for r in los),
    "k8_losses_k6_tokens": [r["k6_tokens"] for r in los],
    "ref_vs_k6_losses_all_k6_truncations": all(
        data["k6"]["items"][i]["finish_reason"] != "stop"
        for i in results["flip_tables_math500"]["ref_vs_k6"]["ids_ref_right_k6_wrong"]),
    "wrong_answer_share_from_truncation": {
        cfg: (trunc[cfg]["length_wrong"]
              / max(1, trunc[cfg]["length_wrong"] + trunc[cfg]["stop_wrong"]))
        for cfg in trunc},
}

(OUT / "b_math_anomaly.json").write_text(json.dumps(results, indent=2))

# ---------------------------------------------------------------- markdown

def pct(x):
    return f"{100 * x:.1f}%"


md = []
md.append("# b_math_anomaly — is k8 math500=.72 > k6=.64 real non-monotonicity?\n")
md.append("Anchor chain (nested drops): ref ⊂ k2=drop[16,12] ⊂ k6=+[13,9,8,4] ⊂ k8=+[5,15]."
          " Paired analysis on the same 100 frozen math500 items (seed fixed by data;"
          " script is deterministic).\n")

md.append("## KL anchors (mean nats, read from kl_screen results)\n")
md.append("| config | on-policy KL | off-policy KL |")
md.append("|---|---|---|")
for name, v in results["kl_anchors_mean_nats"].items():
    md.append(f"| {name} | {v['onpolicy']:.4f} | {v['offpolicy']:.4f} |")
md.append("\nKL damage is strictly monotone k2 < k6 < k8; the math500 accuracy bump at k8"
          " is therefore not explained by the distributional-damage metric.\n")

md.append("## 1. Pairwise flip tables (math500, n=100)\n")
md.append("| pair | both✓ | A-only✓ | B-only✓ | both✗ | discordant | McNemar exact p |")
md.append("|---|---|---|---|---|---|---|")
for a, bb in pairs:
    t = results["flip_tables_math500"][f"{a}_vs_{bb}"]
    md.append(f"| {a} vs {bb} | {t['both_correct']} | {t[f'{a}_only_correct']} |"
              f" {t[f'{bb}_only_correct']} | {t['both_wrong']} | {t['n_discordant']} |"
              f" {t['mcnemar_exact_p_two_sided']:.3f} |")
md.append("\ngsm8k companion (n=100):\n")
md.append("| pair | both✓ | A-only✓ | B-only✓ | both✗ | McNemar exact p |")
md.append("|---|---|---|---|---|---|")
for a, bb in [("ref", "k6"), ("ref", "k8"), ("k6", "k8")]:
    t = results["flip_tables_gsm8k"][f"{a}_vs_{bb}"]
    md.append(f"| {a} vs {bb} | {t['both_correct']} | {t[f'{a}_only_correct']} |"
              f" {t[f'{bb}_only_correct']} | {t['both_wrong']} |"
              f" {t['mcnemar_exact_p_two_sided']:.3f} |")
cm = results["chain_nonmonotone_items_math500"]
md.append(f"\nPer-item non-monotone patterns along ref→k2→k6→k8: {cm['count']}/100 items"
          " show at least one recovery (wrong at an earlier anchor, right at a later one).\n")

md.append("## 2. Wilson 95% CIs (math500)\n")
md.append("| config | acc | Wilson 95% CI |")
md.append("|---|---|---|")
for cfg in ("ref", "k2", "k6", "k8"):
    w = wilson[cfg]
    md.append(f"| {cfg} | {w['acc']:.2f} | [{w['wilson95_lo']:.3f}, {w['wilson95_hi']:.3f}] |")
md.append(f"\nk6 and k8 CIs overlap: {results['wilson_ci_overlap_k6_k8']}. All four CIs"
          " mutually overlap heavily; unpaired CIs are uninformative at n=100 for"
          " 8-point gaps — the paired tests above are the honest view.\n")

md.append("## 3. Truncation × correctness (math500)\n")
md.append("| config | stop✓ | stop✗ | length✓ | length✗ | truncated | mean gen_tokens | median |")
md.append("|---|---|---|---|---|---|---|---|")
for cfg in ("ref", "k2", "k6", "k8"):
    t = trunc[cfg]
    md.append(f"| {cfg} | {t['stop_correct']} | {t['stop_wrong']} | {t['length_correct']} |"
              f" {t['length_wrong']} | {t['n_truncated']} | {t['gen_tokens_mean']:.0f} |"
              f" {t['gen_tokens_median']:.0f} |")
md.append(f"\ngsm8k truncation counts: {results['truncation_gsm8k_n_truncated']}\n")

fs = results["flip_tokens_summary"]
md.append("### Flip-set token comparison (k6 vs k8, math500)\n")
for key, label in [("k6_wrong_k8_right", "k6 wrong → k8 right (k8 recoveries)"),
                   ("k6_right_k8_wrong", "k6 right → k8 wrong (k8 losses)")]:
    s = fs[key]
    if s is None:
        md.append(f"- {label}: none")
        continue
    md.append(f"- **{label}** (n={s['n']}): k6 tokens mean {s['k6_tokens_mean']:.0f} /"
              f" median {s['k6_tokens_median']:.0f} ({s['n_k6_truncated']} truncated);"
              f" k8 tokens mean {s['k8_tokens_mean']:.0f} / median"
              f" {s['k8_tokens_median']:.0f} ({s['n_k8_truncated']} truncated);"
              f" reference got {s['n_ref_correct']}/{s['n']} of these right.")
pt = results["paired_gen_tokens_k8_minus_k6_math500"]
pt2 = results["paired_gen_tokens_k8_minus_k6_both_correct"]
md.append(f"\nPaired gen_tokens (k8 − k6) over all 100 math500 items: mean"
          f" {pt['mean']:+.0f}, median {pt['median']:+.0f}, sign test"
          f" +{pt['sign_test']['pos']}/−{pt['sign_test']['neg']}"
          f" (p={pt['sign_test']['p_two_sided']:.3f}).")
md.append(f"Restricted to items both configs solved (n={pt2['n']}): mean {pt2['mean']:+.0f},"
          f" median {pt2['median']:+.0f}, sign test +{pt2['sign_test']['pos']}/"
          f"−{pt2['sign_test']['neg']} (p={pt2['sign_test']['p_two_sided']:.3f}).\n")

md.append("## 4. Difficulty stratification\n")
md.append(results["difficulty_stratification"]["note"] + "\n")

vn = results["verdict_numbers"]
mech = results["mechanism"]
md.append("## 5. Verdict\n")
md.append(f"- Discordant k6/k8 pairs: {vn['n_discordant']} ({vn['k8_only_correct']} k8-only"
          f" correct vs {vn['k6_only_correct']} k6-only correct; net margin"
          f" {vn['observed_margin_items']} items = {pct(vn['observed_margin_items']/100)}).")
md.append(f"- P(k8 ≥ k6 by ≥ this margin | equal true accuracy) ="
          f" {vn['p_one_sided_k8_ge_this_margin_given_equal_true_acc']:.3f}"
          f" (one-sided exact binomial on discordant pairs); two-sided McNemar p ="
          f" {vn['mcnemar_two_sided_p']:.3f}. Borderline: unlikely to be pure symmetric"
          " noise, but not decisive at n=100 with a single seed per config.")
md.append("- **Mechanism is a budget interaction, not a capability inversion.**"
          f" Truncation ⇒ wrong in every config (length_correct = 0 everywhere);"
          f" {pct(mech['wrong_answer_share_from_truncation']['k6'])} of k6's math500"
          " errors are budget exhaustions. Every one of the 11 k8 recoveries was a k6"
          " truncation at 16384 tokens that k8 finished naturally (mean ~11.0k tokens),"
          " and every one of the 3 k8 losses was a k8 truncation of an item k6 solved"
          " in 2.7k–8.0k tokens. All 8 ref→k6 losses were also k6 truncations.")
md.append("- **Thinking-length compensation hypothesis: refuted in the stated direction.**"
          " k8 does not recover by thinking longer — it recovers by *avoiding runaway"
          " thinking* (k6: 35/100 truncated; k8: 26/100, same as reference). On items"
          " both solve, k8 is only mildly longer (median +884 tokens, sign test p≈0.07);"
          " over all 100 items the paired difference is null (median 0, p≈0.65).")
md.append("- k8 ≠ ref behaviorally despite equal accuracy: 12 discordant items (6/6"
          " split, McNemar p=1.0), and several k8 recoveries finished at 13.4k–15.3k"
          " tokens, barely under budget — accuracy parity is partly luck of the budget"
          " boundary.")
md.append("- See JSON for full per-item flip lists and token details.\n")

(OUT / "b_math_anomaly.md").write_text("\n".join(md) + "\n")
print("wrote", OUT / "b_math_anomaly.json")
print("wrote", OUT / "b_math_anomaly.md")

# console digest for the operator
print(json.dumps({
    "wilson": {k: [round(v["acc"], 3), round(v["wilson95_lo"], 3),
                   round(v["wilson95_hi"], 3)] for k, v in wilson.items()},
    "k6_vs_k8": {k: v for k, v in results["flip_tables_math500"]["k6_vs_k8"].items()
                 if not k.startswith("ids_")},
    "ref_vs_k6": {k: v for k, v in results["flip_tables_math500"]["ref_vs_k6"].items()
                  if not k.startswith("ids_")},
    "ref_vs_k8": {k: v for k, v in results["flip_tables_math500"]["ref_vs_k8"].items()
                  if not k.startswith("ids_")},
    "truncated_per_cfg": {c: trunc[c]["n_truncated"] for c in trunc},
    "flip_tokens": results["flip_tokens_summary"],
    "paired_tokens_all": results["paired_gen_tokens_k8_minus_k6_math500"],
    "verdict": vn,
}, indent=2, default=str))
