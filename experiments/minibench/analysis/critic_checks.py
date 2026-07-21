#!/usr/bin/env python3
"""critic_checks — completeness-critic verification pass over the anchor mini-bench.

Checks (all pure-JSON, stdlib-only, deterministic):
  1. Data integrity: id sets/duplicates across configs + frozen items, dtypes,
     finish_reason domain, gen_tokens censoring at 16384, stored task_accuracy
     and macro_avg vs recomputation, macro-vs-micro divergence.
  2. Global 'truncation => wrong' invariant across all tasks/configs.
  3. Per task/config damage decomposition: accuracy = (1 - trunc_rate) *
     conditional accuracy on completed ('stop') items; McNemar vs ref both on
     all pairs and restricted to pairs where NEITHER run truncated
     (does damage survive once budget exhaustion is removed?).
  4. IFEval instruction-family breakdown (staircase cleanliness check) +
     truncation linkage of the k6/k8 IFEval losses.
  5. GSM8K k2=.92 > ref=.91 anomaly: flip mechanism (same class as math500?).
  6. KL anchors re-read from kl_screen files.
  7. Cross-analysis consistency spot checks against the four existing analyses.

Outputs: critic_checks.json, critic_checks.md (same directory).
"""
import json
import math
import os
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
ITEMS = os.path.join(HERE, "..", "minibench_items.json")
KL_ON = os.path.join(HERE, "..", "..", "kl_screen", "results", "topk_onpolicy.json")
KL_OFF = os.path.join(HERE, "..", "..", "kl_screen", "results", "topk_offpolicy.json")

CONFIGS = ["reference", "k2_16-12", "k6", "k8"]
SHORT = {"reference": "ref", "k2_16-12": "k2", "k6": "k6", "k8": "k8"}
TASKS = ["gsm8k", "math500", "ifeval", "mmlu"]
BUDGET = 16384

out = {}
md = []


def load(cfg):
    with open(os.path.join(RES, cfg + ".json")) as f:
        return json.load(f)


def to_bool(x):
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        return x.strip().lower() in ("true", "1", "yes")
    return bool(x)


def to_int(x):
    return int(x)


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def mcnemar_exact(b, c):
    """Two-sided exact binomial on discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


data = {cfg: load(cfg) for cfg in CONFIGS}
frozen = json.load(open(ITEMS))["items"]

# ---------------- 1. integrity ----------------
integ = {}
frozen_ids = [it["id"] for it in frozen]
integ["frozen_n"] = len(frozen_ids)
integ["frozen_dup_ids"] = [i for i, c in Counter(frozen_ids).items() if c > 1]
frozen_task = {it["id"]: it["task"] for it in frozen}

dtypes = {}
id_sets = {}
for cfg in CONFIGS:
    items = data[cfg]["items"]
    ids = [it["id"] for it in items]
    id_sets[cfg] = set(ids)
    dtypes[cfg] = {
        "n_items": len(items),
        "dup_ids": [i for i, c in Counter(ids).items() if c > 1],
        "correct_types": sorted({type(it["correct"]).__name__ for it in items}),
        "gen_tokens_types": sorted({type(it["gen_tokens"]).__name__ for it in items}),
        "finish_reasons": sorted({it["finish_reason"] for it in items}),
        "task_mismatch_vs_frozen": sum(
            1 for it in items if frozen_task.get(it["id"]) != it["task"]
        ),
        "ids_match_frozen": set(ids) == set(frozen_ids),
        "id_order_equals_frozen_order": ids == frozen_ids,
    }
integ["dtypes"] = dtypes
integ["id_sets_identical_across_configs"] = all(
    id_sets[c] == id_sets[CONFIGS[0]] for c in CONFIGS
)

# expected id pattern
expected = (
    [f"gsm8k-{i:03d}" for i in range(100)]
    + [f"math500-{i:03d}" for i in range(100)]
    + [f"ifeval-{i:03d}" for i in range(100)]
    + [f"mmlu-{i:03d}" for i in range(200)]
)
integ["frozen_ids_match_expected_pattern"] = set(frozen_ids) == set(expected)

# gen_tokens censoring / finish_reason coupling
cens = {}
for cfg in CONFIGS:
    items = data[cfg]["items"]
    length_not_16384 = [
        it["id"] for it in items
        if it["finish_reason"] != "stop" and to_int(it["gen_tokens"]) != BUDGET
    ]
    stop_at_16384 = [
        it["id"] for it in items
        if it["finish_reason"] == "stop" and to_int(it["gen_tokens"]) >= BUDGET
    ]
    nonpos = [it["id"] for it in items if to_int(it["gen_tokens"]) <= 0]
    over = [it["id"] for it in items if to_int(it["gen_tokens"]) > BUDGET]
    cens[SHORT[cfg]] = {
        "length_items_not_at_budget": length_not_16384,
        "stop_items_at_or_over_budget": stop_at_16384,
        "nonpositive_gen_tokens": nonpos,
        "over_budget": over,
    }
integ["censoring"] = cens

# stored vs recomputed accuracies
acc_check = {}
for cfg in CONFIGS:
    items = data[cfg]["items"]
    per_task = defaultdict(lambda: [0, 0])
    for it in items:
        per_task[it["task"]][0] += to_bool(it["correct"])
        per_task[it["task"]][1] += 1
    recomputed = {t: per_task[t][0] / per_task[t][1] for t in TASKS}
    macro = sum(recomputed[t] for t in TASKS) / 4
    micro = sum(per_task[t][0] for t in TASKS) / sum(per_task[t][1] for t in TASKS)
    stored_task = data[cfg]["task_accuracy"]
    acc_check[SHORT[cfg]] = {
        "stored_task_accuracy": stored_task,
        "recomputed_task_accuracy": recomputed,
        "task_accuracy_matches": all(
            abs(stored_task[t] - recomputed[t]) < 1e-12 for t in TASKS
        ),
        "stored_macro": data[cfg]["macro_avg"],
        "recomputed_macro": macro,
        "macro_matches": abs(data[cfg]["macro_avg"] - macro) < 1e-12,
        "micro_avg": micro,
        "macro_minus_micro_pts": round(100 * (macro - micro), 2),
    }
integ["accuracy_recomputation"] = acc_check
out["integrity"] = integ

# pair items by id
by_id = {
    SHORT[cfg]: {it["id"]: it for it in data[cfg]["items"]} for cfg in CONFIGS
}
ids_by_task = defaultdict(list)
for iid in expected:
    ids_by_task[frozen_task[iid]].append(iid)

# ---------------- 2. truncation => wrong, globally ----------------
trunc_correct = []
for cfg in CONFIGS:
    for it in data[cfg]["items"]:
        if it["finish_reason"] != "stop" and to_bool(it["correct"]):
            trunc_correct.append({"config": SHORT[cfg], "id": it["id"]})
out["truncation_implies_wrong"] = {
    "violations": trunc_correct,
    "holds_globally": len(trunc_correct) == 0,
}

# ---------------- 3. damage decomposition per task/config ----------------
decomp = {}
for t in TASKS:
    decomp[t] = {}
    ref = by_id["ref"]
    for s in ["ref", "k2", "k6", "k8"]:
        rows = [by_id[s][i] for i in ids_by_task[t]]
        n = len(rows)
        ncorr = sum(to_bool(r["correct"]) for r in rows)
        ntr = sum(r["finish_reason"] != "stop" for r in rows)
        nstop = n - ntr
        ncorr_stop = sum(
            to_bool(r["correct"]) for r in rows if r["finish_reason"] == "stop"
        )
        cond = ncorr_stop / nstop if nstop else float("nan")
        lo, hi = wilson(ncorr_stop, nstop)
        decomp[t][s] = {
            "n": n,
            "acc": ncorr / n,
            "trunc": ntr,
            "trunc_rate": ntr / n,
            "cond_acc_on_stop": cond,
            "cond_acc_wilson95": [lo, hi],
        }
    # paired vs ref: full McNemar and both-stop-restricted McNemar
    for s in ["k2", "k6", "k8"]:
        b = c = 0            # all pairs
        bs_b = bs_c = 0      # both-stop pairs
        n_bothstop = 0
        loss_cfg_trunc = 0   # cfg-only-wrong flips where cfg truncated
        gain_ref_trunc = 0   # cfg-only-right flips where ref truncated
        for i in ids_by_task[t]:
            r, k = by_id["ref"][i], by_id[s][i]
            rc, kc = to_bool(r["correct"]), to_bool(k["correct"])
            if rc and not kc:
                b += 1
                if k["finish_reason"] != "stop":
                    loss_cfg_trunc += 1
            elif kc and not rc:
                c += 1
                if r["finish_reason"] != "stop":
                    gain_ref_trunc += 1
            if r["finish_reason"] == "stop" and k["finish_reason"] == "stop":
                n_bothstop += 1
                if rc and not kc:
                    bs_b += 1
                elif kc and not rc:
                    bs_c += 1
        decomp[t][s].update({
            "mcnemar_vs_ref": {"b_ref_only": b, "c_cfg_only": c,
                               "p": mcnemar_exact(b, c)},
            "losses_that_are_cfg_truncations": [loss_cfg_trunc, b],
            "gains_that_are_ref_truncations": [gain_ref_trunc, c],
            "both_stop_pairs": n_bothstop,
            "mcnemar_both_stop_only": {"b": bs_b, "c": bs_c,
                                       "p": mcnemar_exact(bs_b, bs_c)},
            "delta_acc_pts": round(100 * (decomp[t][s]["acc"] - decomp[t]["ref"]["acc"]), 1),
            "delta_cond_acc_pts": round(100 * (decomp[t][s]["cond_acc_on_stop"]
                                               - decomp[t]["ref"]["cond_acc_on_stop"]), 1),
            "delta_trunc_pts": round(100 * (decomp[t][s]["trunc_rate"]
                                            - decomp[t]["ref"]["trunc_rate"]), 1),
        })
out["damage_decomposition"] = decomp

# ---------------- 4. IFEval instruction-family analysis ----------------
ife_items = {it["id"]: it for it in frozen if it["task"] == "ifeval"}
fam_of = {}
nins_of = {}
for iid, it in ife_items.items():
    fams = sorted({x.split(":")[0] for x in it["instruction_id_list"]})
    fam_of[iid] = fams
    nins_of[iid] = len(it["instruction_id_list"])

fam_counts = Counter()
for iid in ife_items:
    for f in fam_of[iid]:
        fam_counts[f] += 1

ife = {"n_items": len(ife_items),
       "n_instructions_per_item": dict(Counter(nins_of.values())),
       "family_item_counts": dict(fam_counts)}

fam_table = {}
for fam in sorted(fam_counts):
    iids = [i for i in ids_by_task["ifeval"] if fam in fam_of[i]]
    row = {"n": len(iids)}
    for s in ["ref", "k2", "k6", "k8"]:
        row[s] = sum(to_bool(by_id[s][i]["correct"]) for i in iids)
    # paired k8 vs ref inside family
    b = sum(1 for i in iids
            if to_bool(by_id["ref"][i]["correct"]) and not to_bool(by_id["k8"][i]["correct"]))
    c = sum(1 for i in iids
            if to_bool(by_id["k8"][i]["correct"]) and not to_bool(by_id["ref"][i]["correct"]))
    row["k8_vs_ref_b_c"] = [b, c]
    row["k8_vs_ref_p"] = mcnemar_exact(b, c)
    row["monotone_ref_ge_k2_ge_k6_ge_k8"] = (
        row["ref"] >= row["k2"] >= row["k6"] >= row["k8"]
    )
    fam_table[fam] = row
ife["family_table"] = fam_table
ife["families_monotone"] = sum(
    1 for r in fam_table.values() if r["monotone_ref_ge_k2_ge_k6_ge_k8"]
)

# multi-instruction items: are they hit harder at k8?
multi = [i for i in ids_by_task["ifeval"] if nins_of[i] >= 2]
single = [i for i in ids_by_task["ifeval"] if nins_of[i] == 1]
def _pd(iids, s):
    b = sum(1 for i in iids
            if to_bool(by_id["ref"][i]["correct"]) and not to_bool(by_id[s][i]["correct"]))
    c = sum(1 for i in iids
            if to_bool(by_id[s][i]["correct"]) and not to_bool(by_id["ref"][i]["correct"]))
    return (c - b) / len(iids), b, c
for s in ["k2", "k6", "k8"]:
    dm, bm, cm = _pd(multi, s)
    ds, bs2, cs2 = _pd(single, s)
    ife[f"{s}_paired_delta_multi_n{len(multi)}"] = round(dm, 4)
    ife[f"{s}_paired_delta_single_n{len(single)}"] = round(ds, 4)

# truncation linkage of IFEval losses vs ref
for s in ["k2", "k6", "k8"]:
    losses = [i for i in ids_by_task["ifeval"]
              if to_bool(by_id["ref"][i]["correct"]) and not to_bool(by_id[s][i]["correct"])]
    tr = [i for i in losses if by_id[s][i]["finish_reason"] != "stop"]
    fams = Counter(f for i in losses for f in fam_of[i])
    ife[f"{s}_losses"] = {"n": len(losses), "cfg_truncated": len(tr),
                          "loss_ids": losses, "families": dict(fams)}
out["ifeval"] = ife

# ---------------- 5. GSM8K k2 anomaly ----------------
g = {"acc": {s: sum(to_bool(by_id[s][i]["correct"]) for i in ids_by_task["gsm8k"]) / 100
             for s in ["ref", "k2", "k6", "k8"]}}
flips = {"k2_recoveries": [], "k2_losses": []}
for i in ids_by_task["gsm8k"]:
    r, k = by_id["ref"][i], by_id["k2"][i]
    rc, kc = to_bool(r["correct"]), to_bool(k["correct"])
    rec = {"id": i,
           "ref_tokens": to_int(r["gen_tokens"]), "ref_finish": r["finish_reason"],
           "k2_tokens": to_int(k["gen_tokens"]), "k2_finish": k["finish_reason"]}
    if kc and not rc:
        flips["k2_recoveries"].append(rec)
    elif rc and not kc:
        flips["k2_losses"].append(rec)
g.update(flips)
g["k2_recoveries_ref_truncated"] = sum(
    1 for r in flips["k2_recoveries"] if r["ref_finish"] != "stop")
g["k2_losses_k2_truncated"] = sum(
    1 for r in flips["k2_losses"] if r["k2_finish"] != "stop")
b, c = len(flips["k2_losses"]), len(flips["k2_recoveries"])
g["mcnemar_k2_vs_ref"] = {"b": b, "c": c, "p": mcnemar_exact(b, c)}
# same-class question: is gsm8k damage truncation-mediated at k8 too?
losses_k8 = [i for i in ids_by_task["gsm8k"]
             if to_bool(by_id["ref"][i]["correct"]) and not to_bool(by_id["k8"][i]["correct"])]
g["k8_losses_n"] = len(losses_k8)
g["k8_losses_k8_truncated"] = sum(
    1 for i in losses_k8 if by_id["k8"][i]["finish_reason"] != "stop")
out["gsm8k_k2_anomaly"] = g

# ---------------- 6. KL anchors ----------------
kl = {}
for name, path in [("onpolicy", KL_ON), ("offpolicy", KL_OFF)]:
    d = json.load(open(path))
    kl[name] = {c["name"]: c["mean_kl_nats"] for c in d["candidates"]}
out["kl_anchors"] = kl

# ---------------- 7. cross-analysis consistency spot checks ----------------
cons = {}
# (a) a_mmlu_tail headline: k8 vs ref mmlu b=50 c=13
mm = decomp["mmlu"]["k8"]["mcnemar_vs_ref"]
cons["a_mmlu_k8_b50_c13"] = (mm["b_ref_only"] == 50 and mm["c_cfg_only"] == 13)
# (b) b_math headline: k6 vs k8 discordants 3/11
b_ = sum(1 for i in ids_by_task["math500"]
         if to_bool(by_id["k6"][i]["correct"]) and not to_bool(by_id["k8"][i]["correct"]))
c_ = sum(1 for i in ids_by_task["math500"]
         if to_bool(by_id["k8"][i]["correct"]) and not to_bool(by_id["k6"][i]["correct"]))
cons["b_math_k6k8_discordants_3_11"] = (b_ == 3 and c_ == 11)
# (c) c headline truncation counts
tc = {s: sum(1 for it in data[cfg]["items"] if it["finish_reason"] != "stop")
      for cfg, s in SHORT.items()}
cons["c_overall_trunc_counts"] = tc
cons["c_trunc_counts_match_45_53_65_66"] = (
    tc == {"ref": 45, "k2": 53, "k6": 65, "k8": 66})
# (d) d anchor accuracies vs stored
cons["d_anchor_macros"] = {SHORT[cfg]: data[cfg]["macro_avg"] for cfg in CONFIGS}
# (e) b-vs-c reconciliation on math500: mean tokens rise k6->k8 while truncation falls
m_k6 = [to_int(by_id["k6"][i]["gen_tokens"]) for i in ids_by_task["math500"]]
m_k8 = [to_int(by_id["k8"][i]["gen_tokens"]) for i in ids_by_task["math500"]]
both_stop = [i for i in ids_by_task["math500"]
             if by_id["k6"][i]["finish_reason"] == "stop"
             and by_id["k8"][i]["finish_reason"] == "stop"]
d68 = [to_int(by_id["k8"][i]["gen_tokens"]) - to_int(by_id["k6"][i]["gen_tokens"])
       for i in both_stop]
d68s = sorted(d68)
n = len(d68s)
med = (d68s[n // 2] if n % 2 else (d68s[n // 2 - 1] + d68s[n // 2]) / 2)
cons["math500_mean_tokens_k6_k8"] = [sum(m_k6) / 100, sum(m_k8) / 100]
cons["math500_trunc_k6_k8"] = [decomp["math500"]["k6"]["trunc"],
                               decomp["math500"]["k8"]["trunc"]]
cons["math500_bothstop_k8_minus_k6_tokens"] = {
    "n": n, "mean": sum(d68) / n, "median": med}
out["consistency"] = cons

# ---------------- 8. budget-dependence of the headline damage ----------------
bud = {}
for s in ["k2", "k6", "k8"]:
    b_all = c_all = 0
    b_stop = c_stop = 0
    loss_trunc = 0
    for i in expected:
        r, k = by_id["ref"][i], by_id[s][i]
        rc, kc = to_bool(r["correct"]), to_bool(k["correct"])
        if rc and not kc:
            b_all += 1
            if k["finish_reason"] != "stop":
                loss_trunc += 1
        elif kc and not rc:
            c_all += 1
        if r["finish_reason"] == "stop" and k["finish_reason"] == "stop":
            if rc and not kc:
                b_stop += 1
            elif kc and not rc:
                c_stop += 1
    bud[s] = {
        "regressions_b": b_all, "recoveries_c": c_all,
        "regressions_that_are_cfg_truncations": loss_trunc,
        "trunc_share_of_regressions": round(loss_trunc / b_all, 3) if b_all else None,
        "both_stop_mcnemar": {"b": b_stop, "c": c_stop,
                              "p": mcnemar_exact(b_stop, c_stop)},
    }
# ifeval k8 truncations by instruction multiplicity
k8_ife_trunc = [i for i in ids_by_task["ifeval"]
                if by_id["k8"][i]["finish_reason"] != "stop"]
bud["k8_ifeval_trunc_by_multiplicity"] = dict(
    Counter(nins_of[i] for i in k8_ife_trunc))
bud["ifeval_multiplicity_base_rates"] = dict(Counter(nins_of.values()))
out["budget_dependence"] = bud

# ---------------- markdown ----------------
md.append("# critic_checks — completeness pass over the four anchor analyses")
md.append("")
md.append("## 1. Integrity")
md.append(f"- id sets identical across 4 configs: {integ['id_sets_identical_across_configs']}; "
          f"match frozen items: {all(d['ids_match_frozen'] for d in dtypes.values())}; "
          f"duplicates: {sum(len(d['dup_ids']) for d in dtypes.values())}; "
          f"expected id pattern: {integ['frozen_ids_match_expected_pattern']}")
for cfg in CONFIGS:
    d = dtypes[cfg]
    md.append(f"- {SHORT[cfg]}: n={d['n_items']}, correct={d['correct_types']}, "
              f"gen_tokens={d['gen_tokens_types']}, finish={d['finish_reasons']}, "
              f"order==frozen: {d['id_order_equals_frozen_order']}")
md.append("")
md.append("| config | task_acc matches | macro matches | stored macro | micro | macro-micro (pts) |")
md.append("|---|---|---|---|---|---|")
for s in ["ref", "k2", "k6", "k8"]:
    a = acc_check[s]
    md.append(f"| {s} | {a['task_accuracy_matches']} | {a['macro_matches']} | "
              f"{a['stored_macro']:.4f} | {a['micro_avg']:.4f} | {a['macro_minus_micro_pts']} |")
md.append("")
md.append(f"Censoring: violations of (length <=> gen_tokens==16384): " + json.dumps(
    {k: v for k, v in cens.items()
     if any(v[q] for q in v)}) if any(any(v[q] for q in v) for v in cens.values())
    else "Censoring: every 'length' item has gen_tokens==16384 and no 'stop' item reaches budget; no nonpositive/over-budget token counts.")
md.append("")
md.append("## 2. truncation => wrong")
md.append(f"- Holds globally across all 2000 (config,item) pairs: "
          f"{out['truncation_implies_wrong']['holds_globally']} "
          f"(violations: {len(trunc_correct)})")
md.append("")
md.append("## 3. Damage decomposition (acc = (1-trunc)*cond_acc_on_stop)")
md.append("")
md.append("| task | cfg | acc | Δacc | trunc% | Δtrunc | cond acc (stop) | Δcond | McNemar all (b,c,p) | McNemar both-stop (b,c,p) | losses trunc/total |")
md.append("|---|---|---|---|---|---|---|---|---|---|---|")
for t in TASKS:
    for s in ["ref", "k2", "k6", "k8"]:
        r = decomp[t][s]
        if s == "ref":
            md.append(f"| {t} | ref | {r['acc']:.3f} | — | {100*r['trunc_rate']:.1f} | — | "
                      f"{r['cond_acc_on_stop']:.3f} | — | — | — | — |")
        else:
            m1, m2 = r["mcnemar_vs_ref"], r["mcnemar_both_stop_only"]
            lt = r["losses_that_are_cfg_truncations"]
            md.append(
                f"| {t} | {s} | {r['acc']:.3f} | {r['delta_acc_pts']:+.1f} | "
                f"{100*r['trunc_rate']:.1f} | {r['delta_trunc_pts']:+.1f} | "
                f"{r['cond_acc_on_stop']:.3f} | {r['delta_cond_acc_pts']:+.1f} | "
                f"({m1['b_ref_only']},{m1['c_cfg_only']},{m1['p']:.3g}) | "
                f"({m2['b']},{m2['c']},{m2['p']:.3g}) | {lt[0]}/{lt[1]} |")
md.append("")
md.append("## 4. IFEval instruction families")
md.append(f"- items with 1 instruction: {sum(1 for v in nins_of.values() if v==1)}, "
          f">=2: {sum(1 for v in nins_of.values() if v>=2)} "
          f"(max {max(nins_of.values())})")
md.append("")
md.append("| family | n | ref | k2 | k6 | k8 | monotone? | k8 vs ref (b,c,p) |")
md.append("|---|---|---|---|---|---|---|---|")
for fam, r in sorted(fam_table.items()):
    md.append(f"| {fam} | {r['n']} | {r['ref']} | {r['k2']} | {r['k6']} | {r['k8']} | "
              f"{'Y' if r['monotone_ref_ge_k2_ge_k6_ge_k8'] else 'N'} | "
              f"({r['k8_vs_ref_b_c'][0]},{r['k8_vs_ref_b_c'][1]},{r['k8_vs_ref_p']:.3g}) |")
md.append("")
md.append(f"- families monotone along the chain: {ife['families_monotone']}/{len(fam_table)} "
          f"(ALL n<=~25: individually underpowered)")
for s in ["k2", "k6", "k8"]:
    L = ife[f"{s}_losses"]
    md.append(f"- {s} losses vs ref: {L['n']}, of which {L['cfg_truncated']} are {s} "
              f"budget truncations; families: {L['families']}")
md.append(f"- paired delta multi-instruction (n={len(multi)}) vs single (n={len(single)}): "
          + ", ".join(f"{s}: {ife[f'{s}_paired_delta_multi_n{len(multi)}']:+.3f} vs "
                      f"{ife[f'{s}_paired_delta_single_n{len(single)}']:+.3f}"
                      for s in ["k2", "k6", "k8"]))
md.append("")
md.append("## 5. GSM8K k2 anomaly")
md.append(f"- acc: {g['acc']}")
md.append(f"- k2 vs ref: b={b}, c={c}, exact p={g['mcnemar_k2_vs_ref']['p']:.3g}")
md.append(f"- recoveries with ref truncated: {g['k2_recoveries_ref_truncated']}/{c}; "
          f"losses with k2 truncated: {g['k2_losses_k2_truncated']}/{b}")
md.append(f"- flip details: {json.dumps(flips)}")
md.append(f"- k8 gsm8k losses truncation-linked: {g['k8_losses_k8_truncated']}/{g['k8_losses_n']}")
md.append("")
md.append("## 5b. Budget dependence of headline damage (all 500 items)")
md.append("")
md.append("| config | regressions b | trunc-mediated | share | recoveries c | both-stop McNemar (b,c,p) |")
md.append("|---|---|---|---|---|---|")
for s in ["k2", "k6", "k8"]:
    r = bud[s]
    m = r["both_stop_mcnemar"]
    md.append(f"| {s} | {r['regressions_b']} | "
              f"{r['regressions_that_are_cfg_truncations']} | "
              f"{r['trunc_share_of_regressions']} | {r['recoveries_c']} | "
              f"({m['b']},{m['c']},{m['p']:.3g}) |")
md.append("")
md.append(f"- k8 ifeval truncations by #instructions: "
          f"{bud['k8_ifeval_trunc_by_multiplicity']} "
          f"(base rates {bud['ifeval_multiplicity_base_rates']})")
md.append("")
md.append("## 6. KL anchors (re-read)")
md.append(f"- on-policy: {json.dumps(kl['onpolicy'])}")
md.append(f"- off-policy: {json.dumps(kl['offpolicy'])}")
md.append("")
md.append("## 7. Consistency spot checks")
for k, v in cons.items():
    md.append(f"- {k}: {v}")
md.append("")

with open(os.path.join(HERE, "critic_checks.json"), "w") as f:
    json.dump(out, f, indent=1, default=str)
with open(os.path.join(HERE, "critic_checks.md"), "w") as f:
    f.write("\n".join(md) + "\n")
print("\n".join(md))
