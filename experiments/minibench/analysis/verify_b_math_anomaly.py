#!/usr/bin/env python3
"""Independent re-derivation of every key number in b_math_anomaly.

Written from the raw JSONs only (results/*.json, kl_screen results, frozen items).
Deterministic, stdlib-only exact tests (math.comb). Pairs items BY ID, asserts
identical id sets across all four configs, and coerces dtypes defensively.

Outputs: verify_b_math_anomaly.json / verify_b_math_anomaly.md
"""
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # repo root
RES = ROOT / "experiments" / "minibench" / "results"
KL = ROOT / "experiments" / "kl_screen" / "results"
OUT = Path(__file__).with_suffix("")  # base path, add .json/.md

CONFIGS = {
    "ref": "reference.json",
    "k2": "k2_16-12.json",
    "k6": "k6.json",
    "k8": "k8.json",
}
EXPECTED_DROPS = {
    "ref": None,
    "k2": "16,12",
    "k6": "16,12,13,9,8,4",
    "k8": "16,12,13,9,8,4,5,15",
}
BUDGET = 16384


def as_bool(x):
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        s = x.strip().lower()
        if s in ("true", "1", "yes"):
            return True
        if s in ("false", "0", "no"):
            return False
        raise ValueError(f"unparseable correct value: {x!r}")
    raise TypeError(f"unexpected correct dtype: {type(x)}")


def as_int(x):
    if isinstance(x, bool):
        raise TypeError("bool where int expected")
    if isinstance(x, int):
        return x
    if isinstance(x, str):
        return int(x.strip())
    raise TypeError(f"unexpected gen_tokens dtype: {type(x)}")


def load_items(fname):
    d = json.loads((RES / fname).read_text())
    assert d["max_tokens"] == BUDGET, (fname, d["max_tokens"])
    by_id = {}
    for it in d["items"]:
        iid = it["id"]
        assert iid not in by_id, f"duplicate id {iid} in {fname}"
        by_id[iid] = {
            "task": it["task"],
            "correct": as_bool(it["correct"]),
            "gen_tokens": as_int(it["gen_tokens"]),
            "finish": it["finish_reason"],
        }
    return d, by_id


def binom_sf(k, n):
    """P(X >= k) for X ~ Binomial(n, 0.5), exact."""
    return sum(math.comb(n, j) for j in range(k, n + 1)) / 2 ** n


def binom_cdf(k, n):
    """P(X <= k) for X ~ Binomial(n, 0.5), exact."""
    return sum(math.comb(n, j) for j in range(0, k + 1)) / 2 ** n


def mcnemar_exact_two_sided(b, c):
    """Two-sided exact McNemar: 2 * min tail of Binomial(b+c, .5), capped at 1."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2.0 * binom_cdf(k, n))


def sign_test_two_sided(pos, neg):
    n = pos + neg
    if n == 0:
        return 1.0
    return min(1.0, 2.0 * binom_cdf(min(pos, neg), n))


def wilson95(k, n, z=1.96):
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return center - half, center + half


def flip_table(a, b, ids):
    both = aonly = bonly = neither = 0
    a_only_ids, b_only_ids = [], []
    for i in ids:
        ca, cb = a[i]["correct"], b[i]["correct"]
        if ca and cb:
            both += 1
        elif ca and not cb:
            aonly += 1
            a_only_ids.append(i)
        elif cb and not ca:
            bonly += 1
            b_only_ids.append(i)
        else:
            neither += 1
    return {
        "both_correct": both,
        "a_only": aonly,
        "b_only": bonly,
        "both_wrong": neither,
        "n_discordant": aonly + bonly,
        "mcnemar_p": mcnemar_exact_two_sided(aonly, bonly),
        "a_only_ids": sorted(a_only_ids),
        "b_only_ids": sorted(b_only_ids),
    }


def main():
    raw = {}
    data = {}
    for cfg, fname in CONFIGS.items():
        raw[cfg], data[cfg] = load_items(fname)
        assert raw[cfg]["drop"] == EXPECTED_DROPS[cfg], (cfg, raw[cfg]["drop"])

    # --- id-set identity across configs and vs frozen items ---
    id_sets = {cfg: set(d.keys()) for cfg, d in data.items()}
    ref_ids = id_sets["ref"]
    for cfg, s in id_sets.items():
        assert s == ref_ids, f"id set mismatch: {cfg}"
    frozen = json.loads(
        (ROOT / "experiments" / "minibench" / "minibench_items.json").read_text()
    )
    frozen_items = frozen["items"] if isinstance(frozen, dict) else frozen
    frozen_ids = {it["id"] for it in frozen_items}
    assert frozen_ids == ref_ids, "results ids != frozen item ids"
    assert len(ref_ids) == 500

    math_ids = sorted(i for i in ref_ids if data["ref"][i]["task"] == "math500")
    gsm_ids = sorted(i for i in ref_ids if data["ref"][i]["task"] == "gsm8k")
    assert len(math_ids) == 100 and len(gsm_ids) == 100

    out = {"n_math500": len(math_ids), "n_gsm8k": len(gsm_ids)}

    # --- accuracies + Wilson CIs (math500) ---
    wilson = {}
    for cfg in CONFIGS:
        k = sum(data[cfg][i]["correct"] for i in math_ids)
        lo, hi = wilson95(k, 100)
        wilson[cfg] = {"correct": k, "acc": k / 100, "wilson_lo": lo, "wilson_hi": hi}
    out["wilson_math500"] = wilson
    out["all_cis_overlap_k6_k8"] = (
        wilson["k6"]["wilson_hi"] > wilson["k8"]["wilson_lo"]
        and wilson["k8"]["wilson_hi"] > wilson["k6"]["wilson_lo"]
    )

    # cross-check vs task_accuracy fields recorded in the result files
    out["task_accuracy_recorded"] = {
        cfg: raw[cfg]["task_accuracy"] for cfg in CONFIGS
    }
    for cfg in CONFIGS:
        assert abs(raw[cfg]["task_accuracy"]["math500"] - wilson[cfg]["acc"]) < 1e-12

    # --- pairwise flip tables (math500) ---
    pairs = [("ref", "k6"), ("ref", "k8"), ("k6", "k8"), ("ref", "k2"),
             ("k2", "k6"), ("k2", "k8")]
    ft_math = {}
    for a, b in pairs:
        ft_math[f"{a}_vs_{b}"] = flip_table(data[a], data[b], math_ids)
    out["flip_tables_math500"] = ft_math

    # one-sided exact binomial for k6 vs k8 direction
    t = ft_math["k6_vs_k8"]
    out["k6_vs_k8_one_sided_p"] = binom_sf(t["b_only"], t["n_discordant"])

    # --- gsm8k companion ---
    ft_gsm = {}
    for a, b in [("ref", "k6"), ("ref", "k8"), ("k6", "k8")]:
        ft_gsm[f"{a}_vs_{b}"] = flip_table(data[a], data[b], gsm_ids)
    out["flip_tables_gsm8k"] = ft_gsm

    # --- truncation x correctness (math500) ---
    trunc = {}
    for cfg in CONFIGS:
        cells = {"stop_correct": 0, "stop_wrong": 0,
                 "length_correct": 0, "length_wrong": 0}
        trunc_ids = []
        toks = []
        for i in math_ids:
            it = data[cfg][i]
            key = ("stop" if it["finish"] == "stop" else "length") + (
                "_correct" if it["correct"] else "_wrong")
            cells[key] += 1
            if it["finish"] == "length":
                trunc_ids.append(i)
                # sanity: truncated generations should sit at the budget
                assert it["gen_tokens"] == BUDGET, (cfg, i, it["gen_tokens"])
            toks.append(it["gen_tokens"])
        n_wrong = cells["stop_wrong"] + cells["length_wrong"]
        trunc[cfg] = {
            **cells,
            "n_truncated": len(trunc_ids),
            "truncated_ids": sorted(trunc_ids),
            "wrong_share_from_truncation": (
                cells["length_wrong"] / n_wrong if n_wrong else None),
            "gen_tokens_mean": statistics.mean(toks),
            "gen_tokens_median": statistics.median(toks),
        }
    out["truncation_math500"] = trunc
    out["length_correct_zero_everywhere"] = all(
        trunc[c]["length_correct"] == 0 for c in CONFIGS)

    out["truncation_gsm8k"] = {
        cfg: sum(1 for i in gsm_ids if data[cfg][i]["finish"] == "length")
        for cfg in CONFIGS
    }

    # --- k8 recoveries / losses vs k6, with token + finish detail ---
    rec_ids = ft_math["k6_vs_k8"]["b_only_ids"]  # k6 wrong, k8 right
    loss_ids = ft_math["k6_vs_k8"]["a_only_ids"]  # k6 right, k8 wrong

    def detail(ids):
        rows = []
        for i in sorted(ids):
            rows.append({
                "id": i,
                "k6_tokens": data["k6"][i]["gen_tokens"],
                "k6_finish": data["k6"][i]["finish"],
                "k8_tokens": data["k8"][i]["gen_tokens"],
                "k8_finish": data["k8"][i]["finish"],
                "ref_correct": data["ref"][i]["correct"],
                "ref_tokens": data["ref"][i]["gen_tokens"],
            })
        return rows

    rec = detail(rec_ids)
    loss = detail(loss_ids)
    out["k8_recoveries"] = {
        "n": len(rec),
        "all_k6_truncated_at_budget": all(
            r["k6_finish"] == "length" and r["k6_tokens"] == BUDGET for r in rec),
        "all_k8_finished_naturally": all(r["k8_finish"] == "stop" for r in rec),
        "k8_tokens_mean": statistics.mean(r["k8_tokens"] for r in rec),
        "k8_tokens_median": statistics.median(r["k8_tokens"] for r in rec),
        "n_ref_correct": sum(r["ref_correct"] for r in rec),
        "detail": rec,
    }
    out["k8_losses"] = {
        "n": len(loss),
        "all_k8_truncated_at_budget": all(
            r["k8_finish"] == "length" and r["k8_tokens"] == BUDGET for r in loss),
        "all_k6_finished_naturally": all(r["k6_finish"] == "stop" for r in loss),
        "k6_tokens_min": min(r["k6_tokens"] for r in loss),
        "k6_tokens_max": max(r["k6_tokens"] for r in loss),
        "n_ref_correct": sum(r["ref_correct"] for r in loss),
        "detail": loss,
    }
    # interpretation claim: all 8 ref->k6 losses are k6 truncations
    ref_k6_loss_ids = ft_math["ref_vs_k6"]["a_only_ids"]
    out["ref_vs_k6_losses_all_k6_truncations"] = all(
        data["k6"][i]["finish"] == "length" for i in ref_k6_loss_ids)

    # --- paired gen_tokens k8 - k6 (math500) ---
    diffs_all = [data["k8"][i]["gen_tokens"] - data["k6"][i]["gen_tokens"]
                 for i in math_ids]
    pos = sum(1 for d in diffs_all if d > 0)
    neg = sum(1 for d in diffs_all if d < 0)
    ties = sum(1 for d in diffs_all if d == 0)
    out["paired_tokens_all100"] = {
        "mean": statistics.mean(diffs_all),
        "median": statistics.median(diffs_all),
        "pos": pos, "neg": neg, "ties": ties,
        "sign_p_two_sided": sign_test_two_sided(pos, neg),
    }
    both_ids = [i for i in math_ids
                if data["k6"][i]["correct"] and data["k8"][i]["correct"]]
    diffs_bc = [data["k8"][i]["gen_tokens"] - data["k6"][i]["gen_tokens"]
                for i in both_ids]
    posb = sum(1 for d in diffs_bc if d > 0)
    negb = sum(1 for d in diffs_bc if d < 0)
    out["paired_tokens_both_correct"] = {
        "n": len(both_ids),
        "mean": statistics.mean(diffs_bc),
        "median": statistics.median(diffs_bc),
        "pos": posb, "neg": negb,
        "ties": len(diffs_bc) - posb - negb,
        "sign_p_two_sided": sign_test_two_sided(posb, negb),
    }

    # --- chain non-monotonicity ref -> k2 -> k6 -> k8 ---
    chain = ["ref", "k2", "k6", "k8"]
    nonmono = []
    for i in math_ids:
        pat = [data[c][i]["correct"] for c in chain]
        # "at least one recovery": wrong at some anchor, right at a LATER anchor
        if any(not pat[a] and pat[b] for a in range(4) for b in range(a + 1, 4)):
            nonmono.append({"id": i, "pattern": "".join(str(int(x)) for x in pat)})
    out["chain_nonmonotone_math500"] = {"count": len(nonmono), "items": nonmono}

    # --- KL anchors ---
    kl = {}
    for pol, fname in [("onpolicy", "topk_onpolicy.json"),
                       ("offpolicy", "topk_offpolicy.json")]:
        d = json.loads((KL / fname).read_text())
        kl[pol] = {c["name"]: c["mean_kl_nats"] for c in d["candidates"]}
    names = ["drop[16,12]", "drop[16,12,13,9,8,4]", "drop[16,12,13,9,8,4,5,15]"]
    out["kl_anchors"] = {
        pol: {n: kl[pol][n] for n in names} for pol in kl
    }
    out["kl_strictly_monotone"] = all(
        kl[pol][names[0]] < kl[pol][names[1]] < kl[pol][names[2]] for pol in kl)

    # --- write outputs ---
    OUT.with_suffix(".json").write_text(json.dumps(out, indent=2, default=str))

    r3 = lambda x: round(x, 3)
    md = []
    md.append("# verify_b_math_anomaly — independent re-derivation\n")
    md.append("All numbers computed from raw JSONs, paired by id "
              "(id sets asserted identical across all 4 configs and frozen items).\n")
    md.append("## Wilson 95% CIs (math500, z=1.96)\n")
    md.append("| config | acc | CI |")
    md.append("|---|---|---|")
    for cfg in CONFIGS:
        w = wilson[cfg]
        md.append(f"| {cfg} | {w['acc']:.2f} | "
                  f"[{r3(w['wilson_lo'])}, {r3(w['wilson_hi'])}] |")
    md.append("\n## Flip tables math500 (McNemar exact two-sided)\n")
    md.append("| pair | both✓ | A-only | B-only | both✗ | p |")
    md.append("|---|---|---|---|---|---|")
    for k, t in ft_math.items():
        md.append(f"| {k} | {t['both_correct']} | {t['a_only']} | {t['b_only']} "
                  f"| {t['both_wrong']} | {t['mcnemar_p']:.6f} |")
    md.append(f"\nk6 vs k8 one-sided exact binomial "
              f"P(X>={ft_math['k6_vs_k8']['b_only']} of "
              f"{ft_math['k6_vs_k8']['n_discordant']}) = "
              f"{out['k6_vs_k8_one_sided_p']:.6f}")
    md.append("\n## gsm8k companion\n")
    for k, t in ft_gsm.items():
        md.append(f"- {k}: {t['both_correct']}/{t['a_only']}/{t['b_only']}/"
                  f"{t['both_wrong']}, p={t['mcnemar_p']:.4f}")
    md.append(f"- gsm8k truncations: {out['truncation_gsm8k']}")
    md.append("\n## Truncation x correctness (math500)\n")
    md.append("| config | stop✓ | stop✗ | len✓ | len✗ | wrong-share-from-trunc |")
    md.append("|---|---|---|---|---|---|")
    for cfg in CONFIGS:
        t = trunc[cfg]
        md.append(f"| {cfg} | {t['stop_correct']} | {t['stop_wrong']} | "
                  f"{t['length_correct']} | {t['length_wrong']} | "
                  f"{t['wrong_share_from_truncation']:.4f} |")
    md.append(f"\nlength_correct == 0 everywhere: "
              f"{out['length_correct_zero_everywhere']}")
    md.append("\n## k6/k8 flip mechanism\n")
    r = out["k8_recoveries"]
    l = out["k8_losses"]
    md.append(f"- recoveries n={r['n']}: all k6 truncated at budget = "
              f"{r['all_k6_truncated_at_budget']}; all k8 stop = "
              f"{r['all_k8_finished_naturally']}; k8 tokens mean "
              f"{r['k8_tokens_mean']:.1f} / median {r['k8_tokens_median']}; "
              f"ref correct {r['n_ref_correct']}/{r['n']}")
    md.append(f"- losses n={l['n']}: all k8 truncated = "
              f"{l['all_k8_truncated_at_budget']}; k6 tokens range "
              f"[{l['k6_tokens_min']}, {l['k6_tokens_max']}]; "
              f"ref correct {l['n_ref_correct']}/{l['n']}")
    md.append(f"- all 8 ref→k6 losses are k6 truncations: "
              f"{out['ref_vs_k6_losses_all_k6_truncations']}")
    md.append("\n## Paired gen_tokens k8−k6 (math500)\n")
    a = out["paired_tokens_all100"]
    b = out["paired_tokens_both_correct"]
    md.append(f"- all 100: mean {a['mean']:.2f}, median {a['median']}, "
              f"sign +{a['pos']}/−{a['neg']} (ties {a['ties']}), "
              f"p={a['sign_p_two_sided']:.4f}")
    md.append(f"- both-correct n={b['n']}: mean {b['mean']:.1f}, median "
              f"{b['median']}, sign +{b['pos']}/−{b['neg']}, "
              f"p={b['sign_p_two_sided']:.4f}")
    md.append("\n## Chain + KL\n")
    md.append(f"- chain non-monotone items (≥1 recovery): "
              f"{out['chain_nonmonotone_math500']['count']}/100")
    md.append(f"- KL strictly monotone k2<k6<k8: {out['kl_strictly_monotone']}")
    md.append(f"- on-policy: {[r3(kl['onpolicy'][n]) for n in names]}; "
              f"off-policy: {[r3(kl['offpolicy'][n]) for n in names]}")
    OUT.with_suffix(".md").write_text("\n".join(md) + "\n")
    print(json.dumps({k: v for k, v in out.items()
                      if k not in ("flip_tables_math500", "flip_tables_gsm8k",
                                   "truncation_math500", "k8_recoveries",
                                   "k8_losses", "chain_nonmonotone_math500")},
                     indent=2, default=str))
    print("OK")


if __name__ == "__main__":
    main()
