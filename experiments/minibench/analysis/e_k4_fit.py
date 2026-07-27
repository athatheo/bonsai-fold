"""k4 anchor lands: re-discriminate the KL->damage fit with 5 anchors and
run the standing paired checks (McNemar exact, truncation decomposition)
for k4 vs k2 and k4 vs reference. Deterministic; stdlib + numpy only."""
import json
from math import comb, sqrt
from pathlib import Path

R = Path(__file__).resolve().parents[1] / "results"
CONFIGS = ["reference", "k2_16-12", "k4", "k6", "k8"]
KL_ON = {"reference": 0.0, "k2_16-12": 0.00755, "k4": 0.01770, "k6": 0.03111, "k8": 0.06426}


def mcnemar(a, b):  # exact two-sided on discordant pairs
    x = sum(1 for i in a if a[i] and not b[i])
    y = sum(1 for i in a if b[i] and not a[i])
    n = x + y
    if n == 0:
        return x, y, 1.0
    p = sum(comb(n, k) for k in range(0, min(x, y) + 1)) / 2**n * 2
    return x, y, min(1.0, p + (comb(n, x) / 2**n if x == y else 0))


data = {c: json.loads((R / f"{c}.json").read_text()) for c in CONFIGS}
items = {c: {r["id"]: r for r in data[c]["items"]} for c in CONFIGS}
ids = sorted(items["reference"])
assert all(sorted(items[c]) == ids for c in CONFIGS)

macro = {c: data[c]["macro_avg"] for c in CONFIGS}
drops = {c: (macro["reference"] - macro[c]) * 100 for c in CONFIGS}
print("anchors (KL_on -> macro drop pts):")
for c in CONFIGS:
    print(f"  {c:>10}: {KL_ON[c]:.5f} -> {drops[c]:.3f}")

xs = [KL_ON[c] for c in CONFIGS]
ys = [drops[c] for c in CONFIGS]
for name, f in [("linear b*KL", lambda x: x), ("sqrt c*sqrt(KL)", sqrt)]:
    fx = [f(x) for x in xs]
    coef = sum(a * b for a, b in zip(fx, ys)) / sum(a * a for a in fx)
    resid = [y - coef * a for y, a in zip(ys, fx)]
    ss = sum(r * r for r in resid)
    sst = sum((y - sum(ys) / len(ys)) ** 2 for y in ys)
    print(f"5-anchor fit {name}: coef={coef:.3f}  residuals="
          f"{[round(r,2) for r in resid]}  RMSE={sqrt(ss/len(ys)):.3f}  R2={1-ss/sst:.4f}")

print("\npaired McNemar (a-only-correct, b-only-correct, exact p):")
for a, b in [("k2_16-12", "k4"), ("reference", "k4"), ("reference", "k2_16-12")]:
    for task in ["gsm8k", "math500", "ifeval", "mmlu", None]:
        sel = [i for i in ids if task is None or items[a][i]["task"] == task]
        ca = {i: items[a][i]["correct"] for i in sel}
        cb = {i: items[b][i]["correct"] for i in sel}
        x, y, p = mcnemar(ca, cb)
        print(f"  {a} vs {b} {task or 'ALL':>8}: ({x},{y}) p={p:.4f}")

print("\ntruncation (finish_reason != stop) per config x task:")
for c in CONFIGS:
    t = {}
    for r in items[c].values():
        t.setdefault(r["task"], [0, 0])
        t[r["task"]][r["finish_reason"] != "stop"] += 1
    print(f"  {c:>10}: " + "  ".join(f"{k}:{v[1]}/{sum(v)}" for k, v in sorted(t.items())))
