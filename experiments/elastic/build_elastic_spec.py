"""Q4: generate the elastic-depth spec from measured screens/benches.

One pack + a nested drop ORDER such that every prefix is a measured (or
additive-law-bounded) configuration: prefixes at the anchor points are
byte-for-byte the configs we benched (k2, k4, folded-709) or KL-confirmed
(k4+s8 tier). A deployment picks the longest prefix within its byte budget
and applies it as a zero-copy view (bonsaifold.elastic.elastic_view) or
folds a real pack with the existing operators.

The order follows the BENCHED trajectory, not the A6 frontier: elastic
nesting requires each tier to be a superset of the previous, and the
benched chain k2 ⊂ k4 ⊂ folded-709 ⊂ k4+s8 is the only measured chain with
bench anchors. A6 champions are better per-byte but not nested.

All numbers are read from committed result JSONs. Rerun after any rebench.
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
KL = REPO / "experiments/kl_screen/results"
BENCH = REPO / "experiments/minibench/results"
# depth of the base pack; the nobias pack is the census pack with the bias
# plane stripped, same 64 blocks. Stamped into the spec so elastic_view can
# refuse a folded pack (whose block numbering has shifted).
CENSUS = REPO / "experiments/census/census_Bonsai-27B-mlx-1bit.json"
OUT = Path(__file__).parent / "elastic_spec.json"

# nested op order; anchor prefixes == measured configs
ORDER = ["b16", "b12", "b13", "b9", "a38", "a37", "a58", "a57", "a61", "a8"]
ANCHORS = {
    2: {"name": "k2", "set_key": "drop[16,12]", "set_file": "topk", "bench": "k2_16-12"},
    4: {"name": "k4", "set_key": "drop[16,12,13,9]", "set_file": "topk", "bench": "k4"},
    7: {"name": "folded-709", "set_key": "sub[b16+b12+b13+b9+a38+a37+a58]",
        "set_file": "sublayer_sets", "bench": "folded709_pack"},
    10: {"name": "k4+s8-tier", "set_key": "sub[b16+b12+b13+b9+a38+a37+a58+a57+a61+a8]",
         "set_file": "sublayer_sets", "bench": None},
}
# census-derived structural MB per op class (linear blocks / linear attn)
OP_MB = {"b": 60.0, "a": 16.2, "m": 38.0}


def candidates(fname, regime):
    """Candidate rows of one KL screen result file."""
    return json.loads((KL / f"{fname}_{regime}.json").read_text())["candidates"]


def kl_lookup(fname, regime, key):
    for r in candidates(fname, regime):
        if r["name"] == key:
            return r["mean_kl_nats"]
    raise KeyError(f"{key} not in {fname}_{regime}")


def single_key(op):
    """Op token -> its key in the single-op screen results."""
    return f"drop[{op[1:]}]" if op[0] == "b" else f"sub[{op}]"


def main():
    if len(set(ORDER)) != len(ORDER):
        raise SystemExit(f"ORDER has duplicate ops: {ORDER}")
    if not set(ANCHORS) <= set(range(1, len(ORDER) + 1)):
        raise SystemExit(f"ANCHORS steps outside 1..{len(ORDER)}: {sorted(ANCHORS)}")

    singles_on = {}
    for fname in ("single_block", "sublayer_singles"):
        singles_on.update(
            {r["name"]: r["mean_kl_nats"] for r in candidates(fname, "onpolicy")}
        )

    steps, cum_mb = [], 0.0
    for i, op in enumerate(ORDER, 1):
        cum_mb += OP_MB[op[0]]
        step = {
            "step": i,
            "op": op,
            "cum_structural_mb": round(cum_mb, 1),
            "single_kl_onpolicy": singles_on[single_key(op)],
        }
        if i in ANCHORS:
            a = ANCHORS[i]
            step["anchor"] = a["name"]
            step["set_kl_onpolicy"] = kl_lookup(a["set_file"], "onpolicy", a["set_key"])
            step["set_kl_offpolicy"] = kl_lookup(a["set_file"], "offpolicy", a["set_key"])
            if a["bench"]:
                b = json.loads((BENCH / f"{a['bench']}.json").read_text())
                step["bench_file"] = a["bench"]
                step["bench_macro"] = b["macro_avg"]
                step["bench_tasks"] = b["task_accuracy"]
        steps.append(step)

    spec = {
        "base_pack": "models/Bonsai-27B-mlx-1bit-nobias",
        "base_num_blocks": json.loads(CENSUS.read_text())["num_language_blocks"],
        "note": "prefixes at anchors == benched/KL-confirmed configs; "
                "non-anchor prefixes bounded by the additive law "
                "(cross-pool tax 1.09-1.13x on-policy). Byte-identical: "
                "every tier only deletes; surviving weights untouched.",
        "order": ORDER,
        "steps": steps,
    }
    OUT.write_text(json.dumps(spec, indent=1))
    print(f"wrote {OUT}")
    for s in steps:
        tag = f" <== ANCHOR {s['anchor']}" if "anchor" in s else ""
        macro = f" macro={s['bench_macro']:.4f}" if "bench_macro" in s else ""
        print(f"  {s['step']:2d}. {s['op']:4s} cum={s['cum_structural_mb']:6.1f}MB"
              f" single_kl={s['single_kl_onpolicy']:.5f}{macro}{tag}")


if __name__ == "__main__":
    main()
