"""H5 stage B: compose matched-k drop sets from the stage-A singles results.

Deterministic selection so the chain can run unattended: candidates are
ranked by off-policy single-block KL (the harsher regime; it re-ranked
stage-A singles before and disqualified block 1). Emits `--drop` args for
run_kl_screen.py on stdout and records the selection rationale in
experiments/kl_screen/results/h5_selection.json.

Sets composed (matched-k against existing anchors):
  k8_linear  {16,12,13,9,8,4,5}+L*  vs mixed k8 {16,12,13,9,8,4,5,15} — differ in one member
  k2_full    {15,F*}                vs linear k2 {16,12}
  k4_mixed   {16,12,15,F*}          vs linear k4 {16,12,13,9}
  k4_full    {15,F1,F2,F3}          only if all three new full singles pass the
                                    provisional 0.05-nats off-policy screen
where L* / F* = best new linear / full single by off-policy KL.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "experiments/kl_screen/results"

LINEAR_POOL = [57, 17, 37, 58, 33, 30]  # next-by-BI + BI-matched controls (33~15, 30~55)
FULL_POOL = [55, 39, 23, 47]  # top unscreened full-attention blocks by BI
OFFPOLICY_SCREEN = 0.05  # provisional gate, used only to veto k4_full composition


def load_singles(path):
    data = json.loads(path.read_text())
    kl = {}
    for cand in data["candidates"]:
        spec = cand["name"].removeprefix("drop[").removesuffix("]")
        blocks = tuple(int(b) for b in spec.split(","))
        if len(blocks) == 1:
            kl[blocks[0]] = cand["mean_kl_nats"]
    return kl


def main():
    off = load_singles(RESULTS / "h5_singles_offpolicy.json")
    on = load_singles(RESULTS / "h5_singles_onpolicy.json")
    missing = [b for b in LINEAR_POOL + FULL_POOL if b not in off or b not in on]
    if missing:
        sys.exit(f"singles incomplete, missing blocks {missing}; rerun stage A first")

    lin_ranked = sorted(LINEAR_POOL, key=lambda b: off[b])
    full_ranked = sorted(FULL_POOL, key=lambda b: off[b])
    l_star, f_star = lin_ranked[0], full_ranked[0]

    sets = {
        "k8_linear": [16, 12, 13, 9, 8, 4, 5, l_star],
        "k2_full": [15, f_star],
        "k4_mixed": [16, 12, 15, f_star],
    }
    passing = [b for b in full_ranked if off[b] < OFFPOLICY_SCREEN]
    if len(passing) >= 3:
        sets["k4_full"] = [15] + passing[:3]
    else:
        vetoed = {b: off[b] for b in full_ranked if off[b] >= OFFPOLICY_SCREEN}
        sets["k4_full_vetoed"] = vetoed  # rationale only, not screened

    rationale = {
        "rule": "rank new singles by off-policy KL; L*/F* = best linear/full",
        "singles_onpolicy": {str(b): on[b] for b in LINEAR_POOL + FULL_POOL},
        "singles_offpolicy": {str(b): off[b] for b in LINEAR_POOL + FULL_POOL},
        "l_star": l_star,
        "f_star": f_star,
        "sets": {k: v for k, v in sets.items() if not k.endswith("_vetoed")},
        "vetoed": sets.get("k4_full_vetoed"),
    }
    (RESULTS / "h5_selection.json").write_text(json.dumps(rationale, indent=1))

    args = [
        f"--drop {','.join(str(b) for b in blocks)}"
        for name, blocks in sets.items()
        if not name.endswith("_vetoed")
    ]
    print(" ".join(args))


if __name__ == "__main__":
    main()
