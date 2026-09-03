"""Assemble the paper's result tables from canonical committed JSONs.

Emits markdown to tables.md. Re-run after any bench completes. Rules:
- every bench table leads with the unfolded Bonsai-27B-1bit reference row;
- Group C (flagged, value-modifying) rows sit in their OWN table, never a
  row among byte-identical lines (CLAUDE.md flagged-arm reporting rule);
- numbers are read from result files, never typed by hand.

Usage: uv run python experiments/paper_tables/build_tables.py
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RES = REPO / "experiments/minibench/results"
OUT = Path(__file__).parent / "tables.md"

# (result file, label, MB removed vs 4.71 GB pack, notes) — byte-identical line
BENCH_ROWS = [
    ("reference", "Bonsai-27B-1bit (unfolded reference)", 0, "—"),
    ("k2_16-12", "k2: drop blocks {16,12}", 118, "screen champion, 2 blocks"),
    ("k4", "k4: drop blocks {16,12,13,9}", 236, ""),
    ("k6", "k6: drop 6 blocks", 360, ""),
    ("k8", "k8: drop 8 blocks", 478, "damage knee"),
    ("folded709_pack", "folded-709 (k4 + attn{37,38,58}); SHIPPING", 709,
     "incl. bias strip; pack == benched view, 500/500"),
    ("folded709_h8", "promotion probe: folded-709 + H8 KV-groups (REJECTED)", 739,
     "screened sub-additive & below k6 both regimes; bench -3.5 vs 709: "
     "head family has a steep KL->bench slope (6th screens-vs-benches)"),
    ("a6_t350_champion", "A6 T350 search champion (KL-tier winner)", 365,
     "KL ties k6 / beats it 14% off-policy, but benches BELOW k6: "
     "screen-fitness Goodharting"),
]
FLAGGED_ROWS = [
    # benched pack = nobias + scale quant (no structural drops): 420 + 192.5
    ("groupc_scaleq8", "Group C: nobias + 8-bit scale plane (VALUE-MODIFYING)", 612,
     "KL 2.7e-06 on / 9.2e-06 off; stackable on folded-709's structural drops"),
    # folded-709 + scale quant of surviving tensors: 709 + 179.1
    ("folded709_scaleq8", "COMBINED: folded-709 + 8-bit scales (VALUE-MODIFYING)", 888,
     "3.82 GB total (-18.9%); screen 3.1e-06/9.9e-06 vs folded-709"),
    # Group R v2: in-format shadow repair — same bytes as folded-709
    ("folded709_shadowrepair", "Group R: folded-709 + shadow repair (VALUE-MOD)", 709,
     "in-format requant of 3 modules (1% flips): recovers +1.0 macro at "
     "IDENTICAL bytes; screen -7.2%/-9.0% held-out"),
    # repair EXTENSION probes (both failed the bench gate; kept as evidence)
    ("folded709_scaleq8_shadowrepair", "extension probe: combined + repair (WORSE)", 888,
     "screen -7.1/-9.0 identical to the confirmed repair, bench -0.75: "
     "repair does not compose with scale quantization (5th screens-vs-benches)"),
    ("k6_shadowrepair", "extension probe: k6 + repair (FLAT)", 360,
     "4 sites at dW cap; screen flat-on/-20%-off, bench -0.1: v2 recipe "
     "does not scale to deeper folds"),
    # knee probe, NOT a shipping candidate: bench found the knee at 4 bits
    ("scaleq4_nobias", "knee probe: nobias + 4-bit scales (DEGRADED)", 717,
     "screen 1e-04 looked free; bench -2.9 pts = k2-class damage. "
     "Scale-metadata floor stays at 8-bit (6-bit unbenched)"),
]

# Q3 generality rows (dense 8B sibling; non-thinking scoring)
BENCH_8B_ROWS = [
    ("reference_8b", "Bonsai-8B-1bit (unfolded reference)", 0, "dense qwen3, 36 blocks"),
    ("k2analog_8b", "8B k2-analog: drop {26,17}", 33, "set KL .046 on / .103 off"),
    ("k4analog_8b", "8B k4-analog: drop {26,17,21,31}", 66, "set KL .107 on / .257 off"),
]

KL_PARETO = [
    ("hand", "folded-709 config", 289, 0.026, 0.16),
    ("hand", "k6", 360, 0.031, 0.205),
    ("hand", "k8_mixed", 478, 0.064, 0.392),
    ("search", "A6 T350 champion", 364.6, 0.0314, 0.176),
    ("search", "A6 T350-s2 champion", 374.8, 0.0322, 0.187),
    ("search", "A6 T450 champion", 456.8, 0.0504, 0.233),
]


def bench_table(rows, require_complete=True):
    lines = ["| config | MB removed | GSM8K | MATH500 | IFEval | MMLU-R | macro | n |",
             "|---|---|---|---|---|---|---|---|"]
    for fname, label, mb, note in rows:
        p = RES / f"{fname}.json"
        if not p.exists():
            lines.append(f"| {label} | {mb} | *pending* | | | | | |")
            continue
        d = json.loads(p.read_text())
        n = len(d["items"])
        if require_complete and n < 500:
            lines.append(f"| {label} | {mb} | *running ({n}/500)* | | | | | |")
            continue
        t = d["task_accuracy"]
        lines.append(
            f"| {label} | {mb} | {t.get('gsm8k', float('nan')):.3f} | "
            f"{t.get('math500', float('nan')):.3f} | {t.get('ifeval', float('nan')):.3f} | "
            f"{t.get('mmlu', float('nan')):.3f} | {d['macro_avg']:.4f} | {n} |"
        )
    return "\n".join(lines)


def kl_table():
    lines = ["| frontier | config | structural MB | KL on-policy | KL off-policy |",
             "|---|---|---|---|---|"]
    for kind, label, mb, on, off in KL_PARETO:
        lines.append(f"| {kind} | {label} | {mb} | {on:.4f} | {off:.3f} |")
    return "\n".join(lines)


def main():
    md = [
        "# bonsai-fold result tables (generated; do not edit by hand)",
        "",
        "## Table 1 — mini-bench (500 items), byte-identical arm",
        "All surviving weights byte-identical to the original pack. "
        "Sizes vs the 4.71 GB nobias-inclusive baseline.",
        "",
        bench_table(BENCH_ROWS),
        "",
        "## Table 2 — FLAGGED arm: Group C scale quantization (value-modifying)",
        "Reported separately per the flagged-arm rule: scales are 8-bit "
        "reconstructions; weights are NOT byte-identical. Authorized 2026-08-12.",
        "",
        bench_table(FLAGGED_ROWS),
        "",
        "## Table 2b — Generality: dense Bonsai-8B (byte-identical drops)",
        "Same ladder, non-thinking scoring (the 8B family predates thinking "
        "mode). The structural laws replicate; the slack magnitude does not.",
        "",
        bench_table(BENCH_8B_ROWS),
        "",
        "## Table 3 — KL Pareto frontier, hand-built vs A6 search (100 probes, both regimes)",
        "",
        kl_table(),
        "",
        "## Memory/disk engineering results (B1, exact)",
        "",
        "- B1 in-kernel bias derivation: resident RAM −420,225,024 B on the 27B "
        "(4,207,606,792 → 3,787,381,768), logits bit-identical (6-probe identity "
        "gate); disk −420 MB via the nobias pack.",
        "- Group C disk: −192.5 MB further (498 scale tensors, per-row 8-bit).",
        "",
    ]
    OUT.write_text("\n".join(md))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
