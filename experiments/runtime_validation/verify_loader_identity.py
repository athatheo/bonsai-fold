"""Gate for the layer_types-aware loader: on the UNFOLDED pack, the typed
path must be logit-identical to stock mlx-lm (same weights, same kernels,
same wiring — only the construction path differs). Run once per
environment before any folded-model experiment.

Loads both models (~10 GB total), teacher-forces a few short sequences,
and compares logits. Writes loader_identity.json next to this file.
"""
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm.utils import load_model

from bonsaifold.loader import load_bonsai

SEQUENCES = [
    [151644, 872, 198, 9707, 11, 1246, 525, 498, 30, 151645],  # arbitrary ids
    list(range(1000, 1128)),
    [7, 42] * 64,
]


def main(pack_dir):
    stock, _ = load_model(Path(pack_dir))
    typed, _ = load_bonsai(pack_dir)
    report = {"pack": str(pack_dir), "sequences": []}
    worst = 0.0
    for seq in SEQUENCES:
        tokens = mx.array([seq])
        a = stock(tokens)
        b = typed(tokens)
        mx.eval(a, b)
        av = np.array(a, copy=False)
        bv = np.array(b, copy=False)
        identical = bool((av == bv).all())
        maxd = float(np.abs(av.astype(np.float64) - bv.astype(np.float64)).max())
        worst = max(worst, maxd)
        report["sequences"].append(
            {"tokens": len(seq), "bit_identical": identical, "max_abs_diff": maxd}
        )
        print(f"{len(seq)} tokens: bit_identical={identical} max|d|={maxd:.3e}")
    report["ok"] = all(s["bit_identical"] for s in report["sequences"])
    out = Path(__file__).parent / "loader_identity.json"
    out.write_text(json.dumps(report, indent=1))
    print(("IDENTICAL" if report["ok"] else f"NOT BIT-IDENTICAL (worst {worst:.3e})"), "->", out)
    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "models/Bonsai-27B-mlx-1bit")
