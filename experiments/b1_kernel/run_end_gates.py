"""B1 end gates on the real 27B nobias pack.

  identity  — load the pack twice (materialized biases vs derived kernels),
              teacher-force identical token sequences, require bit-equal
              logits on every item.
  ram       — load one path, eval all params, print mx.get_active_memory().
              Run each path in its own process; the delta is the B1 resident
              saving (expect ~= the 420 MB bias plane).

Usage:
  uv run python experiments/b1_kernel/run_end_gates.py identity [--items 6]
  uv run python experiments/b1_kernel/run_end_gates.py ram --path derived|materialized
"""
import argparse
import json
from pathlib import Path

import mlx.core as mx

from bonsaifold.loader import load_bonsai
from bonsaifold.probes import load_probe_set

REPO = Path(__file__).resolve().parents[2]
PACK = REPO / "models/Bonsai-27B-mlx-1bit-nobias"
PROBES = REPO / "experiments/calibration/probes_onpolicy.json"


def gate_ram(path):
    model, _ = load_bonsai(PACK, derived_kernels=(path == "derived"))
    mx.eval(model.parameters())
    active = mx.get_active_memory()
    print(json.dumps({"path": path, "active_bytes": active,
                      "active_gb": round(active / 2**30, 3)}))


def gate_identity(n_items):
    items = load_probe_set(PROBES)["items"][:n_items]
    ref, _ = load_bonsai(PACK, derived_kernels=False)
    der, _ = load_bonsai(PACK, derived_kernels=True)
    all_ok = True
    for item in items:
        ids = mx.array([item["token_ids"]])
        a = ref(ids)
        b = der(ids)
        mx.eval(a, b)
        ok = bool(mx.array_equal(a, b))
        all_ok &= ok
        print(f"{item['id']}: n={ids.shape[1]} bit-identical={ok}")
    print(f"IDENTITY GATE: {'PASS' if all_ok else 'FAIL'}")
    raise SystemExit(0 if all_ok else 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gate", choices=["identity", "ram"])
    ap.add_argument("--items", type=int, default=6)
    ap.add_argument("--path", choices=["derived", "materialized"])
    args = ap.parse_args()
    if args.gate == "ram":
        if args.path is None:
            ap.error("ram gate requires --path")
        gate_ram(args.path)
    else:
        gate_identity(args.items)


if __name__ == "__main__":
    main()
