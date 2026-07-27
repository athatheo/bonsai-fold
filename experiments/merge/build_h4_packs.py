"""Build (or resume) the H4 merged packs: shortlist pairs x both operators.

Idempotent: an existing pack dir is kept if verify_merge passes (cheap exact
recompute), rebuilt otherwise — so the chain can rerun after any crash.
Packs land under models/ (gitignored, derived artifacts).
"""
import json
import shutil
import sys
from pathlib import Path

from bonsaifold.merge import merge_blocks, verify_merge

SRC = "models/Bonsai-27B-mlx-1bit"
PAIRS = [(4, 5), (18, 20), (17, 18), (60, 61)]
OPERATORS = ("sign_election", "promotion")


def main():
    built = []
    for i, j in PAIRS:
        for operator in OPERATORS:
            dst = Path(f"models/merged_{i}-{j}_{operator}")
            if dst.exists():
                try:
                    n = json.loads((dst / "config.json").read_text())["text_config"][
                        "num_hidden_layers"
                    ]
                    block_map = {k: (k if k < j else k - 1) for k in range(n + 1) if k != j}
                    if verify_merge(SRC, dst, block_map, i, j, operator)["ok"]:
                        print(f"{dst}: exists and verifies, kept")
                        built.append(str(dst))
                        continue
                except Exception as e:  # corrupt/partial pack: rebuild
                    print(f"{dst}: failed verification ({e}); rebuilding")
                shutil.rmtree(dst)
            block_map = merge_blocks(SRC, dst, i, j, operator)
            report = verify_merge(SRC, dst, block_map, i, j, operator)
            if not report["ok"]:
                sys.exit(f"{dst}: verify_merge FAILED after build")
            print(f"{dst}: built and verified")
            built.append(str(dst))
    print("\n".join(built))


if __name__ == "__main__":
    main()
