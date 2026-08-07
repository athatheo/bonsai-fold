"""Width-saliency census: is there a prunable tail of MLP channels, measured
entirely from the quantization scales already in the pack?

Format-native width pruning deletes MLP intermediate channels in aligned
chunks of 128 (the g128 group size), so surviving groups keep their exact
packed words and scales: chunk c removes gate_proj/up_proj ROWS
[128c, 128c+128) and down_proj input-column GROUP c — deletions only, no
value changes, the width analogue of block dropping.

Saliency proxy (free to read, no forward passes): the f16 group scales.
|s_g| is the per-group weight magnitude by construction (1-bit weights are
exactly +/- s_g), so a chunk whose gate/up rows AND down column-group all
carry small scales moves little signal. This census maps the spread of that
proxy; low-spread == width pruning is dead on arrival, heavy tail == worth
a KL screen.

Usage: uv run python experiments/width_census/run_width_census.py
"""
import json
from pathlib import Path

import numpy as np

from bonsaifold.stio import PackReader

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).parent
PACK = REPO / "models/Bonsai-27B-mlx-1bit"
GROUP = 128
INTER = 17408  # MLP intermediate size
CHUNKS = INTER // GROUP  # 136


def chunk_saliency(pack, block):
    base = f"language_model.model.layers.{block}.mlp"
    # gate/up: [17408, 5120] -> scales [17408, 40]; row r belongs to chunk r//128
    gate = np.abs(pack.read(f"{base}.gate_proj.scales").astype(np.float32))
    up = np.abs(pack.read(f"{base}.up_proj.scales").astype(np.float32))
    row_mag = (gate.mean(axis=1) + up.mean(axis=1)).reshape(CHUNKS, GROUP).mean(axis=1)
    # down: [5120, 17408] -> scales [5120, 136]; column-group c IS chunk c
    down = np.abs(pack.read(f"{base}.down_proj.scales").astype(np.float32))
    col_mag = down.mean(axis=0)
    # combined proxy: geometric mean couples "writes little" and "read little"
    return np.sqrt(row_mag * col_mag)


def main():
    pack = PackReader(PACK)
    layer_types = pack.config["text_config"]["layer_types"]
    blocks = []
    all_ratios = []
    for b in range(len(layer_types)):
        s = chunk_saliency(pack, b)
        order = np.argsort(s)
        med = float(np.median(s))
        blocks.append(
            {
                "block": b,
                "type": layer_types[b].split("_")[0],
                "median": round(med, 6),
                "min_over_median": round(float(s.min() / med), 4),
                "p10_over_median": round(float(np.percentile(s, 10) / med), 4),
                "chunks_below_half_median": int((s < 0.5 * med).sum()),
                "weakest_chunks": order[:5].tolist(),
            }
        )
        all_ratios.append(s / med)
        print(f"block {b:2d} {layer_types[b].split('_')[0]:6s} "
              f"min/med {s.min()/med:.3f}  p10/med {np.percentile(s,10)/med:.3f}  "
              f"<0.5med: {(s<0.5*med).sum()}")

    ratios = np.concatenate(all_ratios)
    prunable_10 = int((ratios < 0.5).sum())
    chunk_mb = (GROUP * 5120 * 3) * 1.125 / 8 / 1e6  # gate+up rows + down cols
    summary = {
        "chunks_total": len(ratios),
        "chunks_below_half_median": prunable_10,
        "chunk_size_mb": round(chunk_mb, 3),
        "savings_if_all_weak_chunks_mb": round(prunable_10 * chunk_mb, 1),
        "spread_verdict": (
            "heavy tail -> screen-worthy" if prunable_10 > 200 else
            "modest tail" if prunable_10 > 50 else "flat -> likely DOA"
        ),
    }
    (OUT / "width_census.json").write_text(
        json.dumps({"summary": summary, "blocks": blocks}, indent=1)
    )
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
