import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "experiments/evosearch"))
from run_evosearch import (DROP_ATTN, DROP_BLOCK, DROP_MLP, KEEP, build_view,
                           genome_bytes, valid)
from test_loader import F, L, build


def test_genome_bytes_and_validity():
    lt = ["linear_attention"] * 3 + ["full_attention"]
    g = [KEEP, DROP_BLOCK, DROP_ATTN, DROP_MLP]
    assert abs(genome_bytes(g, lt) - (60.0 + 16.2 + 38.0)) < 1e-9
    assert valid(g, lt)  # block0 keep; linear attn survives at 0; full attn dropped-mlp at 3 keeps attn
    assert not valid([DROP_BLOCK] + g[1:], lt)  # block 0 must be kept
    # dropping every full-attn's attention kills the fa anchor
    g2 = [KEEP, KEEP, KEEP, DROP_ATTN]
    assert not valid(g2, lt)


def test_build_view_matches_manual_composition():
    import run_evosearch as ev
    ev_n = ev.N_BLOCKS
    ev.N_BLOCKS = 4  # tiny model
    try:
        model = build([L, L, F, L])
        tokens = mx.array([[3, 1, 4, 1, 5]])
        from bonsaifold.loader import drop_view, sublayer_view
        # genome: drop block 1, drop attn of block 3
        g = [KEEP, DROP_BLOCK, KEEP, DROP_ATTN]
        got = build_view(model, g)(tokens)
        # manual: sublayer_view for attn-drop, then subset for block-drop is
        # not directly composable via public API; verify against explicit
        # forward: blocks 0,2,3 with 3's attn skipped
        import mlx_lm.models.qwen3_5 as q35
        tm = model.language_model.model
        h = tm.embed_tokens(tokens)
        ssm = q35.create_ssm_mask(h, None); fa = q35.create_attention_mask(h, None)
        for i in [0, 2, 3]:
            l = tm.layers[i]
            if i == 3:
                h = h + l.mlp(l.post_attention_layernorm(h))
            else:
                h = l(h, mask=ssm if l.is_linear else fa, cache=None)
        manual = model.language_model.lm_head(tm.norm(h))
        assert np.array_equal(np.array(got, copy=False), np.array(manual, copy=False))
    finally:
        ev.N_BLOCKS = ev_n
