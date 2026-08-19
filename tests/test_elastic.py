"""Q4 elastic-depth spec + helper tests (no model load)."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from bonsaifold.elastic import elastic_view, load_spec, prefix_for_budget, split_ops

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "experiments/minibench/results"


def fake_model(n_blocks):
    """Stand-in exposing only the depth elastic_view checks."""
    inner = SimpleNamespace(layers=[None] * n_blocks)
    return SimpleNamespace(language_model=SimpleNamespace(model=inner))


def test_spec_integrity():
    s = load_spec()
    steps = s["steps"]
    assert [x["op"] for x in steps] == s["order"]
    assert len(set(s["order"])) == len(s["order"])
    # cumulative MB strictly increasing; anchors carry set KL both regimes
    cums = [x["cum_structural_mb"] for x in steps]
    assert cums == sorted(cums) and len(set(cums)) == len(cums)
    anchors = [x for x in steps if "anchor" in x]
    assert [a["anchor"] for a in anchors] == ["k2", "k4", "folded-709", "k4+s8-tier"]
    for a in anchors:
        assert a["set_kl_onpolicy"] > 0 and a["set_kl_offpolicy"] > 0


def test_benched_anchors_match_committed_rows():
    """Every benched anchor still equals its committed bench JSON (drift)."""
    benched = [x for x in load_spec()["steps"] if "bench_file" in x]
    assert [x["anchor"] for x in benched] == ["k2", "k4", "folded-709"]
    for step in benched:
        row = json.loads((BENCH / f"{step['bench_file']}.json").read_text())
        assert step["bench_macro"] == row["macro_avg"]
        assert step["bench_tasks"] == row["task_accuracy"]


def test_prefix_for_budget():
    s = load_spec()
    assert prefix_for_budget(s, 0) == []
    assert prefix_for_budget(s, 121) == ["b16", "b12"]  # k2 anchor
    assert prefix_for_budget(s, 300) == s["order"][:7]  # folded-709 anchor
    assert prefix_for_budget(s, 10_000) == s["order"]


def test_prefix_for_budget_at_anchor_boundary():
    """A budget exactly at an anchor's cumulative MB includes that anchor."""
    s = load_spec()
    for step in s["steps"]:
        cum = step["cum_structural_mb"]
        assert prefix_for_budget(s, cum) == s["order"][: step["step"]]
        assert prefix_for_budget(s, cum - 0.1) == s["order"][: step["step"] - 1]


def test_split_ops():
    assert split_ops(["b16", "b12", "a38", "m4"]) == ([16, 12], [38], [4])


def test_elastic_view_arg_validation():
    with pytest.raises(ValueError, match="exactly one"):
        elastic_view(object(), budget_mb=100, steps=2)
    with pytest.raises(ValueError, match="exactly one"):
        elastic_view(object())


def test_elastic_view_rejects_out_of_range_steps():
    n = len(load_spec()["order"])
    for bad in (0, -1, n + 1):
        with pytest.raises(ValueError, match="steps must be in"):
            elastic_view(object(), steps=bad)


def test_elastic_view_rejects_non_base_pack():
    """Folded packs renumber blocks, so spec indices must not be applied."""
    s = load_spec()
    with pytest.raises(ValueError, match="base-pack block numbers"):
        elastic_view(fake_model(s["base_num_blocks"] - 7), steps=2)
    with pytest.raises(ValueError, match="not a view"):
        elastic_view(object(), steps=2)


def test_elastic_view_rejects_budget_below_first_tier():
    s = load_spec()
    with pytest.raises(ValueError, match="below the first tier"):
        elastic_view(fake_model(s["base_num_blocks"]), budget_mb=1)
