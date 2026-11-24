"""Load-scaling task: correctness of prompts, supervision, determinism."""
import random

import pytest

from src.data import recall
from src.data.dataset import collate_lm


@pytest.mark.parametrize("fmt", ["icl", "definition"])
@pytest.mark.parametrize("n", [4, 32, 128])
def test_prompt_answers_match_independent_solver(fmt, n):
    for s in range(50):
        p = recall.make_prompt(random.Random(s), n, fmt, 8)
        assert recall.solve(p.text) == [q[2] for q in p.queries]
        cmds = [p.text.split()[i + 1] for i, t in enumerate(p.text.split()) if t in ("D", "M")]
        assert len(cmds) == n == len(set(cmds))                      # n distinct bindings
        assert len(p.queries) == min(8, n)
        assert len({q[0] for q in p.queries}) == len(p.queries)      # distinct queried commands


def test_only_query_answers_are_supervised():
    p = recall.make_prompt(random.Random(0), 16, "icl", 8)
    b = collate_lm([recall.encode(p.text)])
    sup = (b["labels"][0] != -100).nonzero().flatten()
    assert len(sup) == 8
    got = [recall.VOCAB[int(b["labels"][0, i])] for i in sup]
    assert got == [str(q[2]) for q in p.queries]


def test_stream_is_deterministic_and_mixes_loads_and_formats():
    cfg = recall.RecallConfig()
    a = recall.RecallStream(3, cfg, 100, 400)
    b = recall.RecallStream(3, cfg, 0, 500)
    assert all(a[i] == b[i + 100] for i in range(0, 400, 37))
    assert recall.RecallStream(4, cfg, 100, 1)[0] != a[0]
    ns, fmts = set(), set()
    for i in range(400):
        p = recall.sample_training_prompt(recall.case_seed(3, "recall-train", i), cfg)
        ns.add(p.n); fmts.add(p.fmt)
    assert ns == set(cfg.n_choices) and fmts == {"icl", "definition"}


def test_fixed_sets_are_disjoint_and_deterministic():
    v = recall.fixed_set("val", [8], ["icl"], 20, 8)
    t = recall.fixed_set("test", [8], ["icl"], 20, 8)
    assert v == recall.fixed_set("val", [8], ["icl"], 20, 8)
    assert not {r["text"] for r in v} & {r["text"] for r in t}
    for r in v:
        assert recall.solve(r["text"]) == r["answers"]
