import random
from collections import Counter

import pytest
from src.data import generator as G
from src.data import interpreter as I
from src.data.tokenizer import OPS, encode


@pytest.fixture(scope="module")
def cases():
    return [G.sample_case(random.Random(G.case_seed(1, "test", i)), "test", 8, f"c{i}")
            for i in range(300)]


def test_valid_programs_and_answers(cases):
    for c in cases:
        for w in "ab":
            for k in (0, 1, 4, 8):
                p = c.render_icl(w, k)
                encode(p)  # every token in vocabulary
                assert I.answer_prompt(p, c.meanings(w)) == c.answer(w)
            assert I.answer_prompt(c.render_definition(w)) == c.answer(w)


def test_pairs_share_query_and_differ_in_meaning(cases):
    for c in cases:
        for k in (0, 1, 2, 8):
            qa = c.render_icl("a", k).split(" Q ")[1]
            qb = c.render_icl("b", k).split(" Q ")[1]
            assert qa == qb
        assert c.meaning_a != c.meaning_b
        assert c.answer_a != c.answer_b
        assert c.render_definition("a").split(" Q ")[1] == c.render_definition("b").split(" Q ")[1]


def test_demonstrations_establish_different_meanings(cases):
    for c in cases:
        for k in (1, 3, 8):
            ma = I.infer_meanings_from_demos(c.render_icl("a", k))
            mb = I.infer_meanings_from_demos(c.render_icl("b", k))
            assert ma[c.target] == {c.meaning_a}
            assert mb[c.target] == {c.meaning_b}
            for d, m in zip(c.distractors, c.distractor_meanings):
                assert ma[d] == mb[d] == {m}


def test_each_demo_individually_unambiguous():
    rng = random.Random(0)
    for op in OPS:
        for _ in range(200):
            d = G.sample_demo(rng, op)
            p = f"<bos> {d.render('dax', I.apply_op(op, d.init, d.arg))} Q a = 1 ; a ->"
            assert I.infer_meanings_from_demos(p) == {"dax": {op}}


def test_zero_demo_pair_prompts_identical(cases):
    for c in cases[:50]:
        assert c.render_icl("a", 0) == c.render_icl("b", 0)


def test_demo_count_nested_and_counts(cases):
    c = cases[0]
    import re
    demos = lambda p: re.findall(r"D [^.]*\.", p)
    full = demos(c.render_icl("a", 8))
    for k in range(9):
        assert set(demos(c.render_icl("a", k))) <= set(full)  # nested demo sets
        n_demos = c.render_icl("a", k).split().count("D")
        assert n_demos == 3 * k


def test_same_command_different_meanings_across_prompts(cases):
    by_cmd = {}
    for c in cases:
        by_cmd.setdefault(c.target, set()).update({c.meaning_a, c.meaning_b})
    assert all(len(v) == 4 for v in by_cmd.values())


def test_meaning_balance(cases):
    cnt = Counter([c.meaning_a for c in cases] + [c.meaning_b for c in cases])
    assert min(cnt.values()) > 0.15 * sum(cnt.values())


def test_determinism():
    a = G.sample_case(random.Random(123), "test", 8)
    b = G.sample_case(random.Random(123), "test", 8)
    assert a == b
    assert G.sample_training_prompt(5, G.TrainSampleConfig()) == G.sample_training_prompt(5, G.TrainSampleConfig())


def test_template_splits_disjoint_and_no_leakage(cases):
    temps = G.all_templates()
    splits = {G.template_key(t): G.template_split(t) for t in temps}
    cnt = Counter(splits.values())
    assert cnt["train"] > 0 and cnt["val"] > 0 and cnt["test"] > 0
    assert all(splits[c.template] == "test" for c in cases)
    cfg = G.TrainSampleConfig()
    train_templates, train_programs = set(), set()
    for i in range(3000):
        _, meta = G.sample_training_prompt(G.case_seed(0, "train", i), cfg)
        train_templates.add(meta["template"]); train_programs.add(meta["program"])
    assert all(splits[t] == "train" for t in train_templates)
    assert not train_templates & {c.template for c in cases}
    assert not train_programs & {c.program for c in cases}


@pytest.mark.parametrize("n_init,n_stmts", [(2, 2), (2, 3), (3, 3), (3, 4)])
def test_task_configs_valid_and_split(n_init, n_stmts):
    task = G.TaskConfig(n_init, n_stmts)
    for i in range(40):
        c = G.sample_case(random.Random(i), "test", 4, task=task)
        q = c.render_icl("a", 2).split(" Q ")[1]
        assert q.count(";") == n_init + n_stmts
        assert q.split().count(c.target) == 1
        for w in "ab":
            assert I.answer_prompt(c.render_definition(w)) == c.answer(w)
    keys = {G.template_key(t): G.template_split(t) for t in G.all_templates(task)}
    for i in range(40):
        c = G.sample_case(random.Random(1000 + i), "test", 2, task=task)
        assert keys[c.template] == "test"


# ---- Experiment 2: ambiguous demonstrations ---------------------------------
AMB = G.TaskConfig(2, 3, ambiguous_demos=True)


@pytest.fixture(scope="module")
def amb_cases():
    return [G.sample_case(random.Random(G.case_seed(2, "test", i)), "test", 8, f"a{i}", task=AMB)
            for i in range(300)]


def test_ambiguous_demo_consistent_with_exactly_two_ops():
    rng = random.Random(0)
    conf = Counter()
    for op in OPS:
        for _ in range(600):
            d = G.sample_demo(rng, op, ambiguous=True)
            p = f"<bos> {d.render('dax', I.apply_op(op, d.init, d.arg))} Q a = 1 ; a ->"
            s = I.infer_meanings_from_demos(p)["dax"]
            assert len(s) == 2 and op in s
            conf[(op, next(iter(s - {op})))] += 1
    # every op has 3 confusers, chosen uniformly (600/3 = 200 each; loose bounds)
    assert len(conf) == 12 and all(140 < v < 260 for v in conf.values())


def test_ambiguous_cases_valid_and_paired(amb_cases):
    for c in amb_cases:
        assert c.answer_a != c.answer_b and c.meaning_a != c.meaning_b
        for w in "ab":
            for k in (0, 1, 3, 8):
                p = c.render_icl(w, k)
                encode(p)
                assert I.answer_prompt(p, c.meanings(w)) == c.answer(w)
            assert I.answer_prompt(c.render_definition(w)) == c.answer(w)
        assert c.render_icl("a", 0) == c.render_icl("b", 0)
        assert c.render_icl("a", 4).split(" Q ")[1] == c.render_icl("b", 4).split(" Q ")[1]


def test_ambiguous_identifiability_grows_with_k(amb_cases):
    """k=1 never identifies the meaning; identifiability rises with k (~1 - 3^(1-k))."""
    frac = {}
    for k in (1, 2, 4, 8):
        ok = [I.infer_meanings_from_demos(c.render_icl(w, k))[c.target] == {c.meanings(w)[c.target]}
              for c in amb_cases for w in "ab"]
        frac[k] = sum(ok) / len(ok)
    assert frac[1] == 0.0
    assert 0.5 < frac[2] < 0.8 and frac[4] > 0.9 and frac[8] > 0.99
    for c in amb_cases[:100]:      # the true meaning is always among the consistent ones
        for k in (1, 2, 8):
            assert c.meanings("a")[c.target] in I.infer_meanings_from_demos(c.render_icl("a", k))[c.target]


def test_ambiguous_flag_does_not_change_original_task():
    a = G.sample_case(random.Random(5), "test", 8, task=G.TaskConfig(2, 3))
    b = G.sample_case(random.Random(5), "test", 8, task=G.TaskConfig(2, 3, ambiguous_demos=False))
    assert a == b
