import random

import numpy as np
import pandas as pd
import pytest
import torch

from src.data import generator as G
from src.data.dataset import collate_lm
from src.data.tokenizer import ARROW_ID, encode
from src.evaluation import statistics as S
from src.evaluation.evaluate import build_eval_prompts
from src.evaluation.paired_accuracy import pair_table, paired_accuracy, summarize_run


def _df(correct_a, correct_b):
    rows = []
    for i, (ca, cb) in enumerate(zip(correct_a, correct_b)):
        rows.append({"case_id": i, "condition": "icl", "k": 1, "which": "a", "correct": ca, "valid": True})
        rows.append({"case_id": i, "condition": "icl", "k": 1, "which": "b", "correct": cb, "valid": True})
    return pd.DataFrame(rows)


def test_paired_accuracy_hand_computed():
    df = _df([True, True, False, True], [True, False, False, True])
    assert paired_accuracy(df) == 0.5
    s = summarize_run(df)
    assert s.loc[0, "individual_acc"] == 5 / 8 and s.loc[0, "paired_acc"] == 0.5


def test_pair_table_requires_two_prompts():
    df = _df([True], [True]).iloc[:1]
    with pytest.raises(ValueError):
        pair_table(df)


def test_constant_predictor_gets_zero_paired_accuracy():
    cases = [G.sample_case(random.Random(i), "test", 8, f"c{i}") for i in range(50)]
    df = build_eval_prompts(cases, [0, 1, 2])
    # a model whose prediction depends only on the query program cannot solve any pair
    pred = {}
    for _, r in df.iterrows():
        q = r["prompt"].split(" Q ")[1]
        pred.setdefault(q, r["answer"])
    df["correct"] = [pred[r["prompt"].split(" Q ")[1]] == r["answer"] for _, r in df.iterrows()]
    df["valid"] = True
    assert paired_accuracy(df) == 0.0


def test_eval_prompts_deterministic_and_pairs_complete():
    cases = [G.sample_case(random.Random(i), "test", 8, f"c{i}") for i in range(20)]
    d1, d2 = build_eval_prompts(cases, [0, 2, 8]), build_eval_prompts(cases, [0, 2, 8])
    pd.testing.assert_frame_equal(d1, d2)
    assert (d1.groupby(["case_id", "condition", "k"]).size() == 2).all()


def test_collate_supervision_positions():
    s = encode("<bos> D a = 1 ; dax a 2 ; a -> 3 . Q b = 1 ; dax b 1 ; b -> 2 <eos>")
    b = collate_lm([s, s[:5]])
    inp, lab = b["input_ids"][0], b["labels"][0]
    assert ((lab != -100) == (inp == ARROW_ID)).all()
    assert (lab != -100).sum() == 2
    assert b["answer"][0].item() == encode("2")[0]


def test_stats_helpers():
    m = S.mean_ci([1, 2, 3, 4, 5])
    assert m["mean"] == 3 and m["ci_low"] < 3 < m["ci_high"]
    assert S.permutation_test([1, 2, 3], [1, 2, 3]) == 1.0
    assert S.permutation_test([10, 11, 12, 13, 14], [0, 1, 2, 3, 4]) == pytest.approx(2 / 252)
    h = S.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert h["a"] == pytest.approx(0.03) and h["b"] == pytest.approx(0.06) and h["c"] == pytest.approx(0.06)
    a, b = np.ones((5, 100)), np.zeros((5, 100))
    assert S.hierarchical_bootstrap_diff(a, b, n_boot=200)["diff"] == 1.0


def test_train_cache_identical_to_on_the_fly(tmp_path):
    import subprocess, sys, yaml
    from src.data.dataset import TrainStream
    from src.data.generator import TrainSampleConfig
    from src.utils.logging import load_config
    cfg = load_config("configs/base.yaml")
    cfg["train"]["steps"], cfg["train"]["batch_size"] = 3, 10
    cfg["data"]["cache_dir"] = str(tmp_path)
    cp = tmp_path / "cfg.yaml"
    yaml.safe_dump(cfg, open(cp, "w"))
    subprocess.run([sys.executable, "scripts/pregenerate_train.py", "--config", str(cp), "--seeds", "7",
                    "--procs", "2"], check=True)
    dc = cfg["data"]
    scfg = TrainSampleConfig(k_min=dc["k_min"], k_max=dc["k_max"], p_definition=dc["p_definition"],
                             n_init=dc["n_init"], n_stmts=dc["n_stmts"])
    cached = TrainStream(7, scfg, 5, 25, cache_dir=tmp_path)
    fresh = TrainStream(7, scfg, 5, 25)
    assert cached.cached and not fresh.cached
    assert all(cached[i] == fresh[i] for i in range(25))
