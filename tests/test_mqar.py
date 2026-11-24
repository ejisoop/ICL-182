import random

import pytest
import torch

from src.data import mqar, recall, tokenizer
from src.data.dataset import collate_lm
from src.training.train import evaluate_mqar_val


@pytest.mark.parametrize("n", mqar.MQARConfig().n_choices)
def test_queries_match_independent_solver_and_binding_ranks(n):
    prompt = mqar.make_prompt(random.Random(n), n, 8)
    tokens = prompt.text.split()
    bindings = tokens[1:1 + 4 * n]
    queries = tokens[1 + 4 * n:]
    binding_keys = bindings[1::4]
    query_keys = queries[1::5]
    assert len(binding_keys) == len(set(binding_keys)) == n
    assert len(query_keys) == len(set(query_keys)) == min(8, n)
    assert [binding_keys.index(key) for key in query_keys] == prompt.ranks
    assert mqar.solve(prompt.text) == prompt.answers == queries[3::5]
    for index in range(1 + 4 * n + 3, len(tokens), 5):
        tokens[index] = "?"
    assert mqar.solve(" ".join(tokens)) == prompt.answers


def test_vocab_preserves_collator_ids_and_recall_keys():
    assert mqar.VOCAB_SIZE == len(mqar.TOK2ID)
    assert mqar.TOK2ID["<pad>"] == tokenizer.PAD_ID
    assert mqar.TOK2ID["->"] == tokenizer.ARROW_ID
    assert len(mqar.VALUE_IDS) == 64
    assert set(mqar.VALUES).isdisjoint(recall.VOCAB)
    assert all(mqar.TOK2ID[key] == recall.TOK2ID[key] for key in recall.COMMANDS)


def test_supervision_covers_each_query_answer_and_no_other_token():
    prompts = [mqar.make_prompt(random.Random(n), n, 8) for n in (4, 16)]
    batch = collate_lm([mqar.encode(prompt.text) for prompt in prompts])
    for prompt, labels in zip(prompts, batch["labels"]):
        positions = (labels != -100).nonzero().flatten()
        assert positions.tolist() == [1 + 4 * prompt.n + 2 + 5 * q for q in range(min(8, prompt.n))]
        assert [mqar.VOCAB[label] for label in labels[positions].tolist()] == prompt.answers


def test_stream_is_deterministic_and_resumable_by_seed_and_index():
    config = mqar.MQARConfig()
    stream = mqar.MQARStream(7, config, 0, 100)
    resumed = mqar.MQARStream(7, config, 12, 88)
    other = mqar.MQARStream(8, config, 0, 100)
    assert stream[12] == resumed[0] != other[12]
    assert all(stream[12 + index] == resumed[index] for index in range(len(resumed)))
    counts = {stream[index].count(mqar.TOK2ID["M"]) for index in range(len(stream))}
    assert counts == set(config.n_choices)


def test_fixed_sets_use_separate_deterministic_streams():
    config = mqar.MQARConfig((4, 8))
    val = mqar.fixed_set("val", config, 5)
    test = mqar.fixed_set("test", config, 5)
    assert val == mqar.fixed_set("val", config, 5)
    assert {row["text"] for row in val}.isdisjoint(row["text"] for row in test)
    assert [row["n"] for row in val] == [4] * 5 + [8] * 5


def test_validation_scores_every_query_by_load():
    rows = [
        {"n": 2, "text": "<bos> M c0 v0 ; M c1 v1 ; Q c0 -> v0 ; Q c1 -> v1 ;", "answers": ["v0", "v1"]},
        {"n": 1, "text": "<bos> M c0 v0 ; Q c0 -> v0 ;", "answers": ["v0"]},
    ]
    for row in rows:
        row["ids"] = mqar.encode(row["text"])

    class PredictV0(torch.nn.Module):
        def forward(self, ids):
            logits = torch.zeros((*ids.shape, mqar.VOCAB_SIZE), device=ids.device)
            logits[..., mqar.TOK2ID["v0"]] = 1
            logits[..., mqar.TOK2ID["c0"]] = 2  # argmax is restricted to value tokens
            return logits

    scores = evaluate_mqar_val(PredictV0(), rows, torch.device("cpu"), torch.bfloat16, 2)
    assert scores["val_acc_n1"] == 1
    assert scores["val_acc_n2"] == 0.5
