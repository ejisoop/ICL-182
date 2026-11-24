"""Multi-query associative recall with independently sampled values."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

from .generator import case_seed
from .recall import COMMANDS, VOCAB as RECALL_VOCAB

VALUES = [f"v{i}" for i in range(64)]
VOCAB = RECALL_VOCAB + VALUES
TOK2ID = {token: i for i, token in enumerate(VOCAB)}
VALUE_IDS = [TOK2ID[value] for value in VALUES]
VOCAB_SIZE = len(VOCAB)


def encode(text: str) -> list[int]:
    return [TOK2ID[token] for token in text.split()]


@dataclass(frozen=True)
class MQARConfig:
    n_choices: Sequence[int] = (4, 8, 16, 32, 64, 128, 256)
    n_query: int = 8


@dataclass
class MQARPrompt:
    text: str
    n: int
    answers: list[str]
    ranks: list[int]


def make_prompt(rng: random.Random, n: int, n_query: int) -> MQARPrompt:
    keys = rng.sample(COMMANDS, n)
    values = rng.choices(VALUES, k=n)
    tokens = ["<bos>"]
    for key, value in zip(keys, values):
        tokens.extend(("M", key, value, ";"))
    ranks = rng.sample(range(n), min(n_query, n))
    answers = [values[rank] for rank in ranks]
    for rank, answer in zip(ranks, answers):
        tokens.extend(("Q", keys[rank], "->", answer, ";"))
    return MQARPrompt(" ".join(tokens), n, answers, ranks)


def solve(text: str) -> list[str]:
    """Parse the binding/query grammar and look up each key, ignoring the supplied answers."""
    bos, *tokens = text.split()
    if bos != "<bos>" or tokens[-1:] != [";"]:
        raise ValueError("expected a BOS token and semicolon-terminated statements")
    bindings: dict[str, str] = {}
    answers: list[str] = []
    for statement in " ".join(tokens).split(";")[:-1]:
        match statement.split():
            case ["M", key, value] if not answers and key in COMMANDS and value in VALUES:
                if key in bindings:
                    raise ValueError(f"duplicate binding for {key}")
                bindings[key] = value
            case ["Q", key, "->", _] if key in bindings:
                answers.append(bindings[key])
            case _:
                raise ValueError(f"invalid MQAR statement: {statement.strip()}")
    return answers


def sample_training_prompt(seed: int, cfg: MQARConfig) -> MQARPrompt:
    rng = random.Random(seed)
    return make_prompt(rng, rng.choice(cfg.n_choices), cfg.n_query)


def fixed_set(split: str, cfg: MQARConfig, per_n: int, base_seed: int = 12345) -> list[dict]:
    """Generate a separate deterministic stream per split and binding load."""
    rows = []
    for n in cfg.n_choices:
        for index in range(per_n):
            rng = random.Random(case_seed(base_seed, f"mqar-{split}-{n}", index))
            prompt = make_prompt(rng, n, cfg.n_query)
            rows.append({"id": f"{split}-{n}-{index}", "n": n, "text": prompt.text,
                         "answers": prompt.answers, "ranks": prompt.ranks})
    return rows


class MQARStream:
    """Map-style dataset with identical prompts across runs sharing a data seed."""

    cached = False

    def __init__(self, data_seed: int, cfg: MQARConfig, start_index: int, length: int):
        self.data_seed = data_seed
        self.cfg = cfg
        self.start_index = start_index
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        seed = case_seed(self.data_seed, "mqar-train", self.start_index + index)
        return encode(sample_training_prompt(seed, self.cfg).text)
