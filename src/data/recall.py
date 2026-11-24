"""Load-scaling task ("recall"): many made-up commands bound in one prompt.

Motivation (EXPERIMENT_LOG Phase 8, IMPLEMENTATION_PLAN section 11): Experiments 1-2 found that
program *execution*, not rule inference, decided outcomes, and that 3 commands never stress a
recurrent state.  Here execution is a single modular addition and the load is the number N of
command bindings the model must hold until the queries.

Every prompt draws N distinct commands from a pool of POOL made-up command tokens.  Each
command means "add c" (mod 13), c uniform in 0..12, freshly drawn per prompt.  Bindings are shown
in one of two formats (the same format for the whole prompt):

    "icl"         D cmd x = y ;   with y = x + c   (meaning must be inferred: c = y - x)
    "definition"  M cmd c ;                        (meaning stated: control condition)

followed by n_query queries on distinct bound commands, in random order:

    Q cmd x -> a ;               with a = x + c          (the only supervised token is a)

Demonstrations use '=' and queries '->', so ``collate_lm`` (which supervises the token after each
'->') supervises exactly the query answers.  Chance accuracy is 1/13.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

from .generator import case_seed

MODULUS = 13
POOL = 512
SPECIAL = ["<pad>", "<bos>", "<eos>"]
MARKERS = ["D", "M", "Q", "=", ";", "->"]
NUMBERS = [str(i) for i in range(MODULUS)]
COMMANDS = [f"c{i}" for i in range(POOL)]
VOCAB: List[str] = SPECIAL + MARKERS + NUMBERS + COMMANDS
TOK2ID = {t: i for i, t in enumerate(VOCAB)}
assert len(TOK2ID) == len(VOCAB)
VOCAB_SIZE = len(VOCAB)
# collate_lm (src/data/dataset.py) uses these ids from the main tokenizer; they must coincide
from .tokenizer import ARROW_ID as _MAIN_ARROW, PAD_ID as _MAIN_PAD  # noqa: E402
assert TOK2ID["<pad>"] == _MAIN_PAD
ARROW_ID = TOK2ID["->"]
assert ARROW_ID == _MAIN_ARROW


def encode(text: str | Sequence[str]) -> List[int]:
    toks = text.split() if isinstance(text, str) else list(text)
    return [TOK2ID[t] for t in toks]


def decode(ids: Sequence[int]) -> str:
    return " ".join(VOCAB[i] for i in ids)


@dataclass
class RecallConfig:
    n_choices: Sequence[int] = (4, 8, 16, 32, 64, 128)   # training: N uniform over these
    n_query: int = 8
    p_definition: float = 0.25


@dataclass
class RecallPrompt:
    text: str
    fmt: str                      # "icl" | "definition"
    n: int
    meanings: Dict[str, int] = field(default_factory=dict)
    queries: List[tuple] = field(default_factory=list)   # (cmd, x, answer, binding_rank)


def make_prompt(rng: random.Random, n: int, fmt: str, n_query: int) -> RecallPrompt:
    cmds = rng.sample(COMMANDS, n)
    meanings = {c: rng.randrange(MODULUS) for c in cmds}
    parts = ["<bos>"]
    for c in cmds:                                   # binding order = cmds order (random)
        if fmt == "icl":
            x = rng.randrange(MODULUS)
            parts += ["D", c, str(x), "=", str((x + meanings[c]) % MODULUS), ";"]
        else:
            parts += ["M", c, str(meanings[c]), ";"]
    qc = rng.sample(range(n), min(n_query, n))       # indices of queried bindings
    queries = []
    for i in qc:
        c, x = cmds[i], rng.randrange(MODULUS)
        a = (x + meanings[c]) % MODULUS
        parts += ["Q", c, str(x), "->", str(a), ";"]
        queries.append((c, x, a, i))
    return RecallPrompt(" ".join(parts), fmt, n, meanings, queries)


def sample_training_prompt(seed: int, cfg: RecallConfig) -> RecallPrompt:
    rng = random.Random(seed)
    n = rng.choice(list(cfg.n_choices))
    fmt = "definition" if rng.random() < cfg.p_definition else "icl"
    return make_prompt(rng, n, fmt, cfg.n_query)


def train_item(data_seed: int, cfg: RecallConfig, index: int) -> List[int]:
    return encode(sample_training_prompt(case_seed(data_seed, "recall-train", index), cfg).text)


def solve(text: str) -> List[int]:
    """Independent reference: parse a prompt and return the correct answer for each query."""
    toks = text.split()
    mean, out, i = {}, [], 0
    while i < len(toks):
        t = toks[i]
        if t == "D":
            c, x, y = toks[i + 1], int(toks[i + 2]), int(toks[i + 4]); mean[c] = (y - x) % MODULUS; i += 6
        elif t == "M":
            mean[toks[i + 1]] = int(toks[i + 2]); i += 4
        elif t == "Q":
            out.append((int(toks[i + 2]) + mean[toks[i + 1]]) % MODULUS); i += 6
        else:
            i += 1
    return out


def fixed_set(split: str, ns: Sequence[int], fmts: Sequence[str], per_cell: int, n_query: int,
              base_seed: int = 12345) -> List[dict]:
    """Deterministic evaluation prompts: per_cell prompts for every (N, format)."""
    rows = []
    for n in ns:
        for fmt in fmts:
            for j in range(per_cell):
                rng = random.Random(case_seed(base_seed, f"recall-{split}-{n}-{fmt}", j))
                p = make_prompt(rng, n, fmt, n_query)
                rows.append({"text": p.text, "format": fmt, "n": n, "id": f"{split}-{n}-{fmt}-{j}",
                             "answers": [q[2] for q in p.queries],
                             "ranks": [q[3] for q in p.queries]})
    return rows


class RecallStream:
    """Map-style dataset: example i = train_item(data_seed, cfg, start_index + i). Deterministic,
    identical across architectures for one data seed, resumable from start_index."""

    cached = False

    def __init__(self, data_seed: int, cfg: RecallConfig, start_index: int, length: int):
        self.data_seed, self.cfg, self.start, self.length = data_seed, cfg, start_index, length

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        return train_item(self.data_seed, self.cfg, self.start + i)
