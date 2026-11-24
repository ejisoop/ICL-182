"""Fixed word-level tokenizer for the synthetic program-execution task.

Every token in the task language is a whitespace-separated word, so the
tokenizer is a simple closed vocabulary lookup.  The vocabulary is fixed and
independent of the data split, so tokenisation is identical for every model.
"""
from __future__ import annotations

from typing import List, Sequence

MODULUS = 13  # arithmetic is performed modulo this prime; values are 0..12

SPECIAL = ["<pad>", "<bos>", "<eos>"]
MARKERS = ["D", "M", "Q", "=", ";", "->", ".", "means"]
OPS = ["set", "add", "sub", "mul"]  # built-in operations (fixed meaning)
COMMANDS = ["dax", "wug", "blick", "fep", "zup", "kiki", "toma", "lorp"]  # made-up
VARIABLES = list("abcdefgh")
NUMBERS = [str(i) for i in range(MODULUS)]

VOCAB: List[str] = SPECIAL + MARKERS + OPS + COMMANDS + VARIABLES + NUMBERS
TOK2ID = {t: i for i, t in enumerate(VOCAB)}
assert len(TOK2ID) == len(VOCAB), "duplicate token in vocabulary"

PAD_ID = TOK2ID["<pad>"]
BOS_ID = TOK2ID["<bos>"]
EOS_ID = TOK2ID["<eos>"]
ARROW_ID = TOK2ID["->"]
NUMBER_IDS = [TOK2ID[n] for n in NUMBERS]
VOCAB_SIZE = len(VOCAB)


def encode(text: str | Sequence[str]) -> List[int]:
    toks = text.split() if isinstance(text, str) else list(text)
    try:
        return [TOK2ID[t] for t in toks]
    except KeyError as e:  # pragma: no cover - defensive
        raise ValueError(f"unknown token {e} in {toks}") from None


def decode(ids: Sequence[int]) -> str:
    return " ".join(VOCAB[i] for i in ids)


def id_to_number(i: int) -> int | None:
    """Map a token id to its integer value, or None if it is not a number."""
    tok = VOCAB[i]
    return int(tok) if tok.isdigit() else None
