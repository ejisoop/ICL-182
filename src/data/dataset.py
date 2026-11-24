"""PyTorch datasets / collation.

Training data is generated on the fly: example ``i`` of the stream for data
seed ``s`` is ``sample_training_prompt(case_seed(s, 'train', i))``.  This makes
the stream infinite, deterministic, identical across architectures for the same
seed, and trivially resumable (start at index ``step * batch_size``).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np

import torch
from torch.utils.data import Dataset

from .generator import TrainSampleConfig, case_seed, sample_training_prompt
from .tokenizer import ARROW_ID, PAD_ID, encode


def train_item(data_seed: int, cfg: TrainSampleConfig, index: int) -> List[int]:
    text, _ = sample_training_prompt(case_seed(data_seed, "train", index), cfg)
    return encode(text)


def cache_tag(cfg: TrainSampleConfig) -> str:
    import hashlib
    key = f"v2|{cfg.k_min}|{cfg.k_max}|{cfg.p_definition}|{cfg.n_init}|{cfg.n_stmts}"
    if cfg.ambiguous_demos:          # unchanged key for the original task (existing caches)
        key += "|ambiguous"
    return hashlib.md5(key.encode()).hexdigest()[:10]


def cache_paths(cache_dir: str | Path, cfg: TrainSampleConfig, data_seed: int):
    base = Path(cache_dir) / f"{cache_tag(cfg)}_s{data_seed}"
    return base.with_name(base.name + "_tokens.npy"), base.with_name(base.name + "_offsets.npy")


class TrainStream(Dataset):
    """Example i of the stream = train_item(data_seed, cfg, start_index + i).

    If a pre-generated cache (scripts/pregenerate_train.py) covering the
    requested range exists it is memory-mapped; its content is byte-identical
    to on-the-fly generation (checked in tests), it merely saves CPU."""

    def __init__(self, data_seed: int, cfg: TrainSampleConfig, start_index: int, length: int,
                 cache_dir: str | Path | None = None):
        self.data_seed, self.cfg = data_seed, cfg
        self.start, self.length = start_index, length
        self.tok = self.off = None
        if cache_dir is not None:
            tp, op = cache_paths(cache_dir, cfg, data_seed)
            if tp.exists() and op.exists():
                off = np.load(op, mmap_mode="r")
                if len(off) - 1 >= start_index + length:
                    self.tok, self.off = np.load(tp, mmap_mode="r"), off

    @property
    def cached(self) -> bool:
        return self.tok is not None

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        j = self.start + i
        if self.tok is not None:
            return self.tok[self.off[j]:self.off[j + 1]].astype(np.int64).tolist()
        return train_item(self.data_seed, self.cfg, j)


class TokenListDataset(Dataset):
    def __init__(self, seqs: List[List[int]]):
        self.seqs = seqs

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        return self.seqs[i]


def collate_lm(batch: List[List[int]]):
    """Right-pad; build next-token targets supervised only at result positions.

    A position t is supervised iff input token t is '->' (the next token is an
    interpreter-determined result: a demonstration result or the final answer).
    ``answer_pos`` is the last '->' in each sequence (the query answer).
    """
    T = max(len(s) for s in batch)
    ids = torch.full((len(batch), T), PAD_ID, dtype=torch.long)
    for i, s in enumerate(batch):
        ids[i, :len(s)] = torch.tensor(s)
    inp, tgt = ids[:, :-1], ids[:, 1:].clone()
    sup = inp == ARROW_ID
    labels = torch.where(sup, tgt, torch.full_like(tgt, -100))
    # last arrow position
    idx = torch.arange(inp.shape[1]).expand_as(inp)
    answer_pos = torch.where(sup, idx, torch.full_like(idx, -1)).max(dim=1).values
    return {"input_ids": inp, "labels": labels, "answer_pos": answer_pos,
            "answer": tgt[torch.arange(len(batch)), answer_pos]}


def load_val(path: str):
    """Validation file: jsonl with fields text, format, k, op."""
    rows = [json.loads(l) for l in open(path)]
    return rows
