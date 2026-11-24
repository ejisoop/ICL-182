"""Pre-generate the deterministic training stream for given seeds (CPU only).

    python scripts/pregenerate_train.py --config configs/base.yaml --seeds 0 1 2 3 4 [--procs 32]

Example i of seed s is exactly src.data.dataset.train_item(s, cfg, i); the cache
only avoids regenerating the same stream for each of the 4 architectures.
"""
from __future__ import annotations

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.dataset import cache_paths, train_item  # noqa: E402
from src.data.generator import TrainSampleConfig  # noqa: E402
from src.utils.logging import load_config  # noqa: E402

_CFG = _SEED = None


def _init(cfg, seed):
    global _CFG, _SEED
    _CFG, _SEED = cfg, seed


def _chunk(rng):
    a, b = rng
    return [np.asarray(train_item(_SEED, _CFG, i), dtype=np.uint8) for i in range(a, b)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--procs", type=int, default=32)
    a = ap.parse_args()
    cfg_all = load_config(a.config)
    dc, tc = cfg_all["data"], cfg_all["train"]
    cfg = TrainSampleConfig(k_min=dc["k_min"], k_max=dc["k_max"], p_definition=dc["p_definition"],
                            n_init=dc["n_init"], n_stmts=dc["n_stmts"],
                            ambiguous_demos=dc.get("ambiguous_demos", False))
    n = tc["steps"] * tc["batch_size"]
    for seed in a.seeds:
        tp, op = cache_paths(dc["cache_dir"], cfg, seed)
        if op.exists() and len(np.load(op, mmap_mode="r")) - 1 >= n:
            print(f"seed {seed}: cache exists ({tp})"); continue
        tp.parent.mkdir(parents=True, exist_ok=True)
        step = 20000
        with Pool(a.procs, initializer=_init, initargs=(cfg, seed)) as pool:
            parts = pool.map(_chunk, [(i, min(i + step, n)) for i in range(0, n, step)])
        seqs = [x for p in parts for x in p]
        lens = np.array([len(x) for x in seqs], dtype=np.int64)
        off = np.concatenate([[0], np.cumsum(lens)])
        np.save(tp, np.concatenate(seqs))
        np.save(op, off)
        print(f"seed {seed}: {n} examples, {off[-1]} tokens, mean len {lens.mean():.1f}, max {lens.max()} -> {tp}")


if __name__ == "__main__":
    main()
