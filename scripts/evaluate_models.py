"""Evaluate trained checkpoints on the fixed paired test cases.

    python scripts/evaluate_models.py --run checkpoints/main/pure_s0     # one run
    python scripts/evaluate_models.py --all checkpoints/main             # every finished run

Writes results/raw/<run_name>.csv (one row per prompt) unless it already exists.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.evaluate import evaluate_model, load_cases  # noqa: E402
from src.models import ModelConfig, build_model  # noqa: E402
from src.data.tokenizer import VOCAB_SIZE  # noqa: E402
from src.training.checkpoint import load_checkpoint  # noqa: E402
from src.utils.reproducibility import set_seed  # noqa: E402


def evaluate_run(run_dir: str | Path, raw_dir: str | Path = "results/raw", force: bool = False,
                 checkpoint: str = "final.pt") -> Path | None:
    run_dir, raw_dir = Path(run_dir), Path(raw_dir)
    out = raw_dir / f"{run_dir.name}.csv"
    if out.exists() and not force:
        print(f"[eval] exists: {out}")
        return out
    if not (run_dir / checkpoint).exists():
        print(f"[eval] missing checkpoint in {run_dir}")
        return None
    meta = json.load(open(run_dir / "metadata.json"))
    cfg = meta["config"]
    set_seed(0)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(meta["arch"], ModelConfig(vocab_size=VOCAB_SIZE, **cfg["model"])).to(dev)
    load_checkpoint(run_dir / checkpoint, model, restore_rng=False, map_location=dev)
    cases = load_cases(cfg["data"]["test_file"])
    t0 = time.time()
    df = evaluate_model(model, cases, cfg["eval"]["demo_counts"], dev, cfg["eval"]["batch_size"])
    df.insert(0, "run", run_dir.name)
    df.insert(1, "arch", meta["arch"])
    df.insert(2, "seed", meta["seed"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    s = df.groupby(["condition", "k"])["correct"].mean()
    print(f"[eval] {run_dir.name} ({time.time()-t0:.0f}s) individual acc:\n{s.to_string()}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run")
    ap.add_argument("--all")
    ap.add_argument("--raw-dir", default="results/raw")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    runs = [Path(a.run)] if a.run else sorted(p for p in Path(a.all).iterdir() if p.is_dir())
    for r in runs:
        evaluate_run(r, a.raw_dir, a.force)


if __name__ == "__main__":
    main()
