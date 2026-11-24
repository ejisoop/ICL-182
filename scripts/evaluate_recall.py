"""Evaluate a load-scaling run on the fixed test prompts: every query answer is scored.

    python scripts/evaluate_recall.py --run checkpoints/recall/<run> [--raw-dir results/raw_recall]

Writes <raw-dir>/<run>.csv with one row per query:
id, arch, seed, n, format, query_idx, rank (position of the queried binding), answer, pred, correct.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import recall  # noqa: E402
from src.data.dataset import collate_lm  # noqa: E402
from src.models import build_model  # noqa: E402
from src.training.checkpoint import load_checkpoint  # noqa: E402
from src.training.train import model_config_from  # noqa: E402

NUM_IDS = torch.tensor([recall.TOK2ID[n] for n in recall.NUMBERS])


@torch.no_grad()
def evaluate(model, rows, device, batch_size=128):
    model.eval()
    out = []
    order = sorted(range(len(rows)), key=lambda i: len(rows[i]["text"]))
    for s in range(0, len(order), batch_size):
        idx = order[s:s + batch_size]
        b = collate_lm([recall.encode(rows[i]["text"]) for i in idx])
        inp = b["input_ids"].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(inp).float()
        # restrict prediction to number tokens (answers are always numbers)
        num_logits = logits[..., NUM_IDS.to(device)]
        pred = num_logits.argmax(-1).cpu()
        for j, i in enumerate(idx):
            pos = (b["labels"][j] != -100).nonzero().flatten().tolist()
            r = rows[i]
            assert len(pos) == len(r["answers"])
            for q, p in enumerate(pos):
                pv = int(pred[j, p])
                out.append({"id": r["id"], "n": r["n"], "format": r["format"], "query_idx": q,
                            "rank": r["ranks"][q], "answer": r["answers"][q], "pred": pv,
                            "correct": int(pv == r["answers"][q])})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--raw-dir", default="results/raw_recall")
    a = ap.parse_args()
    run = Path(a.run)
    meta = json.load(open(run / "metadata.json"))
    cfg = meta["config"]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(meta["arch"], model_config_from(cfg)).to(dev)
    load_checkpoint(run / "final.pt", model, None, restore_rng=False, map_location=dev)
    rows = [json.loads(l) for l in open(ROOT / cfg["data"]["test_file"])]
    df = pd.DataFrame(evaluate(model, rows, dev))
    df.insert(1, "arch", meta["arch"]); df.insert(2, "seed", meta["seed"])
    Path(a.raw_dir).mkdir(parents=True, exist_ok=True)
    df.to_csv(Path(a.raw_dir) / f"{run.name}.csv", index=False)
    print(df.groupby(["format", "n"]).correct.mean().unstack(0).round(3).to_string())


if __name__ == "__main__":
    main()
