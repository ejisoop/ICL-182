"""Score the fixed MQAR test set from a completed training run."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import mqar  # noqa: E402
from src.evaluation.mqar import score_queries  # noqa: E402
from src.models import build_model  # noqa: E402
from src.training.checkpoint import load_checkpoint  # noqa: E402
from src.training.train import model_config_from  # noqa: E402

FIELDS = ["id", "arch", "seed", "gdn_head_dim", "n", "query_idx", "rank", "answer", "pred", "correct"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--raw-dir", default=Path("results/raw_mqar"), type=Path)
    args = parser.parse_args()
    meta = json.loads((args.run / "metadata.json").read_text())
    cfg = meta["config"]
    steps = cfg["train"]["steps"]
    if cfg["data"].get("task") != "mqar" or meta.get("final_step") != steps:
        raise ValueError(f"{args.run} is not a completed MQAR run")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mcfg = model_config_from(cfg)
    model = build_model(meta["arch"], mcfg).to(device).eval()
    checkpoint = load_checkpoint(args.run / "final.pt", model, restore_rng=False, map_location=device)
    if checkpoint["step"] != steps:
        raise ValueError(f"expected the final step {steps}, got {checkpoint['step']}")
    del checkpoint

    path = ROOT / cfg["data"]["test_file"]
    manifest = json.loads(path.with_name("data_manifest.json").read_text())
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != manifest["test"]["sha256"]:
        raise ValueError(f"fixed MQAR test set hash mismatch: {path}")
    rows = [json.loads(line) for line in content.splitlines()]
    for row in rows:
        row["ids"] = mqar.encode(row["text"])

    args.raw_dir.mkdir(parents=True, exist_ok=True)
    output = args.raw_dir / f"{args.run.name}.csv"
    tmp = output.with_suffix(".csv.tmp")
    amp_dtype = getattr(torch, cfg["train"]["amp_dtype"])
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row, query, prediction, _ in score_queries(model, rows, device, amp_dtype, cfg["eval"]["batch_size"]):
            answer = row["answers"][query]
            pred = mqar.VALUES[prediction]
            writer.writerow({"id": row["id"], "arch": meta["arch"], "seed": meta["seed"],
                             "gdn_head_dim": mcfg.gdn_head_dim, "n": row["n"], "query_idx": query,
                             "rank": row["ranks"][query], "answer": answer, "pred": pred,
                             "correct": int(pred == answer)})
    tmp.replace(output)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
