"""Generate the fixed validation set and the fixed paired test cases.

    python scripts/generate_data.py --config configs/base.yaml [--n-test-cases 1000] [--n-val 3000]

Outputs (directory taken from data.test_file in the config)
  data/val.jsonl          validation prompts (val templates), mixed formats
  data/test_cases.jsonl   paired evaluation cases (test templates), k_max=8
  data/data_manifest.json seeds, counts, hashes
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.generator import (TrainSampleConfig, case_seed, sample_case,  # noqa: E402
                                sample_training_prompt)
from src.utils.logging import load_config  # noqa: E402

VAL_SEED = 777_777
TEST_SEED = 20_241_118


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-test-cases", type=int, default=1000)
    ap.add_argument("--n-val", type=int, default=3000)
    ap.add_argument("--config", default="configs/base.yaml")
    a = ap.parse_args()
    dc = load_config(a.config)["data"]
    out = Path(dc["test_file"]).parent
    assert Path(dc["val_file"]).parent == out
    out.mkdir(parents=True, exist_ok=True)

    cfg = TrainSampleConfig(k_min=dc["k_min"], k_max=dc["k_max"], p_definition=dc["p_definition"],
                            n_init=dc["n_init"], n_stmts=dc["n_stmts"],
                            ambiguous_demos=dc.get("ambiguous_demos", False))
    with open(out / "val.jsonl", "w") as f:
        for i in range(a.n_val):
            text, meta = sample_training_prompt(case_seed(VAL_SEED, "val", i), cfg, split="val")
            f.write(json.dumps({"text": text, **meta}) + "\n")

    with open(out / "test_cases.jsonl", "w") as f:
        for i in range(a.n_test_cases):
            c = sample_case(random.Random(case_seed(TEST_SEED, "test", i)), "test", cfg.k_max,
                            case_id=f"test{i:05d}", task=cfg.task)
            f.write(json.dumps(asdict(c)) + "\n")

    manifest = {"val_seed": VAL_SEED, "test_seed": TEST_SEED, "n_val": a.n_val,
                "n_test_cases": a.n_test_cases, "data_config": dc}
    for fn in ("val.jsonl", "test_cases.jsonl"):
        manifest[f"sha256_{fn}"] = hashlib.sha256((out / fn).read_bytes()).hexdigest()
    json.dump(manifest, open(out / "data_manifest.json", "w"), indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
