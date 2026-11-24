"""Fixed validation / test prompts for the load-scaling task (deterministic; hashes recorded).

    python scripts/generate_recall_data.py --config configs/recall.yaml
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import recall  # noqa: E402
from src.utils.logging import load_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/recall.yaml")
    a = ap.parse_args()
    dc = load_config(a.config)["data"]
    out = Path(dc["test_file"]).parent
    out.mkdir(parents=True, exist_ok=True)
    fmts = ("icl", "definition")
    sets = {dc["val_file"]: recall.fixed_set("val", dc["n_choices"], fmts, dc["val_per_cell"], dc["n_query"]),
            dc["test_file"]: recall.fixed_set("test", dc["n_choices"], fmts, dc["test_per_cell"], dc["n_query"])}
    manifest = {}
    for path, rows in sets.items():
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        manifest[f"sha256_{Path(path).name}"] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        manifest[f"n_{Path(path).name}"] = len(rows)
    json.dump(manifest, open(out / "data_manifest.json", "w"), indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
