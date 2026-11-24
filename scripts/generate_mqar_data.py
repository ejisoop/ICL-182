"""Generate fixed MQAR validation and test prompts and record their SHA-256 hashes."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.mqar import MQARConfig, fixed_set  # noqa: E402
from src.utils.logging import load_config  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mqar.yaml")
    args = parser.parse_args()
    data = load_config(args.config)["data"]
    config = MQARConfig(data["n_choices"], data["n_query"])
    manifest = {}
    for split in ("val", "test"):
        path = Path(data[f"{split}_file"])
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = fixed_set(split, config, data[f"{split}_per_n"])
        content = "".join(json.dumps(row) + "\n" for row in rows).encode("utf-8")
        path.write_bytes(content)
        manifest[split] = {
            "file": path.name,
            "sha256": hashlib.sha256(content).hexdigest(),
            "prompts": len(rows),
            "queries_per_n": {n: data[f"{split}_per_n"] * min(config.n_query, n)
                              for n in config.n_choices},
        }
    manifest_path = Path(data["test_file"]).with_name("data_manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
