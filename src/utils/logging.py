"""Minimal JSON-lines logger and config loading."""
from __future__ import annotations

import json
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


class JsonlLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: dict) -> None:
        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def _merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: str | Path, overrides: dict | None = None) -> dict:
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = ROOT / path
    cfg = yaml.safe_load(open(path))
    parent = cfg.pop("inherit", None)
    if parent:
        cfg = _merge(load_config(path.parent / parent), cfg)
    if overrides:
        cfg = _merge(cfg, overrides)
    return cfg
