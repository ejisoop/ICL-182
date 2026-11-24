"""Evaluate a trained model on the fixed paired test cases.

For every test case and every demonstration count k in ``demo_counts`` we render
two ICL prompts (meaning A / B, identical query program); additionally the two
direct-definition control prompts.  The model's prediction is the argmax of the
next-token distribution after the final '->' over the *full* vocabulary (a
non-number prediction is simply wrong and is counted as an invalid output).

Output: one row per prompt (raw results), saved as CSV.
"""
from __future__ import annotations

import json
from typing import List

import pandas as pd
import torch

from src.data.dataset import collate_lm
from src.data.generator import Case
from src.data.tokenizer import NUMBER_IDS, encode, id_to_number


def load_cases(path: str) -> List[Case]:
    out = []
    for line in open(path):
        d = json.loads(line)
        d["slot_order"] = [tuple(x) for x in d["slot_order"]]
        out.append(Case(**d))
    return out


def build_eval_prompts(cases: List[Case], demo_counts: List[int]) -> pd.DataFrame:
    rows = []
    for c in cases:
        for which in "ab":
            op = c.meaning_a if which == "a" else c.meaning_b
            base = {"case_id": c.case_id, "which": which, "op": op, "target": c.target,
                    "answer": c.answer(which), "template": c.template}
            for k in demo_counts:
                rows.append({**base, "condition": "icl", "k": k, "prompt": c.render_icl(which, k)})
            rows.append({**base, "condition": "definition", "k": -1,
                         "prompt": c.render_definition(which)})
    return pd.DataFrame(rows)


@torch.no_grad()
def predict(model, prompts: List[str], device, amp_dtype=torch.bfloat16, batch_size=256):
    """Return (pred_token_id, logprob_of_answer_dist) for each prompt (prompt ends in '->')."""
    model.eval()
    seqs = [encode(p) for p in prompts]
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
    preds = [None] * len(seqs)
    num_ids = torch.tensor(NUMBER_IDS, device=device)
    for s in range(0, len(order), batch_size):
        idx = order[s:s + batch_size]
        # append a dummy token so that collate's '->' detection covers the last position
        b = collate_lm([seqs[i] + [0] for i in idx])
        inp = b["input_ids"].to(device)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
            logits = model(inp)
        la = logits[torch.arange(len(idx), device=device), b["answer_pos"].to(device)].float()
        p = la.argmax(-1).tolist()
        pnum = num_ids[la[:, num_ids].argmax(-1)].tolist()
        for j, i in enumerate(idx):
            preds[i] = (p[j], pnum[j])
    return preds


def evaluate_model(model, cases: List[Case], demo_counts: List[int], device,
                   batch_size: int = 256) -> pd.DataFrame:
    df = build_eval_prompts(cases, demo_counts)
    preds = predict(model, df["prompt"].tolist(), device, batch_size=batch_size)
    df["pred_id"] = [p[0] for p in preds]
    df["pred"] = [id_to_number(p[0]) for p in preds]
    df["valid"] = df["pred"].notna()
    df["pred"] = df["pred"].fillna(-1).astype(int)
    df["correct"] = df["pred"] == df["answer"]
    # secondary diagnostic: argmax restricted to number tokens
    df["pred_numeric_only"] = [id_to_number(p[1]) for p in preds]
    df["correct_numeric_only"] = df["pred_numeric_only"] == df["answer"]
    return df.drop(columns=["prompt"])
