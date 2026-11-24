"""Shared query scoring for MQAR validation and final test evaluation."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from src.data.dataset import collate_lm
from src.data.mqar import VALUE_IDS


@torch.no_grad()
def score_queries(model, rows, device, amp_dtype, batch_size):
    """Score each query using only the 64 legal answers for the argmax."""
    value_ids = torch.tensor(VALUE_IDS, device=device)
    order = sorted(rows, key=lambda row: len(row["ids"]))
    for start in range(0, len(order), batch_size):
        batch_rows = order[start:start + batch_size]
        batch = collate_lm([row["ids"] for row in batch_rows])
        labels = batch["labels"]
        mask = labels != -100
        inputs = batch["input_ids"].to(device)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
            logits = model(inputs)[mask.to(device)].float()
        targets = labels[mask].to(device)
        losses = F.cross_entropy(logits, targets, reduction="none").tolist()
        predictions = logits[:, value_ids].argmax(dim=-1).tolist()
        counts = mask.sum(dim=-1).tolist()
        offset = 0
        for row, count in zip(batch_rows, counts):
            for query in range(count):
                yield row, query, predictions[offset], losses[offset]
                offset += 1
