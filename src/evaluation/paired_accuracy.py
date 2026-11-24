"""Paired accuracy and related per-run metrics computed from raw prediction rows."""
from __future__ import annotations

import pandas as pd


def pair_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (case_id, condition, k): pair_correct = both A and B correct."""
    g = df.groupby(["case_id", "condition", "k"])
    out = g["correct"].agg(n="size", n_correct="sum").reset_index()
    if (out["n"] != 2).any():
        raise ValueError("every pair must contain exactly two prompts (A and B)")
    out["pair_correct"] = out["n_correct"] == 2
    return out.drop(columns=["n"])


def paired_accuracy(df: pd.DataFrame) -> float:
    """number of pairs with both answers correct / number of pairs."""
    pt = pair_table(df)
    return float(pt["pair_correct"].mean())


def summarize_run(df: pd.DataFrame) -> pd.DataFrame:
    """Per (condition, k): individual accuracy, paired accuracy, n pairs, invalid rate."""
    pt = pair_table(df)
    ind = df.groupby(["condition", "k"]).agg(individual_acc=("correct", "mean"),
                                             valid_rate=("valid", "mean"),
                                             n_prompts=("correct", "size")).reset_index()
    pr = pt.groupby(["condition", "k"]).agg(paired_acc=("pair_correct", "mean"),
                                            n_pairs=("pair_correct", "size")).reset_index()
    return ind.merge(pr, on=["condition", "k"])


def accuracy_by(df: pd.DataFrame, col: str) -> pd.DataFrame:
    return df.groupby(["condition", "k", col])["correct"].mean().reset_index()
