"""Statistical analysis of the raw evaluation outputs (pre-registered plan, IMPLEMENTATION_PLAN.md §6).

    python scripts/analyze_results.py [--raw-dir results/raw] [--ckpt-dir checkpoints/main]

Writes results/tables/*.csv|md and results/stats.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation import statistics as S  # noqa: E402
from src.evaluation.paired_accuracy import pair_table, summarize_run  # noqa: E402
from src.utils.logging import read_jsonl  # noqa: E402

ARCH_ORDER = ["pure", "hybrid_early", "hybrid_middle", "hybrid_late"]
HYBRIDS = ARCH_ORDER[1:]
PRIMARY_KS = [1, 2, 3, 4, 6, 8]
SMALL_KS = [1, 2]


def load_raw(raw_dir: Path) -> pd.DataFrame:
    dfs = [pd.read_csv(f) for f in sorted(raw_dir.glob("*.csv"))]
    if not dfs:
        raise SystemExit(f"no raw results in {raw_dir}")
    return pd.concat(dfs, ignore_index=True)


def fmt(m, lo=None, hi=None, pct=True):
    f = (lambda x: f"{100*x:.1f}") if pct else (lambda x: f"{x:.3f}")
    return f"{f(m)} [{f(lo)}, {f(hi)}]" if lo is not None else f(m)


def df_to_md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="results/raw")
    ap.add_argument("--ckpt-dir", default="checkpoints/main")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    raw_dir, out = Path(a.raw_dir), Path(a.out)
    tdir = out / "tables"
    tdir.mkdir(parents=True, exist_ok=True)
    raw = load_raw(raw_dir)
    archs = [x for x in ARCH_ORDER if x in set(raw["arch"])]
    res: dict = {"archs": archs, "primary_ks": PRIMARY_KS,
                 "seeds": {x: sorted(raw.loc[raw.arch == x, "seed"].unique().tolist()) for x in archs}}

    # ---------------- per-run summaries --------------------------------------
    per_run = []
    pairs_all = []
    for (arch, seed), d in raw.groupby(["arch", "seed"]):
        s = summarize_run(d)
        s.insert(0, "arch", arch); s.insert(1, "seed", seed)
        per_run.append(s)
        pt = pair_table(d)
        pt.insert(0, "arch", arch); pt.insert(1, "seed", seed)
        pairs_all.append(pt)
    per_run = pd.concat(per_run, ignore_index=True)
    pairs_all = pd.concat(pairs_all, ignore_index=True)
    per_run.to_csv(tdir / "per_run_condition_k.csv", index=False)

    # seed-level endpoints
    icl = per_run[per_run.condition == "icl"]
    dfn = per_run[per_run.condition == "definition"]
    ep = (icl[icl.k.isin(PRIMARY_KS)].groupby(["arch", "seed"])["paired_acc"].mean()
          .rename("icl_primary").to_frame())
    ep["icl_small_k"] = icl[icl.k.isin(SMALL_KS)].groupby(["arch", "seed"])["paired_acc"].mean()
    ep["icl_primary_individual"] = icl[icl.k.isin(PRIMARY_KS)].groupby(["arch", "seed"])["individual_acc"].mean()
    ep["definition_paired"] = dfn.groupby(["arch", "seed"])["paired_acc"].mean()
    ep["definition_individual"] = dfn.groupby(["arch", "seed"])["individual_acc"].mean()
    ep["icl_minus_definition"] = ep["icl_primary"] - ep["definition_paired"]
    ep = ep.reset_index()
    ep.to_csv(tdir / "seed_endpoints.csv", index=False)

    # ---------------- Table: accuracy by arch x k (seed-level mean, sd, CI) ---
    rows = []
    for (arch, cond, k), g in per_run.groupby(["arch", "condition", "k"]):
        for metric in ("paired_acc", "individual_acc", "valid_rate"):
            m = S.mean_ci(g[metric].values)
            rows.append({"arch": arch, "condition": cond, "k": k, "metric": metric, **m})
    acc_tab = pd.DataFrame(rows)
    acc_tab.to_csv(tdir / "accuracy_by_arch_k.csv", index=False)

    def wide(metric):
        t = acc_tab[acc_tab.metric == metric].copy()
        t["cell"] = [f"{100*m:.1f} ± {100*s:.1f}" for m, s in zip(t["mean"], t["sd"])]
        t["col"] = [("def" if c == "definition" else f"k={k}") for c, k in zip(t.condition, t.k)]
        w = t.pivot(index="arch", columns="col", values="cell")
        cols = [f"k={k}" for k in sorted(per_run[per_run.condition == 'icl'].k.unique())] + ["def"]
        return w.reindex(index=archs, columns=cols).reset_index()

    md = ["## Paired accuracy (%) — mean ± SD across seeds\n", df_to_md(wide("paired_acc")),
          "\n\n## Individual accuracy (%) — mean ± SD across seeds\n", df_to_md(wide("individual_acc"))]
    (tdir / "accuracy_by_arch_k.md").write_text("\n".join(md) + "\n")

    # ---------------- endpoint summary table ----------------------------------
    rows = []
    for arch in archs:
        g = ep[ep.arch == arch]
        r = {"arch": arch, "n_seeds": len(g)}
        for col in ("icl_primary", "icl_small_k", "icl_primary_individual", "definition_paired",
                    "icl_minus_definition"):
            m = S.mean_ci(g[col].values)
            r[col] = fmt(m["mean"], m["ci_low"], m["ci_high"])
            res.setdefault("endpoints", {}).setdefault(arch, {})[col] = m
        rows.append(r)
    ep_tab = pd.DataFrame(rows)
    ep_tab.to_csv(tdir / "endpoints_summary.csv", index=False)
    (tdir / "endpoints_summary.md").write_text(
        "Mean [95% t-CI across seeds], in %. icl_primary = ICL paired accuracy averaged over k in "
        f"{PRIMARY_KS}; icl_small_k over k in {SMALL_KS}.\n\n" + df_to_md(ep_tab) + "\n")

    # ---------------- per-case matrices for hierarchical bootstrap -----------
    def case_matrix(arch, cond, ks):
        p = pairs_all[(pairs_all.arch == arch) & (pairs_all.condition == cond)]
        if ks is not None:
            p = p[p.k.isin(ks)]
        m = p.groupby(["seed", "case_id"])["pair_correct"].mean().unstack("case_id")
        return m.sort_index(axis=1).values

    # ---------------- H1: each hybrid vs pure ----------------------------------
    h1 = {}
    if "pure" in archs:
        pure = ep[ep.arch == "pure"].set_index("seed")
        for endpoint, cond, ks in (("icl_primary", "icl", PRIMARY_KS), ("icl_small_k", "icl", SMALL_KS),
                                   ("definition_paired", "definition", None)):
            h1[endpoint] = {}
            for h in [x for x in HYBRIDS if x in archs]:
                hy = ep[ep.arch == h].set_index("seed")
                w = S.welch(hy[endpoint].values, pure[endpoint].values)
                common = sorted(set(hy.index) & set(pure.index))
                paired_t = sps.ttest_rel(hy.loc[common, endpoint], pure.loc[common, endpoint]) \
                    if len(common) > 1 else None
                boot = S.hierarchical_bootstrap_diff(case_matrix(h, cond, ks), case_matrix("pure", cond, ks),
                                                     n_boot=5000, seed=1)
                h1[endpoint][h] = {
                    "hybrid_mean": float(hy[endpoint].mean()), "pure_mean": float(pure[endpoint].mean()),
                    "welch": w, "permutation_p": S.permutation_test(hy[endpoint].values, pure[endpoint].values),
                    "cohens_d": S.cohens_d(hy[endpoint].values, pure[endpoint].values),
                    "relative_improvement": float(w["diff"] / pure[endpoint].mean()) if pure[endpoint].mean() > 0 else None,
                    "seed_paired_t_p": float(paired_t.pvalue) if paired_t is not None else None,
                    "hier_bootstrap": boot}
            holm = S.holm({h: v["welch"]["p"] for h, v in h1[endpoint].items()
                           if not np.isnan(v["welch"]["p"])})
            for h in h1[endpoint]:
                h1[endpoint][h]["welch_p_holm"] = holm.get(h)
        # per-k exploratory
        per_k = []
        for k in sorted(icl.k.unique()):
            pk = icl[(icl.k == k) & (icl.arch == "pure")]["paired_acc"].values
            for h in [x for x in HYBRIDS if x in archs]:
                hk = icl[(icl.k == k) & (icl.arch == h)]["paired_acc"].values
                w = S.welch(hk, pk)
                per_k.append({"k": int(k), "hybrid": h, "diff": w["diff"], "ci_low": w["ci_low"],
                              "ci_high": w["ci_high"], "p_uncorrected": w["p"]})
        res["h1_per_k_exploratory"] = per_k
        pd.DataFrame(per_k).to_csv(tdir / "h1_per_k_differences.csv", index=False)
    res["h1"] = h1

    # ---------------- H2: placement --------------------------------------------
    hy_present = [x for x in HYBRIDS if x in archs]
    if len(hy_present) >= 2:
        groups = {h: ep[ep.arch == h]["icl_primary"].values for h in hy_present}
        res["h2"] = {"anova": S.one_way_anova(groups), "pairwise": {}}
        pw = {}
        for i, x in enumerate(hy_present):
            for y in hy_present[i + 1:]:
                pw[f"{x}_vs_{y}"] = {"welch": S.welch(groups[x], groups[y]),
                                     "permutation_p": S.permutation_test(groups[x], groups[y])}
        holm = S.holm({k: v["welch"]["p"] for k, v in pw.items() if not np.isnan(v["welch"]["p"])})
        for k in pw:
            pw[k]["welch_p_holm"] = holm.get(k)
        res["h2"]["pairwise"] = pw

    # ---------------- H3: interaction ------------------------------------------
    if "pure" in archs:
        h3 = {}
        pd_ = ep[ep.arch == "pure"]["icl_minus_definition"].values
        for h in hy_present:
            hd = ep[ep.arch == h]["icl_minus_definition"].values
            h3[h] = {"advantage_icl": h1["icl_primary"][h]["welch"]["diff"],
                     "advantage_definition": h1["definition_paired"][h]["welch"]["diff"],
                     "interaction_welch": S.welch(hd, pd_),
                     "interaction_permutation_p": S.permutation_test(hd, pd_)}
        holm = S.holm({h: v["interaction_welch"]["p"] for h, v in h3.items()
                       if not np.isnan(v["interaction_welch"]["p"])})
        for h in h3:
            h3[h]["interaction_p_holm"] = holm.get(h)
        res["h3"] = h3

    # ---------------- comparison table (markdown) --------------------------------
    if h1:
        rows = []
        for endpoint in h1:
            for h, v in h1[endpoint].items():
                w = v["welch"]; b = v["hier_bootstrap"]
                rows.append({"endpoint": endpoint, "hybrid": h,
                             "diff (pp)": f"{100*w['diff']:+.1f}",
                             "95% CI seeds (pp)": f"[{100*w['ci_low']:+.1f}, {100*w['ci_high']:+.1f}]",
                             "95% CI hier. bootstrap (pp)": f"[{100*b['ci_low']:+.1f}, {100*b['ci_high']:+.1f}]",
                             "Welch p": f"{w['p']:.3g}", "Holm p": f"{v['welch_p_holm']:.3g}" if v['welch_p_holm'] is not None else "nan",
                             "perm. p": f"{v['permutation_p']:.3g}", "Cohen d": f"{v['cohens_d']:.2f}"})
        cmp = pd.DataFrame(rows)
        cmp.to_csv(tdir / "h1_comparisons.csv", index=False)
        (tdir / "h1_comparisons.md").write_text(df_to_md(cmp) + "\n")
    if "h3" in res:
        rows = [{"hybrid": h, "adv. ICL (pp)": f"{100*v['advantage_icl']:+.1f}",
                 "adv. definition (pp)": f"{100*v['advantage_definition']:+.1f}",
                 "interaction (pp)": f"{100*v['interaction_welch']['diff']:+.1f}",
                 "95% CI (pp)": f"[{100*v['interaction_welch']['ci_low']:+.1f}, {100*v['interaction_welch']['ci_high']:+.1f}]",
                 "Welch p": f"{v['interaction_welch']['p']:.3g}", "Holm p": f"{v['interaction_p_holm']:.3g}" if v['interaction_p_holm'] is not None else "nan",
                 "perm. p": f"{v['interaction_permutation_p']:.3g}"} for h, v in res["h3"].items()]
        pd.DataFrame(rows).to_csv(tdir / "h3_interaction.csv", index=False)
        (tdir / "h3_interaction.md").write_text(df_to_md(pd.DataFrame(rows)) + "\n")
    if "h2" in res:
        rows = [{"comparison": k, "diff (pp)": f"{100*v['welch']['diff']:+.1f}",
                 "95% CI (pp)": f"[{100*v['welch']['ci_low']:+.1f}, {100*v['welch']['ci_high']:+.1f}]",
                 "Welch p": f"{v['welch']['p']:.3g}", "Holm p": f"{v['welch_p_holm']:.3g}" if v['welch_p_holm'] is not None else "nan",
                 "perm. p": f"{v['permutation_p']:.3g}"} for k, v in res["h2"]["pairwise"].items()]
        a_ = res["h2"]["anova"]
        (tdir / "h2_placement.md").write_text(
            f"One-way ANOVA across placements: F = {a_.get('F', float('nan')):.3f}, p = {a_.get('p', float('nan')):.3g}; "
            f"Kruskal-Wallis H = {a_.get('kruskal_H', float('nan')):.3f}, p = {a_.get('kruskal_p', float('nan')):.3g}\n\n"
            + df_to_md(pd.DataFrame(rows)) + "\n")

    # ---------------- breakdowns -----------------------------------------------
    r = raw[raw.condition == "icl"]
    r = r[r.k.isin(PRIMARY_KS)]
    by_op = r.groupby(["arch", "seed", "op"])["correct"].mean().groupby(["arch", "op"]).agg(["mean", "std"]).reset_index()
    by_op.to_csv(tdir / "accuracy_by_operation.csv", index=False)
    by_cmd = r.groupby(["arch", "seed", "target"])["correct"].mean().groupby(["arch", "target"]).agg(["mean", "std"]).reset_index()
    by_cmd.to_csv(tdir / "accuracy_by_command.csv", index=False)
    d_op = raw[raw.condition == "definition"].groupby(["arch", "seed", "op"])["correct"].mean() \
        .groupby(["arch", "op"]).agg(["mean", "std"]).reset_index()
    d_op.to_csv(tdir / "definition_accuracy_by_operation.csv", index=False)
    ep.sort_values(["arch", "seed"]).to_csv(tdir / "accuracy_by_seed.csv", index=False)
    bo = by_op.copy(); bo["cell"] = [f"{100*m:.1f} ± {100*s:.1f}" for m, s in zip(bo["mean"], bo["std"])]
    (tdir / "accuracy_by_operation.md").write_text(
        "Individual ICL accuracy (%) by the target command's true operation, k in 1..8, mean ± SD across seeds\n\n"
        + df_to_md(bo.pivot(index="arch", columns="op", values="cell").reindex(archs).reset_index()) + "\n")
    bc = by_cmd.copy(); bc["cell"] = [f"{100*m:.1f}" for m in bc["mean"]]
    (tdir / "accuracy_by_command.md").write_text(
        "Individual ICL accuracy (%) by target command token, k in 1..8, mean across seeds\n\n"
        + df_to_md(bc.pivot(index="arch", columns="target", values="cell").reindex(archs).reset_index()) + "\n")

    # ---------------- training metadata ----------------------------------------
    ck = Path(a.ckpt_dir)
    rows = []
    for d in sorted(ck.glob("*_s*")):
        if not (d / "metadata.json").exists():
            continue
        m = json.load(open(d / "metadata.json"))
        log = read_jsonl(d / "train_log.jsonl") if (d / "train_log.jsonl").exists() else []
        tr = [x for x in log if "train_loss" in x]
        va = [x for x in log if "val_loss_all" in x]
        rows.append({"arch": m["arch"], "seed": m["seed"], "n_params": m["n_params"],
                     "attention_layers": str(m["attention_layers"]), "final_step": m.get("final_step"),
                     "train_minutes": round(m.get("train_seconds", float("nan")) / 60, 1),
                     "final_train_loss": tr[-1]["train_loss"] if tr else None,
                     "final_val_answer_loss": va[-1]["val_loss_all"] if va else None,
                     "final_val_acc_icl": va[-1].get("val_acc_icl") if va else None,
                     "final_val_acc_definition": va[-1].get("val_acc_definition") if va else None,
                     "gpu": m["env"].get("gpu")})
    if rows:
        tm = pd.DataFrame(rows)
        tm.to_csv(tdir / "training_summary.csv", index=False)
        agg = tm.groupby("arch").agg(n_params=("n_params", "first"), n_runs=("seed", "size"),
                                     train_min=("train_minutes", "mean"),
                                     final_train_loss=("final_train_loss", "mean"),
                                     final_val_loss=("final_val_answer_loss", "mean"),
                                     final_val_loss_sd=("final_val_answer_loss", "std"),
                                     val_acc_icl=("final_val_acc_icl", "mean"),
                                     val_acc_def=("final_val_acc_definition", "mean")).reindex(archs).reset_index()
        (tdir / "training_summary.md").write_text(df_to_md(agg.round(4)) + "\n")
        res["training"] = agg.to_dict(orient="records")

    json.dump(res, open(out / "stats.json", "w"), indent=2, default=float)
    print((tdir / "endpoints_summary.md").read_text())
    if (tdir / "h1_comparisons.md").exists():
        print((tdir / "h1_comparisons.md").read_text())
    if (tdir / "h3_interaction.md").exists():
        print((tdir / "h3_interaction.md").read_text())
    if (tdir / "h2_placement.md").exists():
        print((tdir / "h2_placement.md").read_text())


if __name__ == "__main__":
    main()
