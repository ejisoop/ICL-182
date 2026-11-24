"""Experiment 2 analysis (pre-registered in IMPLEMENTATION_PLAN.md §9).

    python scripts/analyze_exp2.py [--raw-dir results/raw_amb] [--ckpt-dir checkpoints/amb] \
        [--test-file data/amb/test_cases.jsonl] [--out results/amb]

1. runs the Experiment-1 pipeline (scripts/analyze_results.py) on the Experiment-2 raw
   predictions -> <out>/tables/*, <out>/stats.json (descriptives + unpaired tests);
2. primary seed-paired tests (H1, H1 small-k, H3; Holm over the 3 hybrids), Friedman + paired
   placement comparisons (H2) -> <out>/tables/exp2_paired.md, <out>/stats_exp2.json;
3. exploratory: accuracy split by whether the shown demonstrations identify the meaning
   -> <out>/tables/exp2_identifiability.{csv,md}.
"""
from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import interpreter  # noqa: E402
from src.evaluation.evaluate import load_cases  # noqa: E402
from src.evaluation.statistics import holm  # noqa: E402

HYBRIDS = ["hybrid_early", "hybrid_middle", "hybrid_late"]
ARCHS = ["pure"] + HYBRIDS


def signflip_p(d: np.ndarray) -> float:
    """Exact two-sided sign-flip permutation p for the mean of paired differences."""
    d = np.asarray(d, float)
    obs = abs(d.mean())
    means = [abs((d * np.array(s)).mean()) for s in itertools.product([1, -1], repeat=len(d))]
    return float(np.mean([m >= obs - 1e-12 for m in means]))


def paired(a: np.ndarray, b: np.ndarray) -> dict:
    d = np.asarray(a, float) - np.asarray(b, float)
    n = len(d)
    m, se = d.mean(), d.std(ddof=1) / np.sqrt(n)
    h = st.t.ppf(0.975, n - 1) * se
    p = float(st.ttest_rel(a, b).pvalue) if np.any(d != d[0]) else (0.0 if d[0] != 0 else 1.0)
    return {"diff": float(m), "ci_low": float(m - h), "ci_high": float(m + h), "t_p": p,
            "signflip_p": signflip_p(d), "n_pos": int((d > 0).sum()), "n": n,
            "per_seed": [float(x) for x in d]}


def fmt_pp(x):
    return f"{x * 100:+.1f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="results/raw_amb")
    ap.add_argument("--ckpt-dir", default="checkpoints/amb")
    ap.add_argument("--test-file", default="data/amb/test_cases.jsonl")
    ap.add_argument("--out", default="results/amb")
    ap.add_argument("--skip-base", action="store_true", help="do not rerun analyze_results.py")
    a = ap.parse_args()
    out = Path(a.out)
    if not a.skip_base:
        subprocess.run([sys.executable, str(ROOT / "scripts/analyze_results.py"), "--raw-dir", a.raw_dir,
                        "--ckpt-dir", a.ckpt_dir, "--out", a.out], check=True)
    tdir = out / "tables"
    ep = pd.read_csv(tdir / "seed_endpoints.csv")
    seeds = sorted(set.intersection(*[set(ep[ep.arch == x].seed) for x in ARCHS]))
    piv = {m: ep.pivot(index="seed", columns="arch", values=m).loc[seeds]
           for m in ("icl_primary", "icl_small_k", "definition_paired", "icl_minus_definition")}

    res = {"seeds": seeds, "h1": {}, "h1_small_k": {}, "h3": {}, "h2": {}}
    for key, metric in (("h1", "icl_primary"), ("h1_small_k", "icl_small_k"),
                        ("h3", "icl_minus_definition"), ("definition", "definition_paired")):
        res.setdefault(key, {})
        for h in HYBRIDS:
            res[key][h] = paired(piv[metric][h].values, piv[metric]["pure"].values)
        if key != "definition":
            ph = holm({h: res[key][h]["t_p"] for h in HYBRIDS})
            for h in HYBRIDS:
                res[key][h]["t_p_holm"] = float(ph[h])

    # H2: Friedman across placements (repeated measures over seeds) + paired comparisons
    x = piv["icl_primary"][HYBRIDS].values
    fr = st.friedmanchisquare(*[x[:, i] for i in range(3)])
    res["h2"]["friedman"] = {"chi2": float(fr.statistic), "p": float(fr.pvalue)}
    pairs = list(itertools.combinations(HYBRIDS, 2))
    comp = {f"{p}_vs_{q}": paired(piv["icl_primary"][p].values, piv["icl_primary"][q].values)
            for p, q in pairs}
    ph = holm({k: v["t_p"] for k, v in comp.items()})
    for k, v in comp.items():
        v["t_p_holm"] = float(ph[k])
    res["h2"]["pairwise"] = comp

    lines = [f"Seed-paired tests (pre-registered primary for Experiment 2), seeds {seeds}. "
             "diff = hybrid - pure, percentage points; 95% t-CI over seed differences; "
             "sign-flip p is exact (2^n patterns).\n",
             "| test | hybrid | diff (pp) | 95% CI (pp) | per-seed diffs (pp) | hybrid>pure | paired t p | Holm p | sign-flip p |",
             "|---|---|---|---|---|---|---|---|---|"]
    names = {"h1": "H1 ICL k=1-8 (primary)", "h1_small_k": "H1 ICL k=1-2",
             "definition": "definition (control)", "h3": "H3 interaction (ICL-def)"}
    for key in ("h1", "h1_small_k", "definition", "h3"):
        for h in HYBRIDS:
            r = res[key][h]
            lines.append(f"| {names[key]} | {h} | {fmt_pp(r['diff'])} | [{fmt_pp(r['ci_low'])}, "
                         f"{fmt_pp(r['ci_high'])}] | {', '.join(fmt_pp(v) for v in r['per_seed'])} | "
                         f"{r['n_pos']}/{r['n']} | {r['t_p']:.3g} | "
                         f"{r.get('t_p_holm', float('nan')):.3g} | {r['signflip_p']:.3g} |")
    lines += ["", f"H2: Friedman across placements chi2 = {res['h2']['friedman']['chi2']:.3f}, "
              f"p = {res['h2']['friedman']['p']:.3g}", "",
              "| comparison | diff (pp) | 95% CI (pp) | paired t p | Holm p | sign-flip p |",
              "|---|---|---|---|---|---|"]
    for k, r in comp.items():
        lines.append(f"| {k} | {fmt_pp(r['diff'])} | [{fmt_pp(r['ci_low'])}, {fmt_pp(r['ci_high'])}] | "
                     f"{r['t_p']:.3g} | {r['t_p_holm']:.3g} | {r['signflip_p']:.3g} |")
    (tdir / "exp2_paired.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    # exploratory: identifiability of the target meaning from the shown demonstrations
    cases = {c.case_id: c for c in load_cases(a.test_file)}
    ident = {}
    for cid, c in cases.items():
        for w in "ab":
            true = c.meanings(w)[c.target]
            for k in (1, 2, 3, 4, 6, 8):
                s = interpreter.infer_meanings_from_demos(c.render_icl(w, k))[c.target]
                ident[(cid, w, k)] = s == {true}
    rows = []
    for f in sorted(Path(a.raw_dir).glob("*.csv")):
        df = pd.read_csv(f)
        df = df[(df.condition == "icl") & (df.k >= 1)].copy()
        df["identifiable"] = [ident[(c, w, int(k))] for c, w, k in zip(df.case_id, df.which, df.k)]
        g = df.groupby(["k", "identifiable"])["correct"].agg(["mean", "size"]).reset_index()
        g.insert(0, "arch", df.arch.iloc[0]); g.insert(1, "seed", int(df.seed.iloc[0]))
        rows.append(g)
    idf = pd.concat(rows)
    idf.to_csv(tdir / "exp2_identifiability.csv", index=False)
    agg = (idf.groupby(["arch", "identifiable", "seed"]).apply(
        lambda g: np.average(g["mean"], weights=g["size"]), include_groups=False)
        .groupby(["arch", "identifiable"]).agg(["mean", "std"]) * 100)
    md = ["Individual ICL accuracy (%, k = 1-8 pooled; mean +- SD over seeds) split by whether the "
          "shown demonstrations identify the target meaning (exploratory).\n",
          "| arch | identifiable | not identifiable |", "|---|---|---|"]
    for arch in ARCHS:
        if arch in agg.index.get_level_values(0):
            r = agg.loc[arch]
            cell = lambda b: (f"{r.loc[b, 'mean']:.1f} ± {r.loc[b, 'std']:.1f}" if b in r.index else "-")
            md.append(f"| {arch} | {cell(True)} | {cell(False)} |")
    frac = idf.groupby(["k", "identifiable"])["size"].sum().unstack(fill_value=0)
    md += ["", "Fraction of ICL prompts with identifiable meaning, by k: " +
           ", ".join(f"k={k}: {frac.loc[k].get(True, 0) / frac.loc[k].sum():.2f}" for k in frac.index)]
    (tdir / "exp2_identifiability.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    res["identifiability"] = {f"{arch}|{b}": {"mean": float(agg.loc[(arch, b), "mean"]),
                                             "sd": float(agg.loc[(arch, b), "std"])}
                              for arch, b in agg.index}
    # exploratory (post hoc): per-run take-off status, k=1 accuracy vs the 2-op ceiling, concordance
    runs = []
    for f in sorted(Path(a.raw_dir).glob("*.csv")):
        df = pd.read_csv(f)
        arch, seed = df.arch.iloc[0], int(df.seed.iloc[0])
        g = idf[(idf.arch == arch) & (idf.seed == seed)]
        wavg = lambda m: float(np.average(g[m]["mean"], weights=g[m]["size"])) if m.any() else float("nan")
        dfn = float(df[df.condition == "definition"].correct.mean())
        k1 = float(g[g.k == 1]["mean"].iloc[0])
        runs.append({"arch": arch, "seed": seed, "definition_individual": dfn, "k1_individual": k1,
                     "identifiable": wavg(g.identifiable), "not_identifiable": wavg(~g.identifiable),
                     "k1_over_half_definition": k1 / (0.5 * dfn), "takeoff": dfn > 0.7})
    rdf = pd.DataFrame(runs)
    rdf.to_csv(tdir / "exp2_runs.csv", index=False)
    p_take = rdf.takeoff.mean()
    agree = int(sum(rdf[rdf.seed == sd].takeoff.nunique() == 1 for sd in seeds))
    p_agree = p_take ** 4 + (1 - p_take) ** 4
    md = ["Per-run individual accuracy (%) (exploratory, post hoc). take-off = definition individual "
          "accuracy > 70%. k=1 / (def/2): ratio of k=1 accuracy to half the definition accuracy "
          "(choosing one of the two consistent operations at random).\n",
          "| arch | seed | definition | k=1 | identifiable | not identifiable | k=1 / (def/2) | take-off |",
          "|---|---|---|---|---|---|---|---|"]
    for r in runs:
        md.append(f"| {r['arch']} | {r['seed']} | {r['definition_individual']*100:.1f} | "
                  f"{r['k1_individual']*100:.1f} | {r['identifiable']*100:.1f} | "
                  f"{r['not_identifiable']*100:.1f} | {r['k1_over_half_definition']:.2f} | "
                  f"{'yes' if r['takeoff'] else 'no'} |")
    md += ["", f"Take-off rate {rdf.takeoff.sum()}/{len(rdf)}. Seeds in which all 4 architectures agree: "
           f"{agree}/{len(seeds)} (expected under independence at this rate: "
           f"{p_agree * len(seeds):.1f}; per-seed probability {p_agree:.2f})."]
    (tdir / "exp2_runs.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    res["concordance"] = {"takeoff_rate": float(p_take), "seeds_all_agree": agree,
                          "expected_under_independence": float(p_agree * len(seeds))}
    json.dump(res, open(out / "stats_exp2.json", "w"), indent=2)


if __name__ == "__main__":
    main()
