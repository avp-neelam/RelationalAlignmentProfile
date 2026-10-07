"""Permutation-based significance test + multiple-comparisons correction for
the curve-statistic vs. conventional-graph-statistic correlation table
(curve_vs_conventional_correlations.json, produced by real_rap_profile.py's
plot_homophily_correlations).

Why this exists
----------------
scipy.stats.spearmanr's reported p-value is an asymptotic (t-distribution)
approximation that is not reliable at n=14 -- exactly the sample size of
our real-dataset suite. This script instead computes an exact-in-spirit
Monte Carlo permutation p-value for every (curve_stat, conventional_stat)
pair already in the correlations file: shuffle the conventional statistic
across datasets (breaking any real association while preserving each
variable's own marginal distribution), recompute Spearman rho, repeat
n_perm times, and report the two-sided fraction of permuted |rho| at least
as extreme as the observed value. It then applies Benjamini-Hochberg FDR
correction across the whole family of tests (168 pairs by default) so a
reader can see which correlations survive multiple-comparisons correction
rather than reading 168 individually-computed p-values at face value.

No torch required -- this operates purely on the JSON already written by
real_rap_profile.py, so it runs anywhere numpy/scipy are available.

Usage
-----
    python scripts/diagnostics/curve_correlation_significance.py \
        --out results/real_rap_profile

    # More permutations (slower, tighter p-value resolution):
    python scripts/diagnostics/curve_correlation_significance.py \
        --out results/real_rap_profile --n-perm 100000

    # Only re-test the pairs already flagged as "strongest" in the paper
    # appendix, skip the full 168-pair sweep:
    python scripts/diagnostics/curve_correlation_significance.py \
        --out results/real_rap_profile --only-flagged
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

# The 9 rows already surfaced as the strongest / most interpretable in the
# paper appendix (Table "app-curve-conventional-corr") -- used by
# --only-flagged to skip the full 168-pair sweep and just re-verify these.
FLAGGED_PAIRS = [
    ("align_role", "label_informativeness"),
    ("align_t*", "adjusted_homophily"),
    ("align_rho*", "adjusted_homophily"),
    ("align_rho*", "label_informativeness"),
    ("align_t*", "label_informativeness"),
    ("align_role", "adjusted_homophily"),
    ("task_role", "density"),
    ("task_local", "log1p(n)"),
    ("task_rho*", "density"),
]

CURVE_STAT_KEYS = ["align_t*", "align_rho*", "align_local", "align_role",
                   "task_t*", "task_rho*", "task_local", "task_role"]


def _curve_stat_value(record: dict, key: str):
    kind, stat = key.split("_", 1)
    curve = record.get(kind)
    if curve is None:
        return None
    if stat == "t*":
        return curve["t_star"]
    if stat == "rho*":
        return curve["rho_star"]
    return curve["scalar"].get(stat)  # "local" / "role"


def permutation_spearman_p(x: np.ndarray, y: np.ndarray, n_perm: int, seed: int) -> tuple[float, float]:
    """Two-sided Monte Carlo permutation p-value for Spearman rho(x, y).
    Returns (observed_rho, p_value).

    Vectorized: Spearman rho is Pearson correlation on ranks, and shuffling
    the raw y values and re-ranking is equivalent to directly shuffling the
    (tie-aware, average-rank) rank vector of y once computed -- a
    permutation of ranks is itself a valid permutation. So ranks are
    computed ONCE (via scipy.stats.rankdata, average-rank tie handling,
    matching spearmanr's own convention) and then n_perm rank-vector
    shuffles are correlated against the fixed rank_x in a single batched
    numpy call, rather than n_perm individual scipy.stats.spearmanr calls
    -- several orders of magnitude faster at n_perm in the tens of
    thousands, which is what makes sweeping the full ~168-pair table
    (rather than only a hand-picked subset) practical here.
    """
    from scipy.stats import rankdata

    n = len(x)
    if n < 4 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return 0.0, 1.0
    rx = rankdata(x)
    ry = rankdata(y)
    rx_c = rx - rx.mean()
    obs_rho = float(np.sum(rx_c * (ry - ry.mean())) / (np.linalg.norm(rx_c) * np.linalg.norm(ry - ry.mean())))

    rng = np.random.default_rng(seed)
    ry_perm = np.tile(ry, (n_perm, 1))
    rng.permuted(ry_perm, axis=1, out=ry_perm)
    ry_perm_c = ry_perm - ry_perm.mean(axis=1, keepdims=True)
    num = ry_perm_c @ rx_c
    den = np.linalg.norm(ry_perm_c, axis=1) * np.linalg.norm(rx_c)
    with np.errstate(invalid="ignore", divide="ignore"):
        r_perm = np.where(den > 0, num / den, 0.0)
    count_ge = int(np.sum(np.abs(r_perm) >= abs(obs_rho) - 1e-9))
    # +1 / +1 smoothing: a permutation test can never report exactly p=0.
    p = (count_ge + 1) / (n_perm + 1)
    return obs_rho, float(p)


def bh_fdr(pvals: list[float]) -> list[float]:
    """Benjamini-Hochberg FDR-adjusted q-values (independent/positive-
    dependence assumption -- standard default; conservative alternative is
    Benjamini-Yekutieli, not needed here since these tests are positively
    correlated by construction, not adversarially dependent)."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q_raw = ranked * n / (np.arange(n) + 1)
    q_monotone = np.minimum.accumulate(q_raw[::-1])[::-1]
    q = np.empty(n)
    q[order] = np.clip(q_monotone, 0, 1)
    return q.tolist()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="./results/real_rap_profile")
    ap.add_argument("--n-perm", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only-flagged", action="store_true",
                     help="Only re-test the 9 pairs already flagged as strongest in the appendix, "
                          "instead of the full curve_stat x conventional_stat sweep. Mutually exclusive "
                          "with --conventional-stats.")
    ap.add_argument("--conventional-stats", type=str, default=None, metavar="NAME[,NAME...]",
                     help="Restrict the conventional-statistic side to this comma-separated list "
                          "(e.g. adjusted_homophily,label_informativeness) instead of all 21. This is "
                          "the methodologically honest middle ground between --only-flagged (9 pairs "
                          "hand-picked post-hoc from the full sweep -- FDR correction over a "
                          "self-selected 'winners' subset is circular) and the full 168-pair fishing "
                          "expedition (almost nothing survives FDR at n=14): restricting to statistics "
                          "that were the a priori target of comparison -- e.g. exactly homophily and "
                          "label informativeness, as originally requested -- and correcting only over "
                          "that pre-specified family is a legitimate, much better-powered test.")
    ap.add_argument("--fdr-q", type=float, default=0.10, help="FDR threshold for the 'survives' column")
    args = ap.parse_args()

    out = Path(args.out)
    real_path = out / "real_rap_profile.json"
    if not real_path.exists():
        print(f"No {real_path} found -- run real_rap_profile.py first.")
        return
    per_dataset = json.loads(real_path.read_text())

    names = [nm for nm, rec in per_dataset.items()
             if "error" not in rec and "conventional_statistics" in rec]
    if len(names) < 4:
        print(f"Only {len(names)} dataset(s) have conventional_statistics -- need at least 4. "
              "Run real_rap_profile.py's backfill first (see its own --help).")
        return
    print(f"Testing correlations across {len(names)} datasets: {names}\n")

    conv_labels = per_dataset[names[0]]["conventional_statistics_labels"]
    conv_matrix = {nm: dict(zip(conv_labels, per_dataset[nm]["conventional_statistics"])) for nm in names}

    if args.only_flagged and args.conventional_stats:
        print("--only-flagged and --conventional-stats are mutually exclusive.")
        return
    if args.only_flagged:
        pairs_to_test = FLAGGED_PAIRS
    elif args.conventional_stats:
        wanted = [s.strip() for s in args.conventional_stats.split(",") if s.strip()]
        unknown = [s for s in wanted if s not in conv_labels]
        if unknown:
            print(f"WARNING: unknown conventional stat(s) {unknown}, available: {conv_labels}")
        pairs_to_test = [(cs, conv) for cs in CURVE_STAT_KEYS for conv in wanted if conv in conv_labels]
    else:
        pairs_to_test = [(cs, conv) for cs in CURVE_STAT_KEYS for conv in conv_labels]

    rows = []
    for curve_key, conv_key in pairs_to_test:
        xs, ys = [], []
        for nm in names:
            cv = _curve_stat_value(per_dataset[nm], curve_key)
            if cv is None:
                continue
            xs.append(cv)
            ys.append(conv_matrix[nm][conv_key])
        if len(xs) < 4:
            continue
        obs_rho, p_perm = permutation_spearman_p(np.array(xs), np.array(ys), args.n_perm, seed=args.seed)
        rows.append({
            "curve_stat": curve_key, "conventional_stat": conv_key,
            "spearman_rho": obs_rho, "n": len(xs), "p_permutation": p_perm,
        })

    rows.sort(key=lambda r: -abs(r["spearman_rho"]))
    qvals = bh_fdr([r["p_permutation"] for r in rows])
    for r, q in zip(rows, qvals):
        r["q_bh_fdr"] = q
        r["survives_fdr"] = q < args.fdr_q

    n_survive = sum(1 for r in rows if r["survives_fdr"])
    print(f"{len(rows)} pair(s) tested with {args.n_perm} permutations each. "
          f"{n_survive} survive Benjamini-Hochberg FDR at q<{args.fdr_q}.\n")
    print(f"{'curve_stat':12s} {'conventional_stat':24s} {'rho':>7s} {'p_perm':>9s} {'q_fdr':>9s}  survives")
    for r in rows[:30]:
        flag = "  <-- survives" if r["survives_fdr"] else ""
        print(f"{r['curve_stat']:12s} {r['conventional_stat']:24s} {r['spearman_rho']:+7.3f} "
              f"{r['p_permutation']:9.4f} {r['q_bh_fdr']:9.4f}{flag}")
    if len(rows) > 30:
        print(f"  ... ({len(rows) - 30} more rows in the output file)")

    out_path = out / "curve_vs_conventional_correlations_significance.json"
    out_path.write_text(json.dumps(rows, indent=2))
    print(f"\nFull table written to {out_path}")


if __name__ == "__main__":
    main()
