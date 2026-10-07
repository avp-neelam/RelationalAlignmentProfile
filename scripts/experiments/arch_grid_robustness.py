"""Architecture-ranking robustness under a localized hyperparameter grid
search (Weakness 4: "Architecture Bank Capacity and Hyperparameter
Rigidity").

The main experiments train all 6 architectures with a single, fixed,
untuned ArchConfig each (utils.experiments.runner.DEFAULT_ARCH_CONFIG).
A reasonable reviewer objection is that this disadvantages the
higher-capacity / more hyperparameter-sensitive architectures -- GPS-style
graph_transformer and FAGCN (high_pass) in particular -- relative to
simpler ones (GCN, MLP), and that architecture *selection* results (which
architecture RAP predicts is best for a problem) could just be an artifact
of that fixed, arbitrary configuration.

This script tests that directly: on a representative subset of datasets,
it (1) runs a small local grid over learning rate and hidden width for
JUST high_pass and graph_transformer (the two flagged architectures),
picks each architecture's best per-dataset configuration by validation
performance, then (2) recomputes the full 6-way architecture ranking with
those two architectures' tuned configs substituted in for their fixed
defaults (GCN/MLP/APPNP/SAGE-role untouched), and (3) reports, per
dataset, the Spearman rank correlation between the "fixed" and "tuned"
6-way rankings and whether the argmax (best architecture) changes.

High rank correlation and few argmax flips would support the paper's claim
that its architecture-selection results are not an artifact of an
untuned/unfair hyperparameter grid; a low correlation or frequent flips
would indicate the opposite and should be reported as a limitation.

This does not change any cached E2/E3/E4 result: it uses the SAME
architecture_perf.json cache (keyed by problem fingerprint + full
ArchConfig, see _perf_cache_key), so the untouched architectures' fixed-
config performance is read straight from cache if already computed there,
and only the grid cells for high_pass/graph_transformer need fresh
training.

Usage
-----
    python scripts/experiments/arch_grid_robustness.py \\
        --datasets Cora CiteSeer PubMed Roman-empire Tolokers \\
        --out results/arch_grid_robustness

    # Wider/narrower grid:
    python scripts/experiments/arch_grid_robustness.py \\
        --lrs 0.003 0.01 0.03 --hiddens 32 64 128 \\
        --datasets Cora CiteSeer PubMed --out results/arch_grid_robustness
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.experiments.runner import (  # noqa: E402
    ARCHITECTURE_NAMES,
    REAL_DATASET_SPECS,
    DEFAULT_ARCH_CONFIG,
    ExperimentConfig,
    PerformanceCache,
    _perf_cache_key,
    active_architectures,
    gg,
    load_real_problem,
    train_one_seed,
)

TUNED_ARCHS = ("high_pass", "graph_transformer")


def grid_search_one_arch(problem, role_x, arch_name: str, base_cfg,
                         lrs: List[float], hiddens: List[int],
                         seeds, cache: PerformanceCache) -> Dict[str, Any]:
    """Best (lr, hidden) by mean performance over `seeds`, plus the full grid."""
    grid: Dict[str, float] = {}
    best_key, best_val = None, -np.inf
    for lr in lrs:
        for hidden in hiddens:
            cfg = dataclasses.replace(base_cfg, lr=lr, hidden=hidden)
            vals = []
            for seed in seeds:
                key = _perf_cache_key(problem, f"gridsearch:{arch_name}", seed, cfg)
                v = cache.get(key)
                if v is None:
                    v = train_one_seed(problem, role_x, arch_name, cfg, seed)
                    cache.put(key, v)
                vals.append(v)
            mean_v = float(np.mean(vals))
            grid[f"lr={lr},hidden={hidden}"] = mean_v
            if mean_v > best_val:
                best_val, best_key = mean_v, (lr, hidden)
    return {"grid": grid, "best": {"lr": best_key[0], "hidden": best_key[1], "mean": best_val}}


def run_one_dataset(name: str, spec_by_name, cfg: ExperimentConfig, cache: PerformanceCache,
                    seeds, lrs: List[float], hiddens: List[int]) -> Dict[str, Any]:
    source, _, metric = spec_by_name[name]
    problem = load_real_problem(source, name, metric, cfg, split_id=cfg.real_split_seed)
    role_x = gg.role_signature(problem.G)

    archs = list(active_architectures(cfg))

    # Fixed-default performance for every architecture (reuses E2/E3/E4 cache).
    fixed_mean = {}
    for arch in archs:
        ac = DEFAULT_ARCH_CONFIG[arch]
        vals = []
        for seed in seeds:
            key = _perf_cache_key(problem, arch, seed, ac)
            v = cache.get(key)
            if v is None:
                v = train_one_seed(problem, role_x, arch, ac, seed)
                cache.put(key, v)
            vals.append(v)
        fixed_mean[arch] = float(np.mean(vals))

    # Tuned performance: grid-searched for high_pass/graph_transformer,
    # identical fixed-default value for everything else.
    tuning: Dict[str, Any] = {}
    tuned_mean = dict(fixed_mean)
    for arch in TUNED_ARCHS:
        if arch not in archs:
            continue
        result = grid_search_one_arch(
            problem, role_x, arch, DEFAULT_ARCH_CONFIG[arch], lrs, hiddens, seeds, cache
        )
        tuning[arch] = result
        tuned_mean[arch] = result["best"]["mean"]

    fixed_vec = np.array([fixed_mean[a] for a in archs])
    tuned_vec = np.array([tuned_mean[a] for a in archs])
    rho = spearmanr(fixed_vec, tuned_vec).statistic
    if not np.isfinite(rho):
        rho = 1.0
    argmax_fixed = archs[int(np.argmax(fixed_vec))]
    argmax_tuned = archs[int(np.argmax(tuned_vec))]

    return {
        "architectures": archs,
        "fixed_mean": fixed_mean,
        "tuned_mean": tuned_mean,
        "tuning_detail": tuning,
        "rank_spearman_fixed_vs_tuned": float(rho),
        "argmax_fixed": argmax_fixed,
        "argmax_tuned": argmax_tuned,
        "argmax_changed": argmax_fixed != argmax_tuned,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+",
                    default=["Cora", "CiteSeer", "PubMed", "Roman-empire", "Tolokers"],
                    help="Representative subset spanning homophilous (Cora/CiteSeer/PubMed) "
                         "and heterophilous (Roman-empire, Tolokers) regimes.")
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--cache-dir", default="./cache")
    ap.add_argument("--out", default="results/arch_grid_robustness")
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--lrs", type=float, nargs="+", default=[0.003, 0.01, 0.03])
    ap.add_argument("--hiddens", type=int, nargs="+", default=[32, 64, 128])
    args = ap.parse_args()

    cfg = ExperimentConfig()
    cfg.data_root = args.data_root
    cfg.cache_dir = args.cache_dir
    seeds = args.seeds or list(cfg.model_seeds)

    spec_by_name = {name: (source, name, metric) for source, name, metric in REAL_DATASET_SPECS}
    cache = PerformanceCache(Path(args.cache_dir) / "architecture_perf.json")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Any] = {}
    for name in args.datasets:
        if name not in spec_by_name:
            raise ValueError(f"{name!r} is not in REAL_DATASET_SPECS")
        print(f"[grid-robustness] {name}")
        entry = run_one_dataset(name, spec_by_name, cfg, cache, seeds, args.lrs, args.hiddens)
        results[name] = entry
        print(f"  rank Spearman(fixed, tuned) = {entry['rank_spearman_fixed_vs_tuned']:.3f} | "
              f"argmax fixed={entry['argmax_fixed']} tuned={entry['argmax_tuned']} "
              f"({'CHANGED' if entry['argmax_changed'] else 'stable'})")
        (out_dir / "arch_grid_robustness_results.json").write_text(json.dumps(results, indent=2))

    n_changed = sum(1 for e in results.values() if e["argmax_changed"])
    mean_rho = float(np.mean([e["rank_spearman_fixed_vs_tuned"] for e in results.values()]))
    print(f"\nDone. Mean rank Spearman(fixed, tuned) = {mean_rho:.3f}; "
          f"argmax changed on {n_changed}/{len(results)} datasets.")
    print(f"Full results written to {out_dir / 'arch_grid_robustness_results.json'}")


if __name__ == "__main__":
    main()
