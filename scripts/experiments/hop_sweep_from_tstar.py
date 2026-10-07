"""Continuous propagation time (t*) vs. discrete GNN depth ("Suggested
Additional Experiment 3": Continuous Propagation vs. Discrete Hop Sweeps).

The paper's alignment curves (Sec.~3.3 / Appendix "From diffusion time to
propagation depth", Table~\\ref{tab:depth}) recommend reading a propagation
depth directly off a problem's diffusion peak t* (align.t_star in
real_rap_profile.json), rather than always training every architecture at a
fixed default depth. This script tests that recommendation directly: for a
chosen subset of real datasets whose alignment curve peaks late (by default
CS, PubMed, Photo -- t* = 60, 40, 90 respectively, all resolved well past
the fixed 2-layer/10-hop defaults), it trains

  * GCN with the default 2 stacked layers, and again with a layer count read
    off t* (capped at --gcn-max-layers, since Table~\\ref{tab:depth} itself
    recommends decoupled propagation rather than deep stacking once
    t gtrsim 2*mean-shortest-path -- stacking a plain GCN that deep also
    over-smooths regardless of what t* says), and
  * APPNP with the default K=10 hops, and again with K read off t*
    (uncapped: APPNP decouples propagation depth from representational
    depth, which is exactly the "decoupled propagation" the table
    recommends at high t, so it does not carry the same over-smoothing risk
    as stacking GCNConv layers),

on the SAME problem instance/split/seeds, and reports both variants'
test performance so the two can be compared head-to-head. A win for the
t*-derived depth on the delayed-peak datasets is direct evidence that the
alignment curve's peak scale has actionable value for architecture
*configuration*, beyond just architecture *family* selection (which is all
the main experiments (E1-E4) test).

This reuses utils.experiments.runner's existing Problem loading, model
training (train_one_seed), and performance cache verbatim -- it does not
duplicate any training logic, and the "*_default" variants reuse whatever
architecture_perf.json cache entries already exist from E2/E3/E4 runs (same
problem fingerprint, arch, seed, and ArchConfig hash), so only the two new
t*-derived variants actually need fresh training.

Requires: the full torch / torch_geometric stack utils.experiments.runner
needs generally (see that module's own docstring), plus
results/real_rap_profile/real_rap_profile.json already present (ships in
the repo; regenerate with scripts/diagnostics/real_rap_profile.py if
missing) for each dataset's align.t_star.

Usage
-----
    python scripts/experiments/hop_sweep_from_tstar.py \\
        --datasets CS PubMed Photo \\
        --out results/hop_sweep

    # Wider cap on stacked-GCN depth, custom dataset subset:
    python scripts/experiments/hop_sweep_from_tstar.py \\
        --datasets CS PubMed Photo Computers \\
        --gcn-max-layers 6 --out results/hop_sweep
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.experiments.runner import (  # noqa: E402
    REAL_DATASET_SPECS,
    ArchConfig,
    DEFAULT_ARCH_CONFIG,
    ExperimentConfig,
    PerformanceCache,
    _perf_cache_key,
    gg,
    load_real_problem,
    train_one_seed,
)


def resolved_hops_from_tstar(t_star: float) -> int:
    """Appendix Table~\\ref{tab:depth}: the resolved neighborhood under the
    heat-kernel distance is empirically about half the nominal walk length
    t (a Poisson(t) mean), because the distance compares heat *profiles*
    rather than raw walks. e.g. t*=60 -> ~30 resolved hops, t*=8 -> ~4."""
    return max(1, int(round(t_star / 2.0)))


def with_override(base: ArchConfig, **overrides: Any) -> ArchConfig:
    import dataclasses
    return dataclasses.replace(base, **overrides)


def run_one_dataset(name: str, spec_by_name: Dict[str, tuple], t_star_table: dict,
                    cfg: ExperimentConfig, cache: PerformanceCache,
                    seeds, gcn_max_layers: int) -> Dict[str, Any]:
    if name not in spec_by_name:
        raise ValueError(f"{name!r} is not in REAL_DATASET_SPECS")
    if name not in t_star_table:
        raise ValueError(
            f"{name!r} missing from the t*-star table; regenerate "
            "results/real_rap_profile/real_rap_profile.json first"
        )
    source, _, metric = spec_by_name[name]
    t_star = float(t_star_table[name]["align"]["t_star"])
    resolved = resolved_hops_from_tstar(t_star)
    gcn_layers = max(2, min(resolved, gcn_max_layers))
    appnp_hops = max(2, resolved)

    problem = load_real_problem(source, name, metric, cfg, split_id=cfg.real_split_seed)
    role_x = gg.role_signature(problem.G)

    variants = {
        "gcn_default": ("local_lowpass", DEFAULT_ARCH_CONFIG["local_lowpass"]),
        "gcn_tstar": ("local_lowpass",
                     with_override(DEFAULT_ARCH_CONFIG["local_lowpass"], layers=gcn_layers)),
        "appnp_default": ("multihop", DEFAULT_ARCH_CONFIG["multihop"]),
        "appnp_tstar": ("multihop",
                       with_override(DEFAULT_ARCH_CONFIG["multihop"], hops=appnp_hops)),
    }

    entry: Dict[str, Any] = {
        "t_star": t_star,
        "resolved_hops": resolved,
        "gcn_layers_used": gcn_layers,
        "appnp_hops_used": appnp_hops,
        "per_seed": {},
        "mean": {},
        "std": {},
    }
    for label, (arch_name, arch_cfg) in variants.items():
        vals = []
        for seed in seeds:
            key = _perf_cache_key(problem, f"hopsweep:{label}", seed, arch_cfg)
            v = cache.get(key)
            if v is None:
                v = train_one_seed(problem, role_x, arch_name, arch_cfg, seed)
                cache.put(key, v)
            vals.append(v)
        entry["per_seed"][label] = vals
        entry["mean"][label] = float(np.mean(vals))
        entry["std"][label] = float(np.std(vals))

    entry["gcn_delta"] = entry["mean"]["gcn_tstar"] - entry["mean"]["gcn_default"]
    entry["appnp_delta"] = entry["mean"]["appnp_tstar"] - entry["mean"]["appnp_default"]
    return entry


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=["CS", "PubMed", "Photo"],
                    help="Real datasets to test (must have a delayed/late alignment peak; "
                         "default = the three flagged in the appendix, t* in {40, 60, 90}).")
    ap.add_argument("--t-star-json", default="results/real_rap_profile/real_rap_profile.json")
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--cache-dir", default="./cache")
    ap.add_argument("--out", default="results/hop_sweep")
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="Defaults to ExperimentConfig.model_seeds, i.e. the same 5 seeds "
                         "used everywhere else in the paper.")
    ap.add_argument("--gcn-max-layers", type=int, default=8,
                    help="Cap on stacked-GCN depth derived from t*; APPNP hops are not "
                         "capped (decoupled propagation does not over-smooth the same way; "
                         "see Appendix, 'From diffusion time to propagation depth').")
    args = ap.parse_args()

    cfg = ExperimentConfig()
    cfg.data_root = args.data_root
    cfg.cache_dir = args.cache_dir
    seeds = args.seeds or list(cfg.model_seeds)

    t_star_table = json.loads(Path(args.t_star_json).read_text())
    spec_by_name = {name: (source, name, metric) for source, name, metric in REAL_DATASET_SPECS}

    cache = PerformanceCache(Path(args.cache_dir) / "architecture_perf.json")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Any] = {}
    for name in args.datasets:
        print(f"[hop-sweep] {name}")
        entry = run_one_dataset(name, spec_by_name, t_star_table, cfg, cache,
                                seeds, args.gcn_max_layers)
        results[name] = entry
        print(
            f"  t*={entry['t_star']:.1f} resolved_hops={entry['resolved_hops']} | "
            f"GCN default={entry['mean']['gcn_default']:.4f} "
            f"t*(L={entry['gcn_layers_used']})={entry['mean']['gcn_tstar']:.4f} "
            f"(delta={entry['gcn_delta']:+.4f}) | "
            f"APPNP default={entry['mean']['appnp_default']:.4f} "
            f"t*(K={entry['appnp_hops_used']})={entry['mean']['appnp_tstar']:.4f} "
            f"(delta={entry['appnp_delta']:+.4f})"
        )
        (out_dir / "hop_sweep_results.json").write_text(json.dumps(results, indent=2))

    n_gcn_win = sum(1 for e in results.values() if e["gcn_delta"] > 0)
    n_appnp_win = sum(1 for e in results.values() if e["appnp_delta"] > 0)
    print(f"\nDone. t*-derived depth beat the 2-layer/10-hop default on "
          f"{n_gcn_win}/{len(results)} datasets for GCN and "
          f"{n_appnp_win}/{len(results)} datasets for APPNP.")
    print(f"Full results written to {out_dir / 'hop_sweep_results.json'}")


if __name__ == "__main__":
    main()
