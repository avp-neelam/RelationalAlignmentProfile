"""Ad-hoc diagnostic: how well do the real node-classification datasets fit
the synthetic-trained mechanism dictionaries -- with NO architecture
training and NO GPU.

This is the cheapest possible test of the concern that real graphs are not
as cleanly single-mechanism as the synthetic G_train/G_test families. For
each real dataset it projects the uncompressed alignment fields onto the
frozen native/task dictionaries (fit only on synthetic G_train, exactly as
E2/E3 do -- no real data is used to fit anything here) and reports:

  * the NNLS residual for alpha^(0) (native) and beta^(tau) (task) --
    a large residual means "this graph's structure-feature / structure-label
    alignment doesn't look like a convex combination of the five/four
    planted mechanisms";
  * the abstain() rule from utils.grammar.core, combining that residual with
    a resampling estimate of coordinate spread;
  * gamma(D) = [alpha, beta] itself, for inspection;
  * the conventional-statistics vector (adjusted homophily, label
    informativeness, degree/spectral summaries) used throughout E1-E4, so
    you can eyeball collinearity with gamma;
  * a Spearman correlation table between gamma coordinates and conventional
    summaries across the real datasets run.

Not part of the package API -- a one-off check that reuses
utils.experiments.runner's grammar/geometry pipeline without ever touching
the architecture bank (no PerformanceCache, no training).

Dependencies: needs torch + torch-geometric (and `ogb` for ogbn-arxiv)
purely because that's how utils.experiments.runner.load_real_problem
fetches/wraps the datasets -- utils/experiments/runner.py imports torch
unconditionally at module level (it also defines the architecture bank), so
this is required regardless. No GPU and no architecture training happen in
this script's own code path.

This repo has no pyproject.toml/setup.py -- `utils` is only importable with
the repo root on sys.path (the README's own examples rely on running
`python -m utils.experiments.runner ...` from the repo root for the same
reason). This script adds the repo root to sys.path itself, so it can be
run from anywhere.

Runs on limited/local hardware: results are written to disk after EVERY
dataset (not just at the end), and each dataset is wrapped in a wall-clock
timeout (default 8 minutes, --timeout-per-dataset to change) so one slow
graph -- e.g. Roman-empire, whose near-path/high-diameter topology is a
much harder regime for the role-signature/landmark-BFS primitives than a
citation graph of similar node count, even though it's not "big" by n --
can't stall the whole run. A timed-out dataset is recorded as such and the
script moves on. (Unix/macOS only: uses signal.alarm.)

Usage
-----
    python scripts/diagnostics/real_geometry_check.py --out results/real_geometry_check

    # Skip / restrict to specific datasets. Space- and/or comma-separated,
    # and --skip/--only can each be repeated:
    python scripts/diagnostics/real_geometry_check.py --skip Roman-empire Amazon-ratings
    python scripts/diagnostics/real_geometry_check.py --skip Roman-empire,Amazon-ratings
    python scripts/diagnostics/real_geometry_check.py --skip Roman-empire --skip Amazon-ratings
    python scripts/diagnostics/real_geometry_check.py --only Cora,CiteSeer,PubMed

    python scripts/diagnostics/real_geometry_check.py --include-ogb   # adds ogbn-arxiv (slow: role signature)
    python scripts/diagnostics/real_geometry_check.py --quick         # smaller dictionaries; smoke test only
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.grammar import core as gg  # noqa: E402
from utils.experiments.runner import (  # noqa: E402
    REAL_DATASET_SPECS,
    ExperimentConfig,
    conventional_statistics,
    extract_representations,
    fit_dictionaries,
    load_real_problem,
)

NATIVE_LABELS = list(gg.NATIVE_MECHANISMS)
TASK_LABELS = list(gg.TASK_MECHANISMS)

# Mirrors conventional_statistics' concatenation order in utils/experiments/runner.py.
# If _spectral_summaries' default k changes there, update the spectral_* count below.
CONV_LABELS = (
    ["log1p(n)", "log1p(m)", "density"]
    + [f"degree_{s}" for s in ("mean", "std", "q25", "q50", "q75", "max")]
    + ["feat_edge_dist"]
    + [f"spectral_{i}" for i in range(8)]
    + ["adjusted_homophily", "label_informativeness", "label_entropy"]
)


class DatasetTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise DatasetTimeout()


class dataset_timeout:
    """Context manager: raise DatasetTimeout if the body takes longer than
    `seconds`. Unix/macOS only (SIGALRM); a no-op (no timeout enforced) on
    platforms without it, e.g. native Windows.
    """

    def __init__(self, seconds: int):
        self.seconds = seconds
        self._supported = hasattr(signal, "SIGALRM")

    def __enter__(self):
        if self._supported and self.seconds > 0:
            signal.signal(signal.SIGALRM, _alarm_handler)
            signal.alarm(self.seconds)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._supported and self.seconds > 0:
            signal.alarm(0)
        return False


def _parse_name_list(values: list[str]) -> list[str]:
    """Flatten --skip/--only values into a clean name list. Accepts either
    style (or a mix): repeated flags (--skip A --skip B), space-separated
    (--skip A B, via argparse nargs="+"), and/or comma-separated within one
    token (--skip A,B). Whitespace around names is stripped.
    """
    names: list[str] = []
    for v in values:
        names.extend(part.strip() for part in v.split(",") if part.strip())
    return names


def resampled_spread(problem, native_templates, task_templates, cfg, base_seed, n_resamples):
    """Lightweight proxy for Sec 3.6's estimation uncertainty: repeat profile
    extraction over `n_resamples` independent seeds (fresh pair subsamples,
    landmark choices, and label folds) and return the stacked alpha/beta
    samples. Not identical to Algorithm 1's fixed-fraction subsample
    procedure -- just a fast, honestly-described stand-in for this
    diagnostic.

    Note: each repeat recomputes role_signature/propagation_embeddings from
    scratch (extract_representations doesn't expose a way to reuse them),
    so this loop is roughly (1 + n_resamples)x the cost of a single profile
    extraction. On a slow/large graph, pass --resamples 0 to skip the spread
    estimate entirely and just get the point-estimate residual/abstention.
    """
    if n_resamples <= 0:
        return None, None
    y_train = problem.y_dict(problem.train_mask)
    alphas, betas = [], []
    for b in range(n_resamples):
        rep = extract_representations(
            problem, native_templates, task_templates, cfg,
            seed=base_seed + 1000 * (b + 1), y_observed=y_train,
        )
        alphas.append(rep.alpha)
        betas.append(rep.beta)
    return np.stack(alphas), np.stack(betas)


def process_dataset(source, name, metric, native_templates, task_templates, cfg, seed, n_resamples):
    problem = load_real_problem(source, name, metric, cfg, split_id=seed)
    y_train = problem.y_dict(problem.train_mask)

    rep = extract_representations(
        problem, native_templates, task_templates, cfg,
        seed=seed, y_observed=y_train,
    )
    _, native_residual = gg.project_nnls(rep.raw_native, native_templates)
    task_residual = None
    if rep.task_support:
        _, task_residual = gg.project_nnls(rep.raw_task, task_templates)

    alpha_samples, beta_samples = resampled_spread(
        problem, native_templates, task_templates, cfg, seed, n_resamples
    )
    if alpha_samples is not None:
        native_abstain = gg.abstain(alpha_samples, native_residual)
        task_abstain = (
            gg.abstain(beta_samples, task_residual) if (rep.task_support and task_residual is not None) else True
        )
    else:
        # No resampling requested -> fall back to residual-only abstention
        # (abstain() also checks coordinate spread; skip that half of the rule).
        native_abstain = native_residual > 0.1
        task_abstain = (task_residual > 0.1) if (rep.task_support and task_residual is not None) else True

    conv = conventional_statistics(problem, y_train)

    print(f"  n={problem.G.number_of_nodes()} classes={problem.n_classes} "
          f"train_labels={len(y_train)} task_support={rep.task_support}")
    print(f"  native residual={native_residual:.4f} abstain={native_abstain}")
    if rep.task_support:
        print(f"  task residual={task_residual:.4f} abstain={task_abstain}")
    print(f"  alpha^(0)={np.round(rep.alpha, 3).tolist()}")
    if rep.task_support:
        print(f"  beta^(tau)={np.round(rep.beta, 3).tolist()}")

    record = {
        "source": source, "metric": metric,
        "n_nodes": int(problem.G.number_of_nodes()),
        "n_edges": int(problem.G.number_of_edges()),
        "n_classes": int(problem.n_classes),
        "n_train_labels": int(len(y_train)),
        "task_support": bool(rep.task_support),
        "alpha": rep.alpha.tolist(),
        "beta": rep.beta.tolist(),
        "native_residual": float(native_residual),
        "task_residual": float(task_residual) if task_residual is not None else None,
        "native_abstain": bool(native_abstain),
        "task_abstain": bool(task_abstain),
        "alpha_resample_std": alpha_samples.std(axis=0).tolist() if alpha_samples is not None else None,
        "beta_resample_std": (beta_samples.std(axis=0).tolist() if (alpha_samples is not None and rep.task_support) else None),
        "conventional_statistics": conv.tolist(),
    }
    return record, rep.gamma, conv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="./results/real_geometry_check")
    ap.add_argument("--data-root", type=str, default="./data")
    ap.add_argument("--include-ogb", action="store_true", help="Also run ogbn-arxiv (slow role-signature step)")
    ap.add_argument("--quick", action="store_true", help="Smaller dictionaries; faster, noisier -- smoke test only")
    ap.add_argument("--resamples", type=int, default=3, help="Pair-resampling repeats per dataset for the spread estimate (0 to skip)")
    ap.add_argument("--seed", type=int, default=30)
    ap.add_argument("--timeout-per-dataset", type=int, default=480, help="Seconds before giving up on one dataset and moving on (0 disables)")
    ap.add_argument(
        "--skip", nargs="+", action="extend", default=[], metavar="NAME",
        help="Dataset name(s) to skip. Repeatable, space-separated, and/or comma-separated "
             "within a token, e.g. --skip Roman-empire Amazon-ratings  or  --skip Roman-empire,Amazon-ratings",
    )
    ap.add_argument(
        "--only", nargs="+", action="extend", default=[], metavar="NAME",
        help="Dataset name(s) to run exclusively (overrides --skip). Same syntax as --skip.",
    )
    args = ap.parse_args()

    cfg = ExperimentConfig.quick() if args.quick else ExperimentConfig()
    cfg.data_root = args.data_root
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results_path = out / "real_geometry_check.json"

    print("Fitting native + task dictionaries on synthetic G_train only (no real data used here) ...")
    t0 = time.time()
    native_templates, task_templates = fit_dictionaries(cfg)
    print(f"  done in {time.time()-t0:.1f}s  (native {native_templates.shape}, task {task_templates.shape})")

    specs = list(REAL_DATASET_SPECS)
    if not args.include_ogb:
        specs = [s for s in specs if s[0] != "ogb"]

    only = _parse_name_list(args.only)
    skip = _parse_name_list(args.skip)
    valid_names = {s[1] for s in REAL_DATASET_SPECS}
    for label, names in (("only", only), ("skip", skip)):
        for nm in names:
            if nm not in valid_names:
                print(f"WARNING: --{label} name '{nm}' doesn't match any REAL_DATASET_SPECS entry "
                      f"({sorted(valid_names)}) -- it will have no effect. Names are case-sensitive.")
    if only:
        only_set = set(only)
        specs = [s for s in specs if s[1] in only_set]
    elif skip:
        skip_set = set(skip)
        specs = [s for s in specs if s[1] not in skip_set]

    # Resume support: pick up any per-dataset results already on disk from a
    # previous (possibly interrupted) run at this --out path.
    per_dataset = {}
    if results_path.exists():
        try:
            per_dataset = json.loads(results_path.read_text())
            print(f"Resuming: found {len(per_dataset)} existing result(s) at {results_path}")
        except Exception:
            per_dataset = {}

    rows = []
    for i, (source, name, metric) in enumerate(specs):
        if name in per_dataset and "error" not in per_dataset[name]:
            print(f"\n[{name}] already done, skipping (delete its entry in {results_path.name} to redo)")
            rows.append((name, np.array(per_dataset[name]["alpha"] + per_dataset[name]["beta"]),
                         np.array(per_dataset[name]["conventional_statistics"])))
            continue

        print(f"\n[{name}]")
        t0 = time.time()
        try:
            with dataset_timeout(args.timeout_per_dataset):
                record, gamma, conv = process_dataset(
                    source, name, metric, native_templates, task_templates, cfg,
                    args.seed + i, args.resamples,
                )
            print(f"  ({time.time()-t0:.1f}s)")
            per_dataset[name] = record
            rows.append((name, gamma, conv))
        except DatasetTimeout:
            elapsed = time.time() - t0
            print(f"  TIMED OUT after {elapsed:.0f}s (limit {args.timeout_per_dataset}s) -- skipping, moving on")
            per_dataset[name] = {"source": source, "metric": metric, "error": "timeout",
                                  "timeout_s": args.timeout_per_dataset, "elapsed_s": elapsed}
        except Exception as e:  # keep going -- one bad dataset shouldn't lose the rest
            print(f"  ERROR: {type(e).__name__}: {e} -- skipping, moving on")
            per_dataset[name] = {"source": source, "metric": metric, "error": f"{type(e).__name__}: {e}"}

        # Flush after every dataset so partial progress always survives a
        # Ctrl+C, a crash, or another slow/hung dataset later in the list.
        results_path.write_text(json.dumps(per_dataset, indent=2))

    # Rebuild `rows` from EVERYTHING currently on disk at --out, not just the
    # datasets touched by this invocation -- so a narrow --only/--skip run
    # (e.g. --only Tolokers Questions to fill in stragglers) still produces a
    # correlation table over the full accumulated set, not just what ran here.
    rows = []
    for name, rec in per_dataset.items():
        if "error" in rec:
            continue
        gamma = np.array(rec["alpha"] + rec["beta"])
        conv = np.array(rec["conventional_statistics"])
        rows.append((name, gamma, conv))

    # gamma-vs-conventional-statistics correlation, for eyeballing collinearity.
    # n here = number of real datasets successfully run (typically 8-11): read
    # this as a rough signal, not a significance-tested claim -- several of
    # these datasets share a construction methodology (three Planetoid
    # citation graphs; five from the same heterophily-suite paper), so the
    # effective sample size is smaller than the row count suggests.
    if len(rows) >= 4:
        names, gammas, convs = zip(*rows)
        G_ = np.stack(gammas)
        C_ = np.stack(convs)
        gamma_labels = [f"alpha:{m}" for m in NATIVE_LABELS] + [f"beta:{m}" for m in TASK_LABELS]
        conv_labels = CONV_LABELS if C_.shape[1] == len(CONV_LABELS) else [f"conv_{j}" for j in range(C_.shape[1])]
        corr_rows = []
        for gi, gname in enumerate(gamma_labels):
            for ci, cname in enumerate(conv_labels):
                rho, p = spearmanr(G_[:, gi], C_[:, ci])
                corr_rows.append({
                    "gamma_dim": gname, "conventional_dim": cname,
                    "spearman_rho": float(rho) if rho is not None and not np.isnan(rho) else None,
                    "p": float(p) if p is not None and not np.isnan(p) else None,
                })
        corr_rows_sorted = sorted(
            (r for r in corr_rows if r["spearman_rho"] is not None),
            key=lambda r: -abs(r["spearman_rho"]),
        )
        (out / "gamma_vs_conventional_correlations.json").write_text(json.dumps(corr_rows, indent=2))
        print(f"\nTop |Spearman rho| between gamma coordinates and conventional stats "
              f"(n={len(rows)} datasets -- suggestive only, not significance-tested):")
        for r in corr_rows_sorted[:10]:
            print(f"  {r['gamma_dim']:28s} vs {r['conventional_dim']:20s} "
                  f"rho={r['spearman_rho']:+.2f} (p={r['p']:.2f})")
    else:
        print(f"\nOnly {len(rows)} dataset(s) completed -- skipping the correlation table (needs >= 4).")

    n_errors = sum(1 for v in per_dataset.values() if "error" in v)
    print(f"\nResults written to {results_path}")
    print(f"Completed: {len(rows)}  |  Timed out / errored: {n_errors}  |  Total attempted: {len(specs)}")
    if n_errors:
        failed = [name for name, v in per_dataset.items() if "error" in v]
        print(f"Failed/timed-out datasets: {failed}")
        print("Re-run with the same --out to retry just those (completed ones are skipped automatically).")


if __name__ == "__main__":
    main()
