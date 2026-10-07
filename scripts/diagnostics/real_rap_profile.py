"""Full Relational Alignment Profile (RAP) for the real node-classification
benchmarks: rho fields -> gamma = [alpha, beta] -> the two dictionary-free
alignment curves rho_align(t) / rho_task(t) -- one pass per dataset, with a
per-dataset figure of the two curves for interpretability.

This supersedes running real_geometry_check.py and a curves-only script
separately: for each real dataset (REAL_DATASET_SPECS in
utils.experiments.runner, now 15 datasets including the 3 Airports graphs)
it does, in one place:

  1. Load the Problem with a fresh 60/20/20 split (see load_real_problem;
     ogbn-arxiv keeps its official OGB split, everything else is re-split).
  2. Extract the raw structure-feature / structure-label alignment FIELDS
     (extract_representations -> rep.raw_native, rep.raw_task) and project
     them onto the frozen synthetic-trained NNLS dictionaries to get
     gamma = [alpha, beta] -- exactly what real_geometry_check.py computes,
     reused here rather than reimplemented.
  3. Compute the dictionary-free curves rho_align(t) and rho_task(t)
     (utils.grammar.curves.compute_alignment_curves) over a diffusion-time
     sweep, from the SAME rank-transformed pair samples -- these do not
     depend on the NNLS dictionaries at all, so a large/near-zero dictionary
     residual in step 2 does not change what the curves say.
  4. Also compute conventional_statistics (adjusted homophily, label
     informativeness, label entropy, degree/spectral summary -- same
     function real_geometry_check.py uses) on the SAME train-label split
     everything else in this record used, so it is directly comparable to
     gamma and the curves rather than a leftover from Planetoid's old
     public split. A dataset that already has gamma/curves from an earlier
     run of this script (before this field existed) gets ONLY this stat
     backfilled on re-run -- see backfill_conventional_statistics -- not a
     full recompute.
  5. (--plot) Save one PNG per dataset overlaying both curves against t;
     one PNG summarizing t*/rho* across all datasets for both curves
     (real_summary.png); two "present this to collaborators" figures, one
     per curve type, with every completed dataset as its own small-
     multiple panel (all_datasets_rho_align.png / all_datasets_rho_task.png,
     see plot_all_curves_grid); a Spearman correlation table between the
     curve statistics and the conventional graph statistics across
     datasets (curve_vs_conventional_correlations.json); and a headline
     adjusted-homophily-vs-peak-rho scatter (figures/homophily_vs_curves.png).

gamma answers "which of 5 native / 4 task synthetic mechanisms does this
graph best resemble, as a single 9-dim coordinate." The curves answer "at
what scale, and how strongly" -- read them together: e.g. a dataset with a
big NNLS residual on alpha (none of the 5 planted mechanisms fit well) can
still have a perfectly readable rho_align(t) shape; the curve doesn't
depend on the synthetic mechanisms being a complete basis, the dictionary
projection does.

Synthetic side (optional, --skip-real is not needed to also get this): the
5 native + 4 task mechanism generators, curve-only (no dictionary fitting
against themselves), averaged over held-out seeds -- the "ground truth
shape" check for whether each mechanism's curve looks like its own story.
Kept from the original alignment_curves_check.py; unaffected by anything
above. See utils/grammar/curves.py's own __main__ demo for a caveat: the
propagation-time curve and the binary-adjacency ("local") scalar can
disagree in sign for opposition/bipartite-flavored mechanisms -- a
documented consequence of continuous heat kernels not preserving discrete
hop-parity (core.py's module docstring, fix #9), not a bug here.

Real-dataset loading and dictionary fitting both need torch + torch-
geometric (and `ogb` for ogbn-arxiv) purely because that's how
utils.experiments.runner works -- see that module's own docstring. Those
imports are deferred until real datasets are actually about to be
processed, so `--skip-real` works in a torch-less environment and the
synthetic-only half of this script never imports torch at all.

Only train-split labels ever reach beta / rho_task (same leakage guard
build_real_corpus uses: `problem.y_dict(problem.train_mask)`).

Usage
-----
    # The experiment as designed: rho fields -> gamma -> curves -> figures,
    # for every real dataset, synthetic ground-truth check included.
    python scripts/diagnostics/real_rap_profile.py --out results/real_rap_profile --plot

    # Real datasets only (skip the synthetic ground-truth check):
    python scripts/diagnostics/real_rap_profile.py --skip-synthetic --plot --out results/real_rap_profile

    # Synthetic only, no torch needed:
    python scripts/diagnostics/real_rap_profile.py --skip-real

    python scripts/diagnostics/real_rap_profile.py --only Cora,CiteSeer,PubMed --plot
    python scripts/diagnostics/real_rap_profile.py --only USA,Brazil,Europe --plot   # Airports only
    python scripts/diagnostics/real_rap_profile.py --skip Roman-empire,Amazon-ratings
    python scripts/diagnostics/real_rap_profile.py --quick --plot                    # smaller/faster smoke test

    # Already have real_rap_profile.json / alignment_curves_synthetic.json under
    # --out from a previous run? Regenerate just the figures (e.g. after a
    # plotting-code change) without re-fitting anything or reloading data:
    python scripts/diagnostics/real_rap_profile.py --replot --out results/real_rap_profile

    # Ran this BEFORE conventional_statistics existed (gamma/curves already
    # complete, no "conventional_statistics" key in the JSON yet)? Re-running
    # with the same --out backfills just the missing stats per dataset --
    # cheap (reloads each Problem, no dictionary fit, no curve recompute) --
    # then --plot / --replot will include the homophily correlation output:
    python scripts/diagnostics/real_rap_profile.py --skip-synthetic --plot --out results/real_rap_profile

    # Already ran the full experiment and want more/wider diffusion times
    # without repeating anything already computed? --extend-t-grid adds new
    # t's to every dataset's existing curves in place: no dictionary fit, no
    # gamma recompute, and no recomputation of rho at t's already in the
    # record -- only a Problem reload (fast) plus the diffusion sweep for the
    # NEW t value(s). See extend_real_dataset.
    python scripts/diagnostics/real_rap_profile.py --extend-t-grid 16,20,24 --plot --out results/real_rap_profile
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.grammar import core as gg  # noqa: E402
from utils.grammar.curves import (  # noqa: E402
    DEFAULT_T_GRID,
    compute_alignment_curves,
)

NATIVE_MECHANISMS = list(gg.NATIVE_MECHANISMS)
TASK_MECHANISMS = list(gg.TASK_MECHANISMS)

# Mirrors conventional_statistics' concatenation order in utils/experiments/runner.py.
# If _spectral_summaries' default k changes there, update the spectral_* count below.
CONV_LABELS = (
    ["log1p(n)", "log1p(m)", "density"]
    + [f"degree_{s}" for s in ("mean", "std", "q25", "q50", "q75", "max")]
    + ["feat_edge_dist"]
    + [f"spectral_{i}" for i in range(8)]
    + ["adjusted_homophily", "label_informativeness", "label_entropy"]
)

# Grouping/coloring for the comprehensive multi-dataset figures.
DATASET_CATEGORY = {
    "Cora": "Citation", "CiteSeer": "Citation", "PubMed": "Citation",
    "Roman-empire": "Heterophilous", "Amazon-ratings": "Heterophilous",
    "Minesweeper": "Heterophilous", "Tolokers": "Heterophilous", "Questions": "Heterophilous",
    # Chameleon-Filtered / Squirrel-filtered are the de-duplicated versions
    # from the SAME Platonov et al. paper/repo as the 5 "Heterophilous"
    # entries above -- grouped/colored with them rather than given their
    # own category.
    "Chameleon-Filtered": "Heterophilous", "Squirrel-filtered": "Heterophilous",
    "Photo": "Amazon/Coauthor", "Computers": "Amazon/Coauthor", "CS": "Amazon/Coauthor",
    "USA": "Airports", "Brazil": "Airports", "Europe": "Airports",
    # Texas/Cornell/Wisconsin + Actor: the Geom-GCN paper's small,
    # heavily-imbalanced heterophily benchmarks (see load_real_problem's
    # webkb/actor branches) -- a distinct family from the Platonov
    # "Heterophilous" suite above despite the shared theme.
    "Texas": "WebKB", "Cornell": "WebKB", "Wisconsin": "WebKB", "Actor": "WebKB",
    # BlogCatalog/Flickr: Wang et al.'s attributed social networks (see
    # load_real_problem's "attributed" branch).
    "BlogCatalog": "Attributed", "Flickr": "Attributed",
    "ogbn-arxiv": "OGB",
}
CATEGORY_COLOR = {
    "Citation": "tab:blue", "Heterophilous": "tab:red",
    "Amazon/Coauthor": "tab:green", "Airports": "tab:purple", "OGB": "tab:gray",
    "WebKB": "tab:brown", "Attributed": "tab:orange",
}


# =============================================================================
# Shared small utilities (duplicated from real_geometry_check.py rather than
# imported -- these scripts are explicitly "ad-hoc diagnostics, not part of
# the package API", so they stay standalone).
# =============================================================================

class DatasetTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise DatasetTimeout()


class dataset_timeout:
    """Context manager: raise DatasetTimeout if the body takes longer than
    `seconds`. Unix/macOS only (SIGALRM); a no-op on platforms without it."""

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
    """Flatten --skip/--only values (repeated flags, space-separated,
    and/or comma-separated within one token) into a clean name list."""
    names: list[str] = []
    for v in values:
        names.extend(part.strip() for part in v.split(",") if part.strip())
    return names


def _parse_t_grid(value: str) -> tuple[float, ...]:
    return tuple(sorted(float(x) for x in value.split(",") if x.strip()))


def _symmetric_ylim(values, pad_frac: float = 0.15, floor: float = 0.02) -> tuple[float, float]:
    """Tight, symmetric-around-zero y-limits sized to the actual data.

    Most real datasets here have |rho| well under 0.2 (this repo's own run
    topped out around 0.37, on Brazil-Airports), so a fixed
    correlation-scale [-1, 1] axis leaves 80%+ of every figure empty and
    makes every curve look flat. `floor` keeps an exactly-flat curve (e.g.
    an Airports dataset's rho_align, identically 0 -- see the Airports
    comment on REAL_DATASET_SPECS) from collapsing the axis to zero width.
    """
    vmax = max((abs(v) for v in values if v is not None), default=0.0)
    half = max(vmax * (1.0 + pad_frac), floor)
    return -half, half


# =============================================================================
# Synthetic side (curve-only ground-truth-shape check; gamma is not computed
# here -- projecting a mechanism's own generator onto a dictionary fit from
# that same generator family is a different, less informative check than
# real_geometry_check.py's held-out-seed native recovery test).
# =============================================================================

def synthetic_native_curves(t_grid, n, n_seeds, n_landmarks, m_pairs, seed0):
    """Mean/std rho_align(t) across held-out seeds, per native mechanism."""
    from utils.grammar.curves import structural_alignment_curve
    results = {}
    for mech in NATIVE_MECHANISMS:
        curves = []
        for s in range(n_seeds):
            data = gg.generate_mechanism_graph(mech, n=n, seed=seed0 + s)
            curve = structural_alignment_curve(
                data["G"], data["X"], t_grid=t_grid, n_landmarks=n_landmarks,
                m_pairs=m_pairs, seed=seed0 + 1000 + s,
            )
            curves.append(curve)
        rho_stack = np.stack([c.rho for c in curves])
        results[mech] = {
            "t_grid": curves[0].t_grid.tolist(),
            "rho_mean": rho_stack.mean(0).tolist(),
            "rho_std": rho_stack.std(0).tolist(),
            "t_star_mean": float(np.mean([c.t_star for c in curves])),
            "rho_star_mean": float(np.mean([c.rho_star for c in curves])),
            "local_scalar_mean": float(np.mean([c.scalar.get("local", 0.0) for c in curves])),
            "role_scalar_mean": float(np.mean([c.scalar.get("role", 0.0) for c in curves])),
            "n_seeds": n_seeds,
        }
        print(f"  [native] {mech:22s} t*~{results[mech]['t_star_mean']:5.2f} "
              f"rho*~{results[mech]['rho_star_mean']:+.3f} "
              f"local~{results[mech]['local_scalar_mean']:+.2f} "
              f"role~{results[mech]['role_scalar_mean']:+.2f}")
    return results


def synthetic_task_curves(t_grid, n, n_seeds, n_landmarks, label_pairs_cap, kappa, seed0):
    """Mean/std rho_task(t) across held-out seeds, per task mechanism."""
    from utils.grammar.curves import task_alignment_curve
    results = {}
    for mech in TASK_MECHANISMS:
        curves, abstentions = [], 0
        for s in range(n_seeds):
            data = gg.generate_task_mechanism_graph(mech, n=n, seed=seed0 + s)
            curve, reason = task_alignment_curve(
                data["G"], data["X"], data["y"], t_grid=t_grid, n_landmarks=n_landmarks,
                label_pairs_cap=label_pairs_cap, kappa=kappa, seed=seed0 + 1000 + s,
            )
            if curve is None:
                abstentions += 1
                continue
            curves.append(curve)
        if not curves:
            results[mech] = {"n_seeds": n_seeds, "n_abstained": abstentions, "abstained": True}
            print(f"  [task]   {mech:22s} ABSTAINED on all {n_seeds} seed(s)")
            continue
        rho_stack = np.stack([c.rho for c in curves])
        results[mech] = {
            "t_grid": curves[0].t_grid.tolist(),
            "rho_mean": rho_stack.mean(0).tolist(),
            "rho_std": rho_stack.std(0).tolist(),
            "t_star_mean": float(np.mean([c.t_star for c in curves])),
            "rho_star_mean": float(np.mean([c.rho_star for c in curves])),
            "local_scalar_mean": float(np.mean([c.scalar.get("local", 0.0) for c in curves])),
            "role_scalar_mean": float(np.mean([c.scalar.get("role", 0.0) for c in curves])),
            "n_seeds": n_seeds, "n_abstained": abstentions, "abstained": False,
        }
        print(f"  [task]   {mech:22s} t*~{results[mech]['t_star_mean']:5.2f} "
              f"rho*~{results[mech]['rho_star_mean']:+.3f} "
              f"local~{results[mech]['local_scalar_mean']:+.2f} "
              f"role~{results[mech]['role_scalar_mean']:+.2f}"
              + (f"  ({abstentions}/{n_seeds} seeds abstained)" if abstentions else ""))
    return results


def plot_synthetic(native_results, task_results, out_dir):
    import matplotlib.pyplot as plt

    for title, results, ylabel, fname in (
        ("Native mechanisms: structural alignment curve", native_results,
         r"$\rho_{\mathrm{align}}(t)$", "synthetic_native_curves.png"),
        ("Task mechanisms: task alignment curve", task_results,
         r"$\rho_{\mathrm{task}}(t)$", "synthetic_task_curves.png"),
    ):
        fig, ax = plt.subplots(figsize=(6, 4.2))
        span_values = []
        for mech, rec in results.items():
            if rec.get("abstained"):
                continue
            t = np.array(rec["t_grid"]); mean = np.array(rec["rho_mean"]); std = np.array(rec["rho_std"])
            line, = ax.plot(t, mean, marker="o", label=mech)
            ax.fill_between(t, mean - std, mean + std, color=line.get_color(), alpha=0.15)
            span_values.extend((mean - std).tolist())
            span_values.extend((mean + std).tolist())
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
        ax.set_xlabel("diffusion time t")
        ax.set_ylabel(ylabel)
        ax.set_ylim(*_symmetric_ylim(span_values))
        ax.set_title(title)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=150)
        plt.close(fig)
        print(f"  wrote {out_dir / fname}")


# =============================================================================
# Real side: rho fields -> gamma -> curves, one dataset at a time
# =============================================================================

def process_real_dataset(source, name, metric, cfg, t_grid, seed,
                          load_real_problem, extract_representations, conventional_statistics,
                          native_templates, task_templates):
    problem = load_real_problem(source, name, metric, cfg, split_id=seed)
    y_train = problem.y_dict(problem.train_mask)  # CRITICAL: no validation/test labels

    # Steps 1-2: raw alignment fields -> NNLS-projected gamma = [alpha, beta].
    # Identical call to real_geometry_check.py's process_dataset.
    rep = extract_representations(
        problem, native_templates, task_templates, cfg, seed=seed, y_observed=y_train,
    )

    # Step 3: dictionary-free rho_align(t) / rho_task(t) curves, from the
    # same rank-transformed pair-sampling machinery but never touching
    # native_templates / task_templates / alpha / beta.
    curves = compute_alignment_curves(
        problem.G, problem.X, y_train, t_grid=t_grid,
        n_landmarks=cfg.n_landmarks, m_pairs=cfg.m_pairs,
        label_pairs_cap=cfg.label_pairs_cap, kappa=cfg.kappa, seed=seed,
    )

    # Conventional graph statistics (adjusted homophily, label
    # informativeness, degree/spectral summary) -- same function
    # real_geometry_check.py uses, computed on the SAME train-label split
    # everything else above used, so it is directly comparable to gamma
    # and the curves rather than a leftover from a different split.
    conv = conventional_statistics(problem, y_train)

    align, task = curves.align, curves.task
    print(f"  n={problem.G.number_of_nodes()} classes={problem.n_classes} train_labels={len(y_train)}")
    print(f"  alpha^(0)={np.round(rep.alpha, 3).tolist()}  ({', '.join(NATIVE_MECHANISMS)})")
    if rep.task_support:
        print(f"  beta^(tau)={np.round(rep.beta, 3).tolist()}  ({', '.join(TASK_MECHANISMS)})")
    else:
        print("  beta^(tau): task_support=False (too few / too little cross-fit-valid labeled pairs)")
    print(f"  rho_align: t*={align.t_star:5.2f} rho*={align.rho_star:+.3f} "
          f"local={align.scalar.get('local', 0):+.2f} role={align.scalar.get('role', 0):+.2f}")
    if task is not None:
        print(f"  rho_task:  t*={task.t_star:5.2f} rho*={task.rho_star:+.3f} "
              f"local={task.scalar.get('local', 0):+.2f} role={task.scalar.get('role', 0):+.2f}")
    else:
        print(f"  rho_task:  ABSTAINED ({curves.task_abstain_reason})")
    print(f"  adjusted_homophily={conv[-3]:+.3f} label_informativeness={conv[-2]:.3f}")

    record = {
        "source": source, "metric": metric,
        "n_nodes": int(problem.G.number_of_nodes()),
        "n_edges": int(problem.G.number_of_edges()),
        "n_classes": int(problem.n_classes),
        "n_train_labels": int(len(y_train)),
        "alpha": rep.alpha.tolist(),
        "beta": rep.beta.tolist(),
        "gamma": rep.gamma.tolist(),
        "task_support": bool(rep.task_support),
        "conventional_statistics": conv.tolist(),
        "conventional_statistics_labels": CONV_LABELS,
        **curves.to_dict(),
    }
    return record


def backfill_conventional_statistics(source, name, metric, cfg, seed,
                                      load_real_problem, conventional_statistics):
    """Attach conventional_statistics to a dataset that already has
    gamma/curves from an earlier run of this script (before this field
    existed). Reloads the Problem -- cheap, seconds, no NNLS projection or
    diffusion sweep -- and computes just the homophily / label-
    informativeness / spectral summary, rather than repeating the
    minutes-scale work that already produced everything else in the
    record. Uses the same split_id (and therefore the same 60/20/20 split)
    the original run used, so the backfilled stats line up with the
    alpha/beta/curves already stored, not a different resample of it.
    """
    problem = load_real_problem(source, name, metric, cfg, split_id=seed)
    y_train = problem.y_dict(problem.train_mask)
    conv = conventional_statistics(problem, y_train)
    return conv.tolist()


def _dataset_seed(name: str, args, real_specs) -> int:
    """Reconstruct the seed process_real_dataset used when this dataset was
    first computed: args.seed + its index into the (ogb-filtered)
    REAL_DATASET_SPECS list, exactly matching run_real's `args.seed + i`
    loop. Correct as long as the ORIGINAL run used the same --seed and did
    not use --only/--skip (either would change `i`) -- true for "run the
    whole experiment" usage, which is how this script is documented to be
    run. A record already carrying its own "curve_seed" (written by
    extend_real_dataset, and by process_real_dataset going forward) uses
    that instead and never needs this reconstruction again.
    """
    for i, (_, nm, _) in enumerate(real_specs):
        if nm == name:
            return args.seed + i
    raise KeyError(f"{name!r} not found in REAL_DATASET_SPECS")


def _merge_curve_dict(old: dict, new_curve, genuinely_new: set) -> dict:
    """Splice a freshly computed AlignmentCurve -- covering ONLY the
    genuinely-new t values -- into an existing curve dict, leaving every
    previously computed (t, rho, p) pair untouched. Scalars (local/role)
    don't depend on t at all, so they are carried over from `old` unchanged
    rather than recomputed.
    """
    combined = {float(t): (r, p) for t, r, p in zip(old["t_grid"], old["rho"], old["p_value"])}
    for t, r, p in zip(new_curve.t_grid, new_curve.rho, new_curve.p_value):
        if float(t) in genuinely_new:
            combined[float(t)] = (float(r), float(p))
    t_sorted = sorted(combined)
    rho_sorted = [combined[t][0] for t in t_sorted]
    p_sorted = [combined[t][1] for t in t_sorted]
    idx = int(np.argmax(np.abs(rho_sorted)))
    return {
        "kind": old["kind"], "t_grid": t_sorted, "rho": rho_sorted, "p_value": p_sorted,
        "n_pairs": old["n_pairs"], "t_star": float(t_sorted[idx]), "rho_star": float(rho_sorted[idx]),
        "scalar": old["scalar"], "scalar_p": old["scalar_p"],
    }


def extend_real_dataset(source, name, metric, cfg, seed, new_ts, old_record, load_real_problem):
    """Add new diffusion times to an already-computed dataset's curves
    without repeating anything the new t's don't need: no dictionary fit,
    no NNLS projection (alpha/beta/gamma are t_grid-independent and are
    left exactly as they are in the returned record), and -- critically --
    no recomputation of rho at t values already in the record. Only the
    Problem is reloaded (it isn't stored in the JSON, so this part can't be
    skipped -- it's a fast, no-diffusion step: dataset load + the
    deterministic 60/20/20 split), then the Chebyshev heat-kernel
    propagation runs for the NEW t's alone, on the exact same pair sample /
    fold assignment as the original run (same seed, same
    m_pairs/label_pairs_cap/kappa), so the merged curve is indistinguishable
    from one computed on the full extended grid from the start.
    """
    from utils.grammar.curves import structural_alignment_curve, task_alignment_curve

    problem = load_real_problem(source, name, metric, cfg, split_id=seed)

    old_align_t = {float(t) for t in old_record["align"]["t_grid"]}
    align_new_ts = sorted(t for t in new_ts if t not in old_align_t)
    align_record = old_record["align"]
    if align_new_ts:
        curve = structural_alignment_curve(
            problem.G, problem.X, t_grid=align_new_ts, n_landmarks=cfg.n_landmarks,
            m_pairs=cfg.m_pairs, seed=seed, include_scalar_geometries=False,
        )
        align_record = _merge_curve_dict(old_record["align"], curve, set(align_new_ts))

    task_record = old_record.get("task")
    if task_record is not None:
        old_task_t = {float(t) for t in task_record["t_grid"]}
        task_new_ts = sorted(t for t in new_ts if t not in old_task_t)
        if task_new_ts:
            y_train = problem.y_dict(problem.train_mask)
            curve, reason = task_alignment_curve(
                problem.G, problem.X, y_train, t_grid=task_new_ts, n_landmarks=cfg.n_landmarks,
                label_pairs_cap=cfg.label_pairs_cap, kappa=cfg.kappa, seed=seed,
                include_scalar_geometries=False,
            )
            if curve is not None:
                task_record = _merge_curve_dict(task_record, curve, set(task_new_ts))
            # else: still abstains at the new t's too (unlikely, since it didn't
            # originally) -- leave task_record exactly as it was rather than lose it.

    task_changed = task_record is not old_record.get("task")
    record = dict(old_record)
    record["align"] = align_record
    record["task"] = task_record
    record["curve_seed"] = seed
    return record, (bool(align_new_ts) or task_changed)


def run_extend(args, cfg, out):
    try:
        from utils.experiments.runner import REAL_DATASET_SPECS, load_real_problem
    except Exception as e:
        print(f"\nCannot extend: could not import utils.experiments.runner ({type(e).__name__}: {e})")
        print("Install torch + torch-geometric (and `ogb` for ogbn-arxiv) to extend the real side.")
        return

    results_path = out / "real_rap_profile.json"
    if not results_path.exists():
        print(f"No {results_path} found under {out} -- nothing to extend. Run the experiment first.")
        return
    per_dataset = json.loads(results_path.read_text())

    real_specs = list(REAL_DATASET_SPECS)
    if not args.include_ogb:
        real_specs = [s for s in real_specs if s[0] != "ogb"]

    only = _parse_name_list(args.only)
    skip = _parse_name_list(args.skip)
    names = [nm for nm in per_dataset if "error" not in per_dataset[nm]]
    if only:
        only_set = set(only)
        names = [nm for nm in names if nm in only_set]
    elif skip:
        skip_set = set(skip)
        names = [nm for nm in names if nm not in skip_set]

    new_ts = list(args.extend_t_grid)
    print(f"Extending t_grid with {new_ts} for {len(names)} dataset(s): {names}\n")
    if cfg.data_root:
        pass  # cfg must match the ORIGINAL run's --quick / non-quick setting -- see main()'s warning

    figures_dir = out / "figures"
    if args.plot:
        figures_dir.mkdir(parents=True, exist_ok=True)

    for name in names:
        old_record = per_dataset[name]
        source, metric = old_record["source"], old_record["metric"]
        try:
            seed = old_record.get("curve_seed")
            if seed is None:
                seed = _dataset_seed(name, args, real_specs)
                print(f"[{name}] no stored curve_seed -- reconstructed seed={seed} from REAL_DATASET_SPECS "
                      f"index + --seed (correct as long as the original run used --seed {args.seed} "
                      f"and no --only/--skip)")
            else:
                print(f"[{name}] using stored curve_seed={seed}")
            print(f"  reloading Problem and computing rho at the new t value(s) only ...")
            t0 = time.time()
            with dataset_timeout(args.timeout_per_dataset):
                record, added = extend_real_dataset(
                    source, name, metric, cfg, seed, new_ts, old_record, load_real_problem,
                )
            if added:
                per_dataset[name] = record
                print(f"  align t_grid now {record['align']['t_grid']}  ({time.time()-t0:.1f}s)")
                if record.get("task") is not None:
                    print(f"  task  t_grid now {record['task']['t_grid']}")
                if args.plot:
                    plot_real_dataset(name, record, figures_dir)
            else:
                print(f"  every requested t already in this dataset's curves -- nothing to add")
        except DatasetTimeout:
            print(f"  TIMED OUT after {args.timeout_per_dataset}s -- leaving this dataset's curves as-is")
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e} -- leaving this dataset's curves as-is")

        results_path.write_text(json.dumps(per_dataset, indent=2))

    if args.plot:
        plot_real_summary(per_dataset, figures_dir)
        plot_all_curves_grid("align", per_dataset, figures_dir)
        plot_all_curves_grid("task", per_dataset, figures_dir)
        plot_homophily_correlations(per_dataset, out)

    print(f"\nExtended results written to {results_path}")


def check_components(source, name, metric, cfg, load_real_problem):
    """Connected-component structure of a real dataset's graph.

    Motivated by the diffusion-time-grid note in the paper appendix: for
    CiteSeer, Amazon-Computers, Cora, Amazon-Photo, and the USA Airports
    graph, the normalized-adjacency-based spectral-gap estimate
    (lambda_2 = 1 - spectral_1 in conventional_statistics) reads as exactly
    the degenerate value consistent with >=2 connected components -- but
    that is a proxy, and could equally be an ARPACK convergence artifact on
    a near-degenerate top eigenvalue rather than genuine disconnection.
    This settles it directly with networkx instead of a spectral proxy.
    Graph connectivity does not depend on the train/val/test split, so no
    seed/split_id is needed here -- there is exactly one graph per dataset.
    """
    import networkx as nx

    problem = load_real_problem(source, name, metric, cfg, split_id=0)
    G = problem.G
    n = G.number_of_nodes()
    comp_iter = nx.weakly_connected_components(G) if nx.is_directed(G) else nx.connected_components(G)
    sizes = sorted((len(c) for c in comp_iter), reverse=True)
    n_isolated = sum(1 for _, d in G.degree() if d == 0)
    return {
        "n_nodes": n,
        "n_components": len(sizes),
        "giant_component_size": sizes[0] if sizes else 0,
        "giant_component_frac": (sizes[0] / n) if n and sizes else 0.0,
        "n_isolated_nodes": n_isolated,
        "component_sizes_top10": sizes[:10],
    }


def run_check_components(args, cfg, out):
    try:
        from utils.experiments.runner import REAL_DATASET_SPECS, load_real_problem
    except Exception as e:
        print(f"\nCannot check components: could not import utils.experiments.runner ({type(e).__name__}: {e})")
        print("Install torch + torch-geometric (and `ogb` for ogbn-arxiv) to check the real side.")
        return

    specs = list(REAL_DATASET_SPECS)
    if not args.include_ogb:
        specs = [s for s in specs if s[0] != "ogb"]
    only = _parse_name_list(args.only)
    skip = _parse_name_list(args.skip)
    if only:
        only_set = set(only)
        specs = [s for s in specs if s[1] in only_set]
    elif skip:
        skip_set = set(skip)
        specs = [s for s in specs if s[1] not in skip_set]

    report = {}
    for source, name, metric in specs:
        try:
            info = check_components(source, name, metric, cfg, load_real_problem)
            report[name] = info
            flag = "  <-- DISCONNECTED" if info["n_components"] > 1 else ""
            print(f"[{name:16s}] n={info['n_nodes']:6d}  components={info['n_components']:4d}  "
                  f"giant={info['giant_component_frac']*100:6.2f}%  "
                  f"isolated_nodes={info['n_isolated_nodes']:4d}{flag}")
        except Exception as e:
            print(f"[{name}] ERROR: {type(e).__name__}: {e}")
            report[name] = {"error": f"{type(e).__name__}: {e}"}

    out_path = out / "component_check.json"
    out_path.write_text(json.dumps(report, indent=2))
    n_disc = sum(1 for v in report.values() if v.get("n_components", 1) > 1)
    print(f"\n{n_disc} of {len(report)} dataset(s) are formally disconnected (>=2 components).")
    print(f"Written to {out_path}")


def multiseed_real_curves(source, name, metric, cfg, t_grid, seeds, load_real_problem):
    """Recompute BOTH curves for `name` across several independent seeds --
    a different 60/20/20 split AND a different pair/landmark sample each
    time, not just new t's on the original sample (unlike
    extend_real_dataset, which deliberately reuses the original sample).
    A genuine robustness check needs independent draws throughout. Skips
    the NNLS/gamma machinery entirely (not needed for this check, and it
    is the expensive part of process_real_dataset), so this is cheaper
    per-seed than a full run but still redoes the whole diffusion sweep
    per seed -- budget accordingly for large graphs.
    """
    from utils.grammar.curves import compute_alignment_curves

    align_rhos, task_rhos = [], []
    for s in seeds:
        problem = load_real_problem(source, name, metric, cfg, split_id=s)
        y_train = problem.y_dict(problem.train_mask)
        curves = compute_alignment_curves(
            problem.G, problem.X, y_train, t_grid=t_grid,
            n_landmarks=cfg.n_landmarks, m_pairs=cfg.m_pairs,
            label_pairs_cap=cfg.label_pairs_cap, kappa=cfg.kappa, seed=s,
        )
        align_rhos.append(curves.align.rho)
        task_str = "ABSTAINED"
        if curves.task is not None:
            task_rhos.append(curves.task.rho)
            task_str = f"rho*={curves.task.rho_star:+.3f} at t*={curves.task.t_star:.2f}"
        print(f"    seed={s}: align rho*={curves.align.rho_star:+.3f} at t*={curves.align.t_star:.2f}  "
              f"task {task_str}")

    align_stack = np.stack(align_rhos)
    result = {
        "t_grid": list(t_grid), "seeds": list(seeds),
        "align_rho_per_seed": align_stack.tolist(),
        "align_rho_mean": align_stack.mean(0).tolist(),
        "align_rho_std": align_stack.std(0).tolist(),
        "task_n_abstained": len(seeds) - len(task_rhos),
    }
    if task_rhos:
        task_stack = np.stack(task_rhos)
        result["task_rho_per_seed"] = task_stack.tolist()
        result["task_rho_mean"] = task_stack.mean(0).tolist()
        result["task_rho_std"] = task_stack.std(0).tolist()
    return result


def plot_multiseed_comparison(name, multiseed_record, original_record, out_dir):
    """Overlay the multi-seed mean +/- 1 std band against the ORIGINAL
    single-seed curve already on file, so a discontinuity like Tolokers'
    or CS's task-curve jump (paper appendix, Sec. "Task alignment and the
    value of structure beyond features") can be read as either "within
    the seed-to-seed spread" (an artifact of that one sample) or "outside
    it" (a shape worth trusting and explaining, not explaining away).
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    t = multiseed_record["t_grid"]
    for ax, kind, color in ((axes[0], "align", "tab:blue"), (axes[1], "task", "tab:red")):
        mean_key, std_key = f"{kind}_rho_mean", f"{kind}_rho_std"
        if mean_key not in multiseed_record:
            ax.text(0.5, 0.5, "no valid seeds\n(all abstained)", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"{name}: {kind}")
            continue
        mean = np.array(multiseed_record[mean_key])
        std = np.array(multiseed_record[std_key])
        ax.plot(t, mean, marker="o", color=color, label=f"mean over {len(multiseed_record['seeds'])} new seeds")
        ax.fill_between(t, mean - std, mean + std, color=color, alpha=0.2, label="+/-1 std")
        orig = original_record.get(kind) if original_record else None
        vals = list(mean - std) + list(mean + std)
        if orig is not None:
            ax.plot(orig["t_grid"], orig["rho"], marker="x", ls="--", color="black", alpha=0.7,
                     label="original single seed (on file)")
            vals += list(orig["rho"])
        ax.axhline(0.0, color="black", lw=0.7, alpha=0.4)
        ax.set_ylim(*_symmetric_ylim(vals))
        ax.set_xlabel("diffusion time t")
        ax.set_ylabel(rf"$\rho_{{\mathrm{{{kind}}}}}(t)$")
        ax.set_title(f"{name}: {kind}")
        ax.legend(fontsize=7.5)
    fig.tight_layout()
    path = out_dir / f"multiseed_{name}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  wrote {path}")


def run_multiseed(args, cfg, t_grid, out):
    try:
        from utils.experiments.runner import REAL_DATASET_SPECS, load_real_problem
    except Exception as e:
        print(f"\nCannot run multiseed check: could not import utils.experiments.runner ({type(e).__name__}: {e})")
        print("Install torch + torch-geometric (and `ogb` for ogbn-arxiv) to check the real side.")
        return

    names = _parse_name_list([args.multiseed])
    specs_by_name = {s[1]: s for s in REAL_DATASET_SPECS}
    for nm in names:
        if nm not in specs_by_name:
            print(f"WARNING: '{nm}' doesn't match any REAL_DATASET_SPECS entry ({sorted(specs_by_name)}).")

    results_path = out / "real_rap_profile.json"
    per_dataset = json.loads(results_path.read_text()) if results_path.exists() else {}

    figures_dir = out / "figures"
    if args.plot:
        figures_dir.mkdir(parents=True, exist_ok=True)

    out_json_path = out / "multiseed_robustness.json"
    multiseed_results = json.loads(out_json_path.read_text()) if out_json_path.exists() else {}

    seeds = [args.seed + 1000 + k for k in range(args.n_extra_seeds)]

    for nm in names:
        if nm not in specs_by_name:
            continue
        source, _, metric = specs_by_name[nm]
        print(f"\n[{nm}] multiseed robustness check, {len(seeds)} new seeds={seeds}")
        t0 = time.time()
        try:
            with dataset_timeout(args.timeout_per_dataset * max(1, len(seeds))):
                rec = multiseed_real_curves(source, nm, metric, cfg, t_grid, seeds, load_real_problem)
            multiseed_results[nm] = rec
            print(f"  done in {time.time()-t0:.1f}s")
            out_json_path.write_text(json.dumps(multiseed_results, indent=2))
            if args.plot:
                plot_multiseed_comparison(nm, rec, per_dataset.get(nm), figures_dir)
        except DatasetTimeout:
            print(f"  TIMED OUT after {time.time()-t0:.0f}s")
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")

    if multiseed_results:
        print(f"\nWritten to {out_json_path}")
    else:
        print(f"\nNothing computed -- no valid dataset name(s) matched. Pass e.g. --multiseed Tolokers,CS "
              f"(comma-separated, matching a REAL_DATASET_SPECS name exactly).")


def plot_real_dataset(name, record, out_dir, fmt="png", show_title=True):
    """One figure per dataset: both curves, zoomed to the CURVES' own data
    range (see _symmetric_ylim) rather than the full [-1, 1] correlation
    scale -- real |rho(t)| here is typically well under 0.2, so a fixed
    axis makes every curve look flat.

    The local/role scalars are reported as a text caption, not as
    reference lines: they live "outside the propagation axis" (methodology
    note) and are often much larger in magnitude than anything the t-sweep
    itself reaches (e.g. Cora's align-local is +0.25 against a curve that
    never leaves [0.02, 0.03]) -- drawing them as horizontal lines would
    force the frame back out to whatever they need, undoing the zoom for
    exactly the datasets that need it most.

    ``fmt`` picks the output format (e.g. "png", "pdf" -- pdf is vector, so
    dpi is only relevant to any rasterized bits). ``show_title=False`` drops
    the per-axes ``name`` title, for figures headed straight into a paper
    where a LaTeX caption already supplies it.
    """
    import matplotlib.pyplot as plt

    align = record["align"]
    task = record["task"]

    curve_values = list(align["rho"]) + (list(task["rho"]) if task is not None else [])
    ylo, yhi = _symmetric_ylim(curve_values)

    fig, ax = plt.subplots(figsize=(5.5, 4.3))
    ax.plot(align["t_grid"], align["rho"], marker="o", color="tab:blue", label=r"$\rho_{\mathrm{align}}(t)$")
    if task is not None:
        ax.plot(task["t_grid"], task["rho"], marker="s", color="tab:orange", label=r"$\rho_{\mathrm{task}}(t)$")
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("diffusion time t")
    ax.set_ylabel(r"$\rho(t)$")
    ax.set_ylim(ylo, yhi)
    if show_title:
        ax.set_title(name)
    ax.legend(fontsize=9, loc="best")

    caption = (f"local / role (outside t-axis) — align: "
               f"{align['scalar'].get('local', 0):+.2f} / {align['scalar'].get('role', 0):+.2f}")
    if task is not None:
        caption += (f"   task: {task['scalar'].get('local', 0):+.2f} / "
                     f"{task['scalar'].get('role', 0):+.2f}")
    ax.text(0.5, -0.15, caption, transform=ax.transAxes, ha="center", va="top",
            fontsize=7.5, color="dimgray")

    fig.tight_layout()
    fname = out_dir / f"real_{name}.{fmt}"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  wrote {fname}")


def plot_real_summary(per_dataset, out_dir, fmt="png", show_title=True):
    """One scatter of t*/rho* across all completed real datasets, for both
    curves -- a bird's-eye view to accompany the per-dataset figures."""
    import matplotlib.pyplot as plt

    rows = [(name, rec) for name, rec in per_dataset.items() if "error" not in rec]
    if not rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, kind, title in ((axes[0], "align", r"$\rho_{\mathrm{align}}$: peak strength/scale"),
                             (axes[1], "task", r"$\rho_{\mathrm{task}}$: peak strength/scale")):
        for name, rec in rows:
            curve = rec.get(kind)
            if curve is None:
                continue
            ax.scatter(curve["t_star"], curve["rho_star"])
            ax.annotate(name, (curve["t_star"], curve["rho_star"]), fontsize=7,
                        xytext=(3, 3), textcoords="offset points")
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
        ax.set_xlabel(r"$t^\star$")
        if show_title:
            ax.set_title(title)
    axes[0].set_ylabel(r"$\rho^\star$ (signed peak)")
    fig.tight_layout()
    fname = out_dir / f"real_summary.{fmt}"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  wrote {fname}")


def plot_all_curves_grid(kind, per_dataset, out_dir, fmt="png", show_title=True,
                          show_panel_titles=None, show_legend=True, order=None):
    """The "present this to collaborators" figure: ALL completed real
    datasets' rho_align(t) (kind="align") or rho_task(t) (kind="task") in
    one image, one small-multiple panel per dataset, colored by dataset
    family (citation / heterophilous / Amazon-Coauthor / Airports / WebKB /
    Attributed). Each panel keeps its OWN y-zoom (_symmetric_ylim) --
    sharing one axis across all datasets would just recreate the flatness
    problem at a bigger scale, since Brazil-Airports' task curve alone
    (~0.37) is an order of magnitude bigger than most citation-graph
    curves (~0.02-0.05). A curve that abstained (rho_task with no
    cross-fit support) gets an empty, labeled panel instead of silently
    vanishing from the grid.

    ``show_title`` controls only the figure suptitle. ``show_panel_titles``
    controls the per-panel dataset-name title INDEPENDENTLY of the
    suptitle -- default None falls back to ``show_title`` (old coupled
    behavior, so a bare ``show_title=False`` call still drops both, e.g.
    existing --no-titles output is unchanged); pass True/False explicitly
    to decouple them (e.g. no suptitle but per-panel names still on, for a
    paper figure whose caption doesn't itself say which panel is which
    dataset). ``show_legend`` toggles the family-color key at the top --
    colors are used either way, this only hides the legend swatch/labels.
    ``order``, if given, is an explicit sequence of dataset names
    controlling left-to-right, top-to-bottom panel order instead of the
    default category-then-alphabetical sort; any completed dataset not
    named in ``order`` is appended at the end (alphabetically, with a
    printed warning) rather than silently dropped.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    panel_titles = show_title if show_panel_titles is None else show_panel_titles

    rows = [(name, rec) for name, rec in per_dataset.items() if "error" not in rec]
    if order:
        order_index = {name: i for i, name in enumerate(order)}
        unordered = sorted(n for n, _ in rows if n not in order_index)
        if unordered:
            print(f"  NOTE: {len(unordered)} dataset(s) not in the given panel order, "
                  f"appended at the end: {unordered}")
        rows.sort(key=lambda nr: (order_index.get(nr[0], len(order)), nr[0]))
    else:
        rows.sort(key=lambda nr: (DATASET_CATEGORY.get(nr[0], "Other"), nr[0]))
    if not rows:
        print(f"  (no completed real datasets to plot for '{kind}')")
        return

    ncols = 4
    nrows = -(-len(rows) // ncols)  # ceil division
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.1 * ncols, 2.6 * nrows), squeeze=False)

    for idx, (name, rec) in enumerate(rows):
        ax = axes[idx // ncols][idx % ncols]
        curve = rec.get(kind)
        color = CATEGORY_COLOR.get(DATASET_CATEGORY.get(name, "Other"), "tab:gray")
        if curve is None:
            ax.text(0.5, 0.5, "abstained", ha="center", va="center", fontsize=8,
                     color="dimgray", transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])
        else:
            ax.plot(curve["t_grid"], curve["rho"], marker="o", markersize=3.5, color=color)
            ax.axhline(0.0, color="black", linewidth=0.6, alpha=0.5)
            ax.set_ylim(*_symmetric_ylim(curve["rho"]))
            ax.tick_params(labelsize=7)
        if panel_titles:
            ax.set_title(name, fontsize=9)

    for idx in range(len(rows), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    if show_legend:
        cats = sorted({DATASET_CATEGORY.get(n, "Other") for n, _ in rows})
        handles = [Line2D([0], [0], color=CATEGORY_COLOR.get(c, "tab:gray"), marker="o", linestyle="-")
                   for c in cats]
        fig.legend(handles, cats, loc="upper center", ncol=len(cats), fontsize=9, bbox_to_anchor=(0.5, 1.04))

    if show_title:
        curve_symbol = r"$\rho_{\mathrm{align}}(t)$" if kind == "align" else r"$\rho_{\mathrm{task}}(t)$"
        fig.suptitle(f"{curve_symbol} across all real datasets  (each panel independently zoomed to its own range)",
                     y=1.09, fontsize=12)
    fig.supxlabel("diffusion time t", fontsize=10)
    fig.supylabel(r"$\rho(t)$", fontsize=10)
    fig.tight_layout()
    fname = out_dir / f"all_datasets_rho_{kind}.{fmt}"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {fname}")


def plot_homophily_correlations(per_dataset, out, fmt="png"):
    """Spearman-correlate the curve-derived scalars (t*/rho* for both
    curves, plus their local/role scalars) against conventional graph
    statistics (adjusted homophily, label informativeness, label entropy,
    density, degree/spectral summary) across every completed real dataset
    that has conventional_statistics attached -- same spirit as
    real_geometry_check.py's gamma_vs_conventional_correlations, extended
    to the curves rather than just gamma. Writes the full table as JSON
    and a headline homophily-vs-peak-strength scatter as a PNG; both are
    skipped (with a note) if fewer than 4 datasets have the stats.
    """
    from scipy.stats import spearmanr
    import matplotlib.pyplot as plt

    rows = [(name, rec) for name, rec in per_dataset.items()
            if "error" not in rec and "conventional_statistics" in rec]
    missing = [name for name, rec in per_dataset.items()
               if "error" not in rec and "conventional_statistics" not in rec]
    if missing:
        print(f"  NOTE: {len(missing)} dataset(s) missing conventional_statistics, excluded from "
              f"the correlation analysis: {missing}. Re-run this script with the same --out (existing "
              f"gamma/curves are kept -- only the missing stats get backfilled) to add them.")
    if len(rows) < 4:
        print(f"  Only {len(rows)} dataset(s) have conventional_statistics -- skipping the "
              f"homophily correlation (need >= 4).")
        return

    curve_labels = ["align_t*", "align_rho*", "align_local", "align_role",
                    "task_t*", "task_rho*", "task_local", "task_role"]

    def curve_row(rec):
        a = rec["align"]; t = rec.get("task")
        return [
            a["t_star"], a["rho_star"], a["scalar"].get("local", 0.0), a["scalar"].get("role", 0.0),
            t["t_star"] if t else np.nan, t["rho_star"] if t else np.nan,
            t["scalar"].get("local", 0.0) if t else np.nan, t["scalar"].get("role", 0.0) if t else np.nan,
        ]

    names = [n for n, _ in rows]
    C = np.array([curve_row(rec) for _, rec in rows])
    Conv = np.array([rec["conventional_statistics"] for _, rec in rows])
    conv_labels = (rows[0][1].get("conventional_statistics_labels") or CONV_LABELS)
    if Conv.shape[1] != len(conv_labels):
        conv_labels = [f"conv_{j}" for j in range(Conv.shape[1])]

    corr_rows = []
    for ci, cname in enumerate(curve_labels):
        valid = ~np.isnan(C[:, ci])
        if valid.sum() < 4:
            continue
        for vi, vname in enumerate(conv_labels):
            rho, p = spearmanr(C[valid, ci], Conv[valid, vi])
            if rho is None or np.isnan(rho):
                continue
            corr_rows.append({"curve_stat": cname, "conventional_stat": vname,
                               "spearman_rho": float(rho), "p": float(p), "n": int(valid.sum())})
    corr_rows.sort(key=lambda r: -abs(r["spearman_rho"]))

    corr_path = out / "curve_vs_conventional_correlations.json"
    corr_path.write_text(json.dumps(corr_rows, indent=2))
    print(f"\nTop |Spearman rho| between curve statistics and conventional graph statistics "
          f"(n={len(rows)} datasets -- suggestive only, not significance-tested):")
    for r in corr_rows[:12]:
        print(f"  {r['curve_stat']:11s} vs {r['conventional_stat']:22s} "
              f"rho={r['spearman_rho']:+.2f}  (p={r['p']:.2f}, n={r['n']})")
    print(f"  full table: {corr_path}")

    if "adjusted_homophily" not in conv_labels:
        return
    h_idx = conv_labels.index("adjusted_homophily")
    figures_dir = out / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, ci, title in (
        (axes[0], curve_labels.index("align_rho*"), r"adjusted homophily vs $\rho_{\mathrm{align}}^\star$"),
        (axes[1], curve_labels.index("task_rho*"), r"adjusted homophily vs $\rho_{\mathrm{task}}^\star$"),
    ):
        x, y = Conv[:, h_idx], C[:, ci]
        valid = ~np.isnan(y)
        ax.scatter(x[valid], y[valid])
        for n, xv, yv in zip(np.array(names)[valid], x[valid], y[valid]):
            ax.annotate(n, (xv, yv), fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.axhline(0.0, color="black", linewidth=0.6, alpha=0.5)
        ax.set_xlabel("adjusted homophily")
        ax.set_title(title)
    axes[0].set_ylabel(r"$\rho^\star$ (signed peak)")
    fig.tight_layout()
    fname = figures_dir / f"homophily_vs_curves.{fmt}"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  wrote {fname}")


def run_real(args, cfg, t_grid, out):
    try:
        from utils.experiments.runner import (
            REAL_DATASET_SPECS, load_real_problem, extract_representations,
            fit_dictionaries, conventional_statistics,
        )
    except Exception as e:  # torch / torch-geometric missing, etc.
        print(f"\nSkipping real datasets: could not import utils.experiments.runner ({type(e).__name__}: {e})")
        print("Install torch + torch-geometric (and `ogb` for ogbn-arxiv) to run the real side, "
              "or pass --skip-real to silence this.")
        return

    results_path = out / "real_rap_profile.json"
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

    if not specs:
        print("No real datasets selected after --only/--skip filtering -- nothing to do.")
        return

    per_dataset = {}
    if results_path.exists():
        try:
            per_dataset = json.loads(results_path.read_text())
            print(f"Resuming: found {len(per_dataset)} existing result(s) at {results_path}")
        except Exception:
            per_dataset = {}

    figures_dir = out / "figures"
    if args.plot:
        figures_dir.mkdir(parents=True, exist_ok=True)

    def _is_done(nm):
        return nm in per_dataset and "error" not in per_dataset[nm]

    # gamma requires the frozen synthetic-trained dictionaries (Sec 3.4);
    # fit once up front, exactly as real_geometry_check.py does -- but only
    # if at least one selected dataset actually needs full processing.
    # Datasets that already have gamma/curves from an earlier run only need
    # conventional_statistics backfilled (see backfill_conventional_statistics),
    # which never touches the dictionaries at all -- skip the (~1-3 minute)
    # fit entirely when every selected dataset is in that state.
    needs_full = [s for s in specs if not _is_done(s[1])]
    native_templates = task_templates = None
    if needs_full:
        print("Fitting native + task dictionaries on synthetic G_train only (needed to project real fields into gamma) ...")
        t0 = time.time()
        native_templates, task_templates = fit_dictionaries(cfg)
        print(f"  done in {time.time()-t0:.1f}s  (native {native_templates.shape}, task {task_templates.shape})")
    else:
        print("Every selected dataset already has gamma/curves -- skipping dictionary fitting "
              "(only backfilling conventional_statistics where missing).")

    for i, (source, name, metric) in enumerate(specs):
        if _is_done(name):
            if "conventional_statistics" in per_dataset[name]:
                print(f"\n[{name}] already done, skipping (delete its entry in {results_path.name} to redo)")
                continue
            print(f"\n[{name}] already has gamma/curves but no conventional_statistics -- backfilling stats only")
            t0 = time.time()
            try:
                with dataset_timeout(args.timeout_per_dataset):
                    conv = backfill_conventional_statistics(
                        source, name, metric, cfg, args.seed + i, load_real_problem, conventional_statistics,
                    )
                per_dataset[name]["conventional_statistics"] = conv
                per_dataset[name]["conventional_statistics_labels"] = CONV_LABELS
                print(f"  backfilled ({time.time()-t0:.1f}s)")
            except DatasetTimeout:
                print(f"  TIMED OUT backfilling stats after {time.time()-t0:.0f}s -- leaving without it, moving on")
            except Exception as e:
                print(f"  ERROR backfilling stats: {type(e).__name__}: {e} -- leaving without it, moving on")
            results_path.write_text(json.dumps(per_dataset, indent=2))
            continue

        print(f"\n[{name}]")
        t0 = time.time()
        try:
            with dataset_timeout(args.timeout_per_dataset):
                record = process_real_dataset(
                    source, name, metric, cfg, t_grid, args.seed + i,
                    load_real_problem, extract_representations, conventional_statistics,
                    native_templates, task_templates,
                )
            print(f"  ({time.time()-t0:.1f}s)")
            # Store the seed actually used (args.seed + its index in the
            # FILTERED specs list this run saw, which is what process_real_dataset
            # above was called with) directly on the record. Without this, a later
            # --extend-t-grid run has to reconstruct the seed via _dataset_seed's
            # index-into-REAL_DATASET_SPECS fallback, which is only correct if this
            # dataset was originally processed as part of a full-suite (no
            # --only/--skip) run -- not true when a dataset is first added and run
            # via --only, as with a fresh WebKB add. Storing it now removes that
            # foot-gun for every dataset processed from here on.
            record["curve_seed"] = args.seed + i
            per_dataset[name] = record
            if args.plot:
                plot_real_dataset(name, record, figures_dir)
        except DatasetTimeout:
            elapsed = time.time() - t0
            print(f"  TIMED OUT after {elapsed:.0f}s (limit {args.timeout_per_dataset}s) -- skipping, moving on")
            per_dataset[name] = {"source": source, "metric": metric, "error": "timeout",
                                  "timeout_s": args.timeout_per_dataset, "elapsed_s": elapsed}
        except Exception as e:  # one bad dataset shouldn't lose the rest
            print(f"  ERROR: {type(e).__name__}: {e} -- skipping, moving on")
            per_dataset[name] = {"source": source, "metric": metric, "error": f"{type(e).__name__}: {e}"}

        results_path.write_text(json.dumps(per_dataset, indent=2))

    if args.plot:
        plot_real_summary(per_dataset, figures_dir)
        plot_all_curves_grid("align", per_dataset, figures_dir)
        plot_all_curves_grid("task", per_dataset, figures_dir)
        plot_homophily_correlations(per_dataset, out)

    n_errors = sum(1 for v in per_dataset.values() if "error" in v)
    n_ok = len(per_dataset) - n_errors
    print(f"\nReal-dataset results written to {results_path}")
    print(f"Completed: {n_ok}  |  Timed out / errored: {n_errors}  |  Total attempted: {len(specs)}")


# =============================================================================
# Replot: regenerate figures from existing JSON, no recomputation
# =============================================================================

def replot(out: Path, figures_dir: Optional[Path] = None, fmt: str = "png",
          show_title: bool = True, show_panel_titles: Optional[bool] = None,
          show_legend: bool = True, order: Optional[Sequence[str]] = None) -> None:
    """Re-run just the plotting functions against whatever JSON already
    sits under `out` -- for iterating on figure style (axis scaling,
    reference lines, ...) without re-fitting dictionaries or re-loading
    every real dataset. Both result files are plain JSON produced by this
    same script, so nothing here needs torch either.

    Reads are always from `out` (`out/real_rap_profile.json` and
    `out/alignment_curves_synthetic.json`); writes go to `figures_dir` if
    given, else the usual `out/figures` -- so e.g. a paper-ready re-render
    can land in a separate results subdirectory without disturbing the
    original run's figures. `fmt` ("png"/"pdf"/...) and `show_title` are
    forwarded to the real-dataset curve plots (plot_real_dataset,
    plot_real_summary, plot_all_curves_grid) AND to
    plot_homophily_correlations (fmt only -- it always shows its own
    per-panel titles and has no legend to toggle). `show_panel_titles`,
    `show_legend`, and `order` are forwarded to plot_all_curves_grid only
    (see its docstring) -- plot_real_dataset/plot_real_summary don't have
    an equivalent legend/order concept. The synthetic-curve figures are
    unaffected by any of this and always render as before, into
    `out/figures`.
    """
    import matplotlib  # noqa: F401  (fail fast with a clear message if missing)

    if figures_dir is None:
        figures_dir = out / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    found_any = False

    synth_path = out / "alignment_curves_synthetic.json"
    if synth_path.exists():
        d = json.loads(synth_path.read_text())
        print(f"Replotting synthetic curves from {synth_path} ...")
        plot_synthetic(d.get("native", {}), d.get("task", {}), out / "figures")
        found_any = True
    else:
        print(f"  (no {synth_path.name} under {out} -- skipping synthetic figures)")

    real_path = out / "real_rap_profile.json"
    if real_path.exists():
        per_dataset = json.loads(real_path.read_text())
        print(f"Replotting {sum(1 for v in per_dataset.values() if 'error' not in v)} "
              f"real-dataset curve(s) from {real_path} into {figures_dir} (fmt={fmt}, "
              f"titles={'on' if show_title else 'off'}) ...")
        for name, record in per_dataset.items():
            if "error" in record:
                continue
            plot_real_dataset(name, record, figures_dir, fmt=fmt, show_title=show_title)
        plot_real_summary(per_dataset, figures_dir, fmt=fmt, show_title=show_title)
        plot_all_curves_grid("align", per_dataset, figures_dir, fmt=fmt, show_title=show_title,
                              show_panel_titles=show_panel_titles, show_legend=show_legend, order=order)
        plot_all_curves_grid("task", per_dataset, figures_dir, fmt=fmt, show_title=show_title,
                              show_panel_titles=show_panel_titles, show_legend=show_legend, order=order)
        plot_homophily_correlations(per_dataset, out, fmt=fmt)
        found_any = True
    else:
        print(f"  (no {real_path.name} under {out} -- skipping real-dataset figures)")

    if not found_any:
        print(f"Nothing to replot: no result JSON found under {out}. Run the experiment first.")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="./results/real_rap_profile")
    ap.add_argument("--data-root", type=str, default="./data")
    ap.add_argument("--t-grid", type=_parse_t_grid, default=DEFAULT_T_GRID,
                     help=f"Comma-separated diffusion times, e.g. 0.5,1,2,4,8 (default: {DEFAULT_T_GRID})")
    ap.add_argument("--extend-t-grid", type=_parse_t_grid, default=None,
                     help="Comma-separated diffusion times to ADD to the existing curves under --out "
                          "(e.g. 16,20,24). Reuses every dataset's already-computed alpha/beta/gamma and "
                          "every already-computed rho(t) -- only the new t value(s) trigger a fresh "
                          "diffusion sweep. Requires an existing real_rap_profile.json under --out, and "
                          "the SAME --seed / --quick the original run used (see _dataset_seed).")
    ap.add_argument("--quick", action="store_true", help="Smaller graphs/pairs/seeds/dictionaries -- smoke test only")
    ap.add_argument("--seed", type=int, default=30)
    ap.add_argument("--check-components", action="store_true",
                     help="Skip all curve/gamma computation; just report each selected real dataset's "
                          "connected-component structure (settles whether a lambda2~0 reading in "
                          "conventional_statistics is genuine disconnection or a spectral-proxy "
                          "artifact). Writes --out/component_check.json. Honors --only/--skip.")
    ap.add_argument("--multiseed", type=str, default=None, metavar="NAME[,NAME...]",
                     help="Recompute both curves for these real dataset name(s) across "
                          "--n-extra-seeds independent seeds (fresh split + fresh pair/landmark sample "
                          "each time -- NOT the cheap --extend-t-grid reuse) and plot the mean+/-std "
                          "band against the original single-seed curve on file, to check whether a "
                          "curve shape (e.g. a sharp jump) is stable across seeds or a sampling "
                          "artifact of the one seed already run. Writes --out/multiseed_robustness.json "
                          "and, with --plot, --out/figures/multiseed_<name>.png per dataset.")
    ap.add_argument("--n-extra-seeds", type=int, default=5,
                     help="Number of independent seeds for --multiseed (default 5).")
    ap.add_argument("--plot", action="store_true", help="Also write PNG figures under --out/figures")
    ap.add_argument("--replot", action="store_true",
                     help="Skip all computation; just regenerate figures from the JSON "
                          "already under --out (fast -- use this to iterate on figure style)")
    ap.add_argument("--figures-out", type=str, default=None,
                     help="With --replot: write regenerated real-dataset figures here instead of "
                          "--out/figures (e.g. a separate results subdirectory for paper-ready "
                          "exports). Still reads real_rap_profile.json from --out.")
    ap.add_argument("--fig-format", type=str, default="png", choices=["png", "pdf", "svg"],
                     help="With --replot: output format for the real-dataset curve figures "
                          "(default: png).")
    ap.add_argument("--no-titles", action="store_true",
                     help="With --replot: omit the per-figure/per-panel titles on the real-dataset "
                          "curve figures (plot_real_dataset / plot_real_summary / "
                          "plot_all_curves_grid) -- for figures headed into a paper where a caption "
                          "already labels them. plot_all_curves_grid's per-PANEL dataset-name titles "
                          "can be turned back on independently with --panel-titles.")
    ap.add_argument("--panel-titles", action="store_true",
                     help="With --replot: force the per-panel dataset-name titles on in "
                          "plot_all_curves_grid's all_datasets_rho_{align,task} figure, independent "
                          "of --no-titles (which otherwise also suppresses them, along with the "
                          "figure suptitle). No effect if --no-titles isn't also passed.")
    ap.add_argument("--no-legend", action="store_true",
                     help="With --replot: hide the family-color legend at the top of "
                          "plot_all_curves_grid's all_datasets_rho_{align,task} figure. Colors are "
                          "still used per dataset family -- this only hides the legend itself.")
    ap.add_argument("--panel-order", type=str, default=None, metavar="NAME[,NAME...]",
                     help="With --replot: explicit comma-separated dataset order (left-to-right, "
                          "top-to-bottom) for plot_all_curves_grid's all_datasets_rho_{align,task} "
                          "figure, instead of the default category-then-alphabetical sort. Any "
                          "completed dataset not named here is appended at the end.")

    ap.add_argument("--skip-synthetic", action="store_true", help="Skip the synthetic ground-truth-shape check")
    ap.add_argument("--synthetic-n", type=int, default=300, help="Node count for each synthetic mechanism graph")
    ap.add_argument("--synthetic-seeds", type=int, default=8, help="Held-out seeds averaged per mechanism")

    ap.add_argument("--skip-real", action="store_true")
    ap.add_argument("--include-ogb", action="store_true", help="Also run ogbn-arxiv (slow role-signature step)")
    ap.add_argument("--timeout-per-dataset", type=int, default=480, help="Seconds before giving up on one real dataset (0 disables)")
    ap.add_argument("--skip", nargs="+", action="extend", default=[], metavar="NAME",
                     help="Real dataset name(s) to skip (repeatable / space- / comma-separated)")
    ap.add_argument("--only", nargs="+", action="extend", default=[], metavar="NAME",
                     help="Real dataset name(s) to run exclusively (overrides --skip)")
    args = ap.parse_args()

    if args.quick:
        args.synthetic_n = min(args.synthetic_n, 120)
        args.synthetic_seeds = min(args.synthetic_seeds, 2)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t_grid = tuple(args.t_grid)

    if args.replot:
        print(f"--replot: regenerating figures from existing results under {out} (no recomputation)\n")
        replot(
            out,
            figures_dir=Path(args.figures_out) if args.figures_out else None,
            fmt=args.fig_format,
            show_title=not args.no_titles,
            show_panel_titles=(True if args.panel_titles else None),
            show_legend=not args.no_legend,
            order=_parse_name_list([args.panel_order]) if args.panel_order else None,
        )
        return

    if args.extend_t_grid:
        print(f"--extend-t-grid: adding t={list(args.extend_t_grid)} to existing curves under {out}\n")
        try:
            from utils.experiments.runner import ExperimentConfig
        except Exception as e:
            print(f"Cannot extend: could not import utils.experiments.runner ({type(e).__name__}: {e})")
            print("Install torch + torch-geometric (and `ogb` for ogbn-arxiv) to extend the real side.")
            return
        cfg = ExperimentConfig.quick() if args.quick else ExperimentConfig()
        cfg.data_root = args.data_root
        if args.quick:
            print("WARNING: --quick changes n_landmarks/m_pairs/etc. -- only pass --quick here if the "
                  "ORIGINAL run that produced real_rap_profile.json also used --quick, or the new t's "
                  "will be sampled on a different landmark/pair set than the rest of the curve.")
        run_extend(args, cfg, out)
        return

    if args.check_components:
        print(f"--check-components: reporting connected-component structure under {out}\n")
        try:
            from utils.experiments.runner import ExperimentConfig
        except Exception as e:
            print(f"Cannot check components: could not import utils.experiments.runner ({type(e).__name__}: {e})")
            print("Install torch + torch-geometric (and `ogb` for ogbn-arxiv) to check the real side.")
            return
        cfg = ExperimentConfig.quick() if args.quick else ExperimentConfig()
        cfg.data_root = args.data_root
        run_check_components(args, cfg, out)
        return

    if args.multiseed:
        print(f"--multiseed: robustness-checking {args.multiseed} across {args.n_extra_seeds} new seed(s)\n")
        try:
            from utils.experiments.runner import ExperimentConfig
        except Exception as e:
            print(f"Cannot run multiseed check: could not import utils.experiments.runner ({type(e).__name__}: {e})")
            print("Install torch + torch-geometric (and `ogb` for ogbn-arxiv) to check the real side.")
            return
        cfg = ExperimentConfig.quick() if args.quick else ExperimentConfig()
        cfg.data_root = args.data_root
        run_multiseed(args, cfg, t_grid, out)
        return

    synthetic_native, synthetic_task = {}, {}
    if not args.skip_synthetic:
        print(f"Synthetic side: {args.synthetic_seeds} seed(s) per mechanism, n={args.synthetic_n}, t_grid={t_grid}\n")
        synthetic_native = synthetic_native_curves(
            t_grid, args.synthetic_n, args.synthetic_seeds, n_landmarks=20, m_pairs=3000, seed0=args.seed,
        )
        synthetic_task = synthetic_task_curves(
            t_grid, args.synthetic_n, args.synthetic_seeds, n_landmarks=20,
            label_pairs_cap=4000, kappa=3, seed0=args.seed,
        )
        synth_path = out / "alignment_curves_synthetic.json"
        synth_path.write_text(json.dumps({"native": synthetic_native, "task": synthetic_task}, indent=2))
        print(f"\nSynthetic results written to {synth_path}")

        if args.plot:
            figures_dir = out / "figures"
            figures_dir.mkdir(parents=True, exist_ok=True)
            plot_synthetic(synthetic_native, synthetic_task, figures_dir)
    else:
        print("Skipping synthetic side (--skip-synthetic).")

    if not args.skip_real:
        # ExperimentConfig itself lives in utils.experiments.runner (same
        # torch-at-import-time constraint as load_real_problem), so it is
        # imported here, not at module top level -- keeps --skip-real usable
        # without torch installed at all.
        try:
            from utils.experiments.runner import ExperimentConfig
        except Exception as e:
            print(f"\nSkipping real datasets: could not import utils.experiments.runner ({type(e).__name__}: {e})")
            print("Install torch + torch-geometric (and `ogb` for ogbn-arxiv) to run the real side, "
                  "or pass --skip-real to silence this.")
        else:
            cfg = ExperimentConfig.quick() if args.quick else ExperimentConfig()
            cfg.data_root = args.data_root
            print()
            run_real(args, cfg, t_grid, out)
    else:
        print("\nSkipping real datasets (--skip-real).")


if __name__ == "__main__":
    main()
