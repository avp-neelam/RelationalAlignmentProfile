"""
Dictionary-free alignment curves for the Relational Alignment Profile (RAP).

This module adds the two multiscale correlation curves that sit alongside
the compact gamma = [alpha, beta] coordinate vector already produced by
``utils.grammar.core``. Nothing here touches alpha/beta/gamma or the NNLS
dictionaries -- it is a second, curve-shaped readout of the SAME rank-
transformed pair samples core.py already knows how to build.

  rho_align(t)  structural alignment curve -- does structure organize the
                attributes? Spearman rank correlation between diffusion-
                time structural proximity U_S^(t) and feature proximity
                U_X, for every t on a scale sweep.

  rho_task(t)   task alignment curve -- does structure organize the task
                *beyond* the attributes? Spearman rank correlation between
                U_S^(t) and the endpoint-disjoint cross-fit residual of
                label agreement after regressing on feature proximity,
                sign-flipped so positive means "structurally close pairs
                agree on label more than features alone predict."

See the methodology note, Sec "Reading RAP: Alignment Curves"
(rho_align: Eq. main-rho-align/main-rho-spearman; rho_task: Eq.
main-rho-task) for the definitions this implements.

Implementation note on rank transforms
---------------------------------------
Spearman's rho depends only on the *rank order* of each variable, and every
transform used elsewhere in this codebase to build U_S / U_X
(midrank_uniform / fit_rank_transform) is monotone non-decreasing and tie-
consistent (average-rank midranks, matching scipy.stats.spearmanr's own
tie handling). So Corr(U_S^(t), U_X) computed via scipy.stats.spearmanr on
the *raw* distances is identical to Corr on the rank-transformed U's --
spearmanr performs its own rank transform internally. This lets the curve
code skip an explicit fit_rank_transform round-trip for every t on the
correlation step; the one place a genuine U_X in [0,1] is still required
is the cross-fit residual regression (core._binned_regression_fit bins its
covariate over the unit interval), so U_X is computed explicitly there,
exactly as in core.compute_geometric_grammar.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import networkx as nx
from scipy.stats import spearmanr

from . import core as gg

__all__ = [
    "DEFAULT_T_GRID",
    "AlignmentCurve",
    "RAPCurves",
    "structural_alignment_curve",
    "task_alignment_curve",
    "compute_alignment_curves",
    "plot_alignment_curve",
]


# A log-ish spaced default scale sweep: t << 1 resolves ~1-hop structure,
# t >> graph mixing time approaches the stationary distribution (every U_S
# collapses toward a constant, so rho_align/rho_task decay toward 0 there
# regardless of mechanism). Override per-graph if diameter/mixing time
# suggests different bounds -- e.g. a much larger t_grid for a
# high-diameter graph like Roman-empire.
DEFAULT_T_GRID: Tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0)


@dataclass
class AlignmentCurve:
    """One curve (either rho_align or rho_task) plus its peak summary."""

    kind: str                      # "align" | "task"
    t_grid: np.ndarray             # (T,)
    rho: np.ndarray                # (T,) rho(t)
    p_value: np.ndarray            # (T,) two-sided Spearman p-value at each t
    n_pairs: int
    t_star: float                  # arg max_t |rho(t)|          (Eq. main-peak)
    rho_star: float                # rho(t_star), signed
    scalar: Dict[str, float] = field(default_factory=dict)      # {"local":.., "role":..}
    scalar_p: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "t_grid": np.asarray(self.t_grid).tolist(),
            "rho": np.asarray(self.rho).tolist(),
            "p_value": np.asarray(self.p_value).tolist(),
            "n_pairs": int(self.n_pairs),
            "t_star": float(self.t_star),
            "rho_star": float(self.rho_star),
            "scalar": {k: float(v) for k, v in self.scalar.items()},
            "scalar_p": {k: float(v) for k, v in self.scalar_p.items()},
        }


@dataclass
class RAPCurves:
    """Both curves for one (G, X[, y]) problem."""

    align: AlignmentCurve
    task: Optional[AlignmentCurve]           # None => abstained, see task_abstain_reason
    task_abstain_reason: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "align": self.align.to_dict(),
            "task": self.task.to_dict() if self.task is not None else None,
            "task_abstain_reason": self.task_abstain_reason,
        }


def _safe_spearman(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    """spearmanr guarded against degenerate inputs (constant arrays, too
    few points), which otherwise return NaN and warn."""
    a = np.asarray(a); b = np.asarray(b)
    if len(a) < 3 or np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return 0.0, 1.0
    rho, p = spearmanr(a, b)
    if rho is None or np.isnan(rho):
        return 0.0, 1.0
    return float(rho), (float(p) if p is not None and not np.isnan(p) else 1.0)


def _peak(t_grid: np.ndarray, rho: np.ndarray) -> Tuple[float, float]:
    idx = int(np.argmax(np.abs(rho)))
    return float(t_grid[idx]), float(rho[idx])


# =============================================================================
# rho_align(t)
# =============================================================================

def structural_alignment_curve(
    G: nx.Graph,
    X: np.ndarray,
    t_grid: Sequence[float] = DEFAULT_T_GRID,
    n_landmarks: int = 20,
    m_pairs: int = 3000,
    order: int = 30,
    seed: int = 0,
    include_scalar_geometries: bool = True,
) -> AlignmentCurve:
    """rho_align(t) = Corr(U_S^(t), U_X)  (Eq. main-rho-align).

    Label-free: only needs (G, X). Uses one graph-global uniform pair
    sample shared across every t -- propagation_embeddings amortizes the
    Laplacian factorization across the whole t_grid in one call, so this
    is one Chebyshev pass per scale, not per (scale, pair) combination,
    and every t is evaluated on the identical pair set for a clean
    apples-to-apples sweep.
    """
    t_grid = np.asarray(sorted(set(float(t) for t in t_grid)), dtype=float)
    n = G.number_of_nodes()
    if n < 3:
        raise ValueError("Need at least 3 nodes to compute a rank correlation")

    pairs = gg.uniform_pair_sample(n, m_pairs, seed=seed)
    raw_X = np.linalg.norm(X[pairs[:, 0]] - X[pairs[:, 1]], axis=1)

    emb = gg.propagation_embeddings(G, tuple(t_grid), min(n_landmarks, n - 1), order=order, seed=seed)
    rhos, ps = [], []
    for t in t_grid:
        raw_t = gg.pairwise_from_embedding(emb[t], pairs)
        rho, p = _safe_spearman(raw_t, raw_X)
        rhos.append(rho)
        ps.append(p)
    rhos, ps = np.array(rhos), np.array(ps)
    t_star, rho_star = _peak(t_grid, rhos)

    scalar, scalar_p = {}, {}
    if include_scalar_geometries:
        # d_local: balanced edge/nonedge diagnostic law (core.py Sec 3.1),
        # not the uniform pair law used above -- otherwise edge mass
        # vanishes like |E|/n^2 on sparse graphs and the local channel
        # goes silent. Reported as a scalar "outside the propagation axis"
        # (Sec "Reading RAP"), matching how alpha's local channel is
        # reported: a single coefficient, not a t-sweep.
        local_prs = gg.sample_local_edge_nonedge_pairs(G, m_pairs, seed=seed + 17, edge_fraction=0.5)
        adj = nx.to_scipy_sparse_array(G, format="csr")
        raw_local = 1.0 - np.asarray(adj[local_prs[:, 0], local_prs[:, 1]]).ravel()
        raw_X_local = np.linalg.norm(X[local_prs[:, 0]] - X[local_prs[:, 1]], axis=1)
        scalar["local"], scalar_p["local"] = _safe_spearman(raw_local, raw_X_local)

        role_sig = gg.role_signature(G)
        raw_role = np.linalg.norm(role_sig[pairs[:, 0]] - role_sig[pairs[:, 1]], axis=1)
        scalar["role"], scalar_p["role"] = _safe_spearman(raw_role, raw_X)

    return AlignmentCurve(
        kind="align", t_grid=t_grid, rho=rhos, p_value=ps, n_pairs=len(pairs),
        t_star=t_star, rho_star=rho_star, scalar=scalar, scalar_p=scalar_p,
    )


# =============================================================================
# rho_task(t)
# =============================================================================

def _sample_labeled_pairs(labeled_nodes: np.ndarray, m_pairs: int, seed: int) -> np.ndarray:
    """Same sampling law as core.compute_geometric_grammar's task branch:
    exhaustive V_L x V_L if it fits under the cap, else a uniform ordered
    Monte Carlo sample of labeled pairs."""
    rng = np.random.default_rng(seed)
    max_ordered = len(labeled_nodes) * (len(labeled_nodes) - 1)
    if max_ordered <= m_pairs:
        ii, jj = np.meshgrid(labeled_nodes, labeled_nodes, indexing="ij")
        lp = np.stack([ii.ravel(), jj.ravel()], axis=1)
        return lp[lp[:, 0] != lp[:, 1]]
    n_pairs = min(m_pairs, max_ordered)
    li = rng.integers(0, len(labeled_nodes), size=n_pairs * 2)
    lj = rng.integers(0, len(labeled_nodes), size=n_pairs * 2)
    good = li != lj
    li, lj = li[good][:n_pairs], lj[good][:n_pairs]
    return np.stack([labeled_nodes[li], labeled_nodes[lj]], axis=1)


def _endpoint_disjoint_residuals(
    pairs: np.ndarray, U_X: np.ndarray, Zbar_Y: np.ndarray, node_fold: dict,
    kappa: int, min_train: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-pair endpoint-disjoint cross-fit residuals.

    Mirrors core.endpoint_disjoint_residual_field's kappa^2 fold loop
    exactly (same core._binned_regression_fit, same excluded-fold-pair
    training rule), but returns the raw per-pair residual array and
    validity mask instead of aggregating into a K-bin field: the curve
    needs a residual *per labeled pair* so it can be correlated against
    U_S^(t) at every t, not a single quantile-binned summary.
    """
    fold_of_i = np.array([node_fold[i] for i in pairs[:, 0]])
    fold_of_j = np.array([node_fold[i] for i in pairs[:, 1]])
    residuals = np.zeros(len(pairs))
    valid = np.zeros(len(pairs), dtype=bool)
    for v in range(kappa):
        for w in range(v, kappa):
            excluded = {v, w}
            train_mask = ~np.isin(fold_of_i, list(excluded)) & ~np.isin(fold_of_j, list(excluded))
            eval_mask = ((fold_of_i == v) & (fold_of_j == w)) | ((fold_of_i == w) & (fold_of_j == v))
            if train_mask.sum() < min_train or not eval_mask.any():
                continue
            mu = gg._binned_regression_fit(U_X[train_mask], Zbar_Y[train_mask])
            residuals[eval_mask] = Zbar_Y[eval_mask] - mu(U_X[eval_mask])
            valid[eval_mask] = True
    return residuals, valid


def task_alignment_curve(
    G: nx.Graph,
    X: np.ndarray,
    y: Dict[int, int],
    t_grid: Sequence[float] = DEFAULT_T_GRID,
    n_landmarks: int = 20,
    label_pairs_cap: int = 4000,
    kappa: int = 3,
    order: int = 30,
    seed: int = 0,
    include_scalar_geometries: bool = True,
) -> Tuple[Optional[AlignmentCurve], Optional[str]]:
    """rho_task(t) = -Corr(U_S^(t), residual)  (Eq. main-rho-task).

    Returns (curve, abstain_reason). curve is None, with abstain_reason
    set, exactly when core.compute_geometric_grammar's task grammar would
    also decline to report beta_tau: too few labeled nodes, or too little
    cross-fit-valid support in the residual (mirrors has_support there). A
    flat/near-zero curve is still a real, reportable answer ("no task
    signal at any scale"); abstention means the estimate itself is not
    trustworthy enough to report at all.
    """
    t_grid = np.asarray(sorted(set(float(t) for t in t_grid)), dtype=float)
    labeled_nodes = np.array(sorted(y.keys()), dtype=np.int64)
    if len(labeled_nodes) < max(4, kappa):
        return None, f"too few labeled nodes ({len(labeled_nodes)} < max(4, kappa={kappa}))"

    node_fold = gg.assign_node_folds(labeled_nodes, kappa, seed=seed)
    lp = _sample_labeled_pairs(labeled_nodes, label_pairs_cap, seed)
    if len(lp) == 0:
        return None, "no valid labeled pairs sampled"

    labels_i = np.array([y[i] for i in lp[:, 0]])
    labels_j = np.array([y[j] for j in lp[:, 1]])
    Z = (labels_i == labels_j).astype(float)
    Zbar = Z - Z.mean()

    raw_X = np.linalg.norm(X[lp[:, 0]] - X[lp[:, 1]], axis=1)
    U_X = gg.fit_rank_transform(raw_X)(raw_X)   # genuine uniform ranks: the cross-fit
                                                 # regression bins this covariate over [0,1].

    residuals, valid = _endpoint_disjoint_residuals(lp, U_X, Zbar, node_fold, kappa)
    if valid.sum() < max(20, 0.3 * len(lp)):
        return None, (
            f"insufficient cross-fit support ({int(valid.sum())} valid of "
            f"{len(lp)} labeled pairs; need >= max(20, 30% of pairs))"
        )

    n = G.number_of_nodes()
    emb = gg.propagation_embeddings(G, tuple(t_grid), min(n_landmarks, n - 1), order=order, seed=seed)
    rhos, ps = [], []
    for t in t_grid:
        raw_t = gg.pairwise_from_embedding(emb[t], lp)
        rho, p = _safe_spearman(raw_t[valid], residuals[valid])
        rhos.append(-rho)   # sign flip: positive => structurally close pairs agree
        ps.append(p)        # on label MORE than features alone predict
    rhos, ps = np.array(rhos), np.array(ps)
    t_star, rho_star = _peak(t_grid, rhos)

    scalar, scalar_p = {}, {}
    if include_scalar_geometries:
        adj = nx.to_scipy_sparse_array(G, format="csr")
        raw_local = 1.0 - np.asarray(adj[lp[:, 0], lp[:, 1]]).ravel()
        rho_l, p_l = _safe_spearman(raw_local[valid], residuals[valid])
        scalar["local"], scalar_p["local"] = -rho_l, p_l

        role_sig = gg.role_signature(G)
        raw_role = np.linalg.norm(role_sig[lp[:, 0]] - role_sig[lp[:, 1]], axis=1)
        rho_r, p_r = _safe_spearman(raw_role[valid], residuals[valid])
        scalar["role"], scalar_p["role"] = -rho_r, p_r

    return AlignmentCurve(
        kind="task", t_grid=t_grid, rho=rhos, p_value=ps, n_pairs=int(valid.sum()),
        t_star=t_star, rho_star=rho_star, scalar=scalar, scalar_p=scalar_p,
    ), None


# =============================================================================
# Combined entry point
# =============================================================================

def compute_alignment_curves(
    G: nx.Graph,
    X: np.ndarray,
    y: Optional[Dict[int, int]] = None,
    t_grid: Sequence[float] = DEFAULT_T_GRID,
    n_landmarks: int = 20,
    m_pairs: int = 3000,
    label_pairs_cap: int = 4000,
    kappa: int = 3,
    order: int = 30,
    seed: int = 0,
    include_scalar_geometries: bool = True,
) -> RAPCurves:
    """Compute both curves for one problem in one call.

    Node ids in G, X (rows), and y (keys) must already agree (0..n-1, or
    any consistent integer labeling) -- this mirrors
    extract_representations' convention of leaving relabeling to the
    caller; Problem objects from utils.experiments.runner already satisfy
    it, as do dicts returned by core.generate_mechanism_graph /
    core.generate_task_mechanism_graph.
    """
    align = structural_alignment_curve(
        G, X, t_grid=t_grid, n_landmarks=n_landmarks, m_pairs=m_pairs,
        order=order, seed=seed, include_scalar_geometries=include_scalar_geometries,
    )
    task, reason = None, "no labels provided"
    if y is not None:
        task, reason = task_alignment_curve(
            G, X, y, t_grid=t_grid, n_landmarks=n_landmarks,
            label_pairs_cap=label_pairs_cap, kappa=kappa, order=order,
            seed=seed, include_scalar_geometries=include_scalar_geometries,
        )
    return RAPCurves(align=align, task=task, task_abstain_reason=reason)


# =============================================================================
# Optional plotting (matplotlib imported lazily -- not a hard dependency)
# =============================================================================

def plot_alignment_curve(curve: AlignmentCurve, ax=None, label: Optional[str] = None):
    """Line plot of one curve vs t, for visual inspection (Sec "Reading
    RAP"): a peak-then-decay curve says local structure carries the
    signal; a curve that starts flat and rises says the signal is
    delayed/multihop; a negative curve says opposition/heterophily."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(curve.t_grid, curve.rho, marker="o", label=label or curve.kind)
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("diffusion time t")
    ax.set_ylabel(r"$\rho_{\mathrm{align}}(t)$" if curve.kind == "align" else r"$\rho_{\mathrm{task}}(t)$")
    ax.set_ylim(-1.05, 1.05)
    if label:
        ax.legend()
    return ax


# =============================================================================
# Demo / synthetic sanity check
# =============================================================================

if __name__ == "__main__":
    print("Synthetic sanity check: does each NATIVE mechanism's rho_align(t) shape match its story?")
    print(f"t_grid = {list(DEFAULT_T_GRID)}\n")
    for mech in gg.NATIVE_MECHANISMS:
        data = gg.generate_mechanism_graph(mech, n=300, seed=7)
        curve = structural_alignment_curve(data["G"], data["X"], seed=11)
        shape = " ".join(f"{r:+.2f}" for r in curve.rho)
        print(f"  {mech:22s} t*={curve.t_star:5.2f} rho*={curve.rho_star:+.3f}  "
              f"local={curve.scalar.get('local', 0):+.2f} role={curve.scalar.get('role', 0):+.2f}  [{shape}]")

    print("\nSynthetic sanity check: TASK mechanisms' rho_task(t)\n")
    for mech in gg.TASK_MECHANISMS:
        data = gg.generate_task_mechanism_graph(mech, n=300, seed=7)
        curve, reason = task_alignment_curve(data["G"], data["X"], data["y"], seed=11)
        if curve is None:
            print(f"  {mech:22s} ABSTAIN: {reason}")
            continue
        shape = " ".join(f"{r:+.2f}" for r in curve.rho)
        print(f"  {mech:22s} t*={curve.t_star:5.2f} rho*={curve.rho_star:+.3f}  "
              f"local={curve.scalar.get('local', 0):+.2f} role={curve.scalar.get('role', 0):+.2f}  [{shape}]")

    print("\nDone.")
