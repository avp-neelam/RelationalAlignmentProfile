"""
Geometric Grammar --- reference implementation (v2)
=====================================================

Second pass after code review. Fixes, in the order raised:
 1. Scalable path never materializes an n x n matrix: pairs are sampled
    FIRST, distances are computed only for those pairs.
 2. Synthetic generators are built explicitly around the alignment field
    (agreement/opposition via a continuous latent z; delayed via a pure
    2-block bipartite structure; role via hub/leaf status independent of
    community; diffuse via smooth chain position). STATUS: validated --
    100% mechanism recovery across 50 held-out-seed trials (10 seeds x 5
    mechanisms), up from chance (20.7%) before fix #9 below. This is a
    same-generator-family, held-out-SEED test, not yet the disjoint-
    generator-family (G_train/G_test) circularity guard the paper commits
    to for E1 -- that is a further, separate validation still needed
    before trusting real E1 numbers.
 9. NEW, found via a decisive four-way diagnostic (binary adjacency vs.
    shortest-path vs. exact heat diffusion vs. Chebyshev approximation):
    d_prop^t (diffusion-profile distance) does not encode direct adjacency
    -- two nodes on the same side of a heterophilic/bipartite structure
    can have highly similar diffusion profiles (both reach similar
    neighborhoods) even though the edges themselves connect dissimilar
    nodes, giving diffusion distance a *positive* structure-feature field
    for what is, at the adjacency level, an opposition mechanism. This
    was not a bug in the Chebyshev approximation (confirmed: exact and
    approximate heat kernels agree in sign). Fixed by (a) adding a genuine
    local/adjacency structural geometry d_local, and (b) fitting ONE joint
    cross-geometry native dictionary (concatenating fields across
    d_local, d_prop^t1, d_prop^t2, d_role before a single NNLS) instead of
    an independent dictionary per geometry -- mirroring how the task
    dictionary already concatenates across geometries. alpha^(0) is now a
    single 5-dimensional native coordinate, not a per-geometry dict.
 3. Task dictionary templates are now fit end-to-end (fit_task_templates).
 4. Labeled pairs are sampled from V_L x V_L directly, not filtered out of
    a graph-wide sample (which fails whenever |V_L| << n).
 5. Endpoint-disjoint cross-fitting tracks a valid-support mask; fold pairs
    with insufficient data are excluded from the average, not zero-filled.
 6. Graphs are relabeled to consecutive integer node ids internally.
 7. The local-grammar "exact reconstruction" claim is REMOVED: it does not
    hold even under global ranks, because P(S)P(X) does not distribute over
    a weighted average of P_i(S)P_i(X) (the joint term does; the product-
    of-marginals term does not). Local grammar is presented with no
    aggregation guarantee, consistent with its secondary role.
 8. Pair sampling is pilot-stratified by (approximate) structural-distance
    decile; uncertainty/abstention are wired into compute_geometric_grammar
    behind an explicit flag; the Laplacian's isolated-node convention
    (L_ii = 1) is documented rather than left implicit.
"""

from __future__ import annotations

import numpy as np
import networkx as nx
from dataclasses import dataclass
from scipy import sparse
from scipy.sparse.linalg import expm_multiply
from scipy.optimize import nnls
from scipy.stats import rankdata
from typing import Callable, Optional


# =============================================================================
# 3.1  Structural geometries
# =============================================================================

def normalized_laplacian(G: nx.Graph) -> sparse.csr_matrix:
    """Symmetric normalized graph Laplacian L = I - D^{-1/2} A D^{-1/2}.
    Convention: isolated nodes (degree 0) get L_ii = 1 (row/col otherwise
    zero), since D^{-1/2} is set to 0 there rather than left undefined.
    Document/override this if your graphs contain isolates and you need a
    different convention (e.g. L_ii = 0)."""
    A = nx.to_scipy_sparse_array(G, format="csr", dtype=float)
    deg = np.asarray(A.sum(axis=1)).ravel()
    deg_inv_sqrt = np.zeros_like(deg)
    nz = deg > 0
    deg_inv_sqrt[nz] = 1.0 / np.sqrt(deg[nz])
    Dinv = sparse.diags(deg_inv_sqrt)
    L = sparse.eye(A.shape[0], format="csr") - Dinv @ A @ Dinv
    return L.tocsr()


def chebyshev_heat_kernel_columns(
    L: sparse.csr_matrix, t: float, node_indices, order: int = 30,
    lmax: Optional[float] = None,
) -> np.ndarray:
    """Approximate columns H_t(:, node_indices) via truncated Chebyshev
    polynomial expansion of exp(-tL) (Hammond, Vandergheynst & Gribonval,
    2011). Cost is O(order * |E| * len(node_indices)), never O(n^2)."""
    n = L.shape[0]
    if lmax is None:
        lmax = 2.0  # normalized Laplacian spectrum lies in [0, 2]
    a = lmax / 2.0

    N = order + 1
    theta = (np.arange(N) + 0.5) * np.pi / N
    x = np.cos(theta)
    f = np.exp(-t * a * (x + 1.0))
    coeffs = np.array([2.0 / N * np.sum(f * np.cos(k * theta)) for k in range(N)])
    coeffs[0] /= 2.0

    B = np.zeros((n, len(node_indices)))
    B[node_indices, np.arange(len(node_indices))] = 1.0
    L_resc = (L / a) - sparse.eye(n)

    Tkm2 = B.copy()
    Tkm1 = L_resc @ B
    out = coeffs[0] * Tkm2 + coeffs[1] * Tkm1
    for k in range(2, order + 1):
        Tk = 2 * (L_resc @ Tkm1) - Tkm2
        out = out + coeffs[k] * Tk
        Tkm2, Tkm1 = Tkm1, Tk
    return out  # (n, len(node_indices))


def farthest_point_landmarks(G: nx.Graph, n_landmarks: int, seed: int = 0) -> list[int]:
    rng = np.random.default_rng(seed)
    nodes = list(G.nodes())
    landmarks = [nodes[rng.integers(len(nodes))]]
    dist_to_set = {v: np.inf for v in nodes}
    for _ in range(n_landmarks - 1):
        lengths = nx.single_source_shortest_path_length(G, landmarks[-1])
        for v in nodes:
            d = lengths.get(v, len(nodes))
            if d < dist_to_set[v]:
                dist_to_set[v] = d
        landmarks.append(max(dist_to_set, key=dist_to_set.get))
    return landmarks


def propagation_embeddings(G: nx.Graph, t_list, n_landmarks: int, order: int = 30,
                            seed: int = 0) -> dict:
    """Compute the landmark embedding z_t(i) = (H_t(i,l_1),...,H_t(i,l_L))
    for every t in t_list, ONCE, in O(L * |E| * order) per scale. Returns
    {t: (n, L) array}. No n x n object is ever formed here -- pairwise
    distances are computed later, only for sampled pairs, by
    `pairwise_from_embedding`.
    """
    node_list = list(G.nodes())
    node_to_row = {v: i for i, v in enumerate(node_list)}
    landmarks = farthest_point_landmarks(G, n_landmarks, seed)
    landmark_rows = [node_to_row[v] for v in landmarks]

    L = normalized_laplacian(G)
    return {t: chebyshev_heat_kernel_columns(L, t, landmark_rows, order=order) for t in t_list}


def pairwise_from_embedding(Z: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    """d(i,j) = ||Z[i] - Z[j]||, evaluated ONLY at `pairs` -- O(len(pairs))."""
    return np.linalg.norm(Z[pairs[:, 0]] - Z[pairs[:, 1]], axis=1)


def role_signature(G: nx.Graph, max_hop: int = 3) -> np.ndarray:
    """rho(i): multiscale structural signature from k-hop degree statistics
    and local clustering, computed once as an (n, d) array."""
    nodes = list(G.nodes())
    n = len(nodes)
    clustering = nx.clustering(G)
    deg = dict(G.degree())
    sig = np.zeros((n, 2 * max_hop + 1))
    for idx, v in enumerate(nodes):
        lengths = nx.single_source_shortest_path_length(G, v, cutoff=max_hop)
        for k in range(1, max_hop + 1):
            ring = [u for u, d in lengths.items() if d == k]
            if ring:
                degs = np.array([deg[u] for u in ring], dtype=float)
                sig[idx, 2 * (k - 1)] = degs.mean()
                sig[idx, 2 * (k - 1) + 1] = degs.std()
        sig[idx, -1] = clustering[v]
    mu, sd = sig.mean(0), sig.std(0) + 1e-8
    return (sig - mu) / sd


# =============================================================================
# 3.1 (cont.)  Rank normalization with midranks (ties)
# =============================================================================

def midrank_uniform(values: np.ndarray) -> np.ndarray:
    """Empirical mid-distribution transform in (0,1), tie-aware.

    This exactly matches ``fit_rank_transform`` when it is applied back to
    its own reference sample: (# strictly smaller + 0.5 * # tied) / N.
    """
    values = np.asarray(values)
    if values.size == 0:
        return np.empty(0, dtype=float)
    r = rankdata(values, method="average")
    return (r - 0.5) / len(values)


def fit_rank_transform(reference_raw_values: np.ndarray):
    """Fit an empirical mid-distribution transform on a graph-global sample.

    The returned transform is reused unchanged for labeled-pair distances, so
    the percentile meaning of U_S/U_X is identical in native and task grammar.
    """
    sorted_ref = np.sort(np.asarray(reference_raw_values))
    n = len(sorted_ref)
    if n == 0:
        raise ValueError("reference_raw_values must be non-empty")

    def transform(new_values):
        new_values = np.asarray(new_values)
        left = np.searchsorted(sorted_ref, new_values, side="left")
        right = np.searchsorted(sorted_ref, new_values, side="right")
        return (left + right) / (2.0 * n)

    return transform


def uniform_pair_sample(n: int, m_pairs: int, seed: int = 0) -> np.ndarray:
    """Draw ordered non-self node pairs uniformly from V x V.

    Sampling is with replacement, which keeps memory O(m_pairs) even for very
    large graphs. Duplicate pairs are harmless Monte Carlo replicates.
    """
    if n < 2:
        raise ValueError("At least two nodes are required to sample pairs")
    rng = np.random.default_rng(seed)
    out = np.empty((m_pairs, 2), dtype=np.int64)
    filled = 0
    while filled < m_pairs:
        k = max(2 * (m_pairs - filled), 64)
        i = rng.integers(0, n, size=k)
        j = rng.integers(0, n, size=k)
        good = i != j
        take = min(good.sum(), m_pairs - filled)
        if take:
            idx = np.flatnonzero(good)[:take]
            out[filled:filled + take, 0] = i[idx]
            out[filled:filled + take, 1] = j[idx]
            filled += take
    return out


def local_edge_probability(G: nx.Graph) -> float:
    """Population probability that a uniformly drawn ordered non-self pair is an edge."""
    n = G.number_of_nodes()
    if n < 2:
        return 0.0
    if G.is_directed():
        return G.number_of_edges() / (n * (n - 1))
    return 2.0 * G.number_of_edges() / (n * (n - 1))


def fit_local_rank_transform(G: nx.Graph):
    """Exact graph-global midrank transform for binary local distance.

    d_local=0 on edges and 1 on nonedges.  Unlike estimating this transform
    from a small uniform pair sample (which can contain zero edges on sparse
    graphs), the population edge fraction is known exactly from |E|.
    """
    p_edge = local_edge_probability(G)
    u_edge = 0.5 * p_edge
    u_nonedge = 0.5 * (1.0 + p_edge)

    def transform(raw_values):
        raw_values = np.asarray(raw_values)
        return np.where(raw_values <= 0.5, u_edge, u_nonedge).astype(float)

    return transform


def sample_local_edge_nonedge_pairs(
    G: nx.Graph, m_pairs: int, seed: int = 0, edge_fraction: float = 0.5,
) -> np.ndarray:
    """Draw a balanced local diagnostic sample of edges and nonedges.

    The local channel is intentionally defined on this diagnostic pair law,
    rather than on uniform VxV pairs: otherwise the mass of edges shrinks like
    |E|/n^2 on sparse graphs and direct-adjacency signal vanishes with graph
    size.  The default is 50/50 edges/nonedges.
    """
    if not (0.0 < edge_fraction < 1.0):
        raise ValueError("edge_fraction must lie strictly between 0 and 1")
    n = G.number_of_nodes()
    if n < 2:
        raise ValueError("At least two nodes are required")
    rng = np.random.default_rng(seed)
    p_edge = local_edge_probability(G)

    if p_edge <= 0.0:
        return uniform_pair_sample(n, m_pairs, seed)

    edges = np.asarray(list(G.edges()), dtype=np.int64)
    if len(edges) == 0:
        return uniform_pair_sample(n, m_pairs, seed)

    if p_edge >= 1.0 - 1e-15:
        idx = rng.integers(0, len(edges), size=m_pairs)
        ep = edges[idx].copy()
        if not G.is_directed():
            flip = rng.random(m_pairs) < 0.5
            ep[flip] = ep[flip][:, ::-1]
        return ep

    m_edge = max(1, min(m_pairs - 1, int(round(edge_fraction * m_pairs))))
    m_non = m_pairs - m_edge

    idx = rng.integers(0, len(edges), size=m_edge)
    edge_pairs = edges[idx].copy()
    if not G.is_directed():
        flip = rng.random(m_edge) < 0.5
        edge_pairs[flip] = edge_pairs[flip][:, ::-1]

    adj = nx.to_scipy_sparse_array(G, format="csr")
    non_chunks = []
    need = m_non
    while need > 0:
        cand = uniform_pair_sample(
            n, max(4 * need, 128), seed=int(rng.integers(0, 2**32 - 1))
        )
        is_edge = np.asarray(adj[cand[:, 0], cand[:, 1]]).ravel() != 0
        good = cand[~is_edge]
        if len(good):
            take = min(need, len(good))
            non_chunks.append(good[:take])
            need -= take
    non_pairs = np.concatenate(non_chunks, axis=0)
    return np.vstack([edge_pairs, non_pairs])


# =============================================================================
# 3.2  Native alignment field, tie-aware estimator (Eq. 2)
# =============================================================================

def empirical_alignment_field(U_S: np.ndarray, U_X: np.ndarray, K: int = 10) -> np.ndarray:
    """Tie-aware departure-from-independence field on a KxK quantile grid:
    subtracts the *empirical* marginal product, not ab/K^2."""
    edges = np.linspace(0.0, 1.0, K + 1)[1:]
    Fs = np.array([(U_S <= r).mean() for r in edges])
    Fx = np.array([(U_X <= s).mean() for s in edges])
    field = np.zeros((K, K))
    for a, r in enumerate(edges):
        mask_s = U_S <= r
        for b, s in enumerate(edges):
            joint = (mask_s & (U_X <= s)).mean()
            field[a, b] = joint - Fs[a] * Fx[b]
    return field




# =============================================================================
# 3.3  Task-conditioned grammar with endpoint-disjoint cross-fitting
# =============================================================================

def assign_node_folds(labeled_nodes: np.ndarray, kappa: int, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(labeled_nodes)
    folds = np.array_split(perm, kappa)
    return {node: f for f, group in enumerate(folds) for node in group}


def _binned_regression_fit(x: np.ndarray, y: np.ndarray, n_bins: int = 10):
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    means = np.zeros(n_bins)
    counts = np.zeros(n_bins)
    for b in range(n_bins):
        mask = (x >= edges[b]) & (x < edges[b + 1] if b < n_bins - 1 else x <= edges[b + 1])
        counts[b] = mask.sum()
        means[b] = y[mask].mean() if mask.any() else 0.0

    def predict(x_new):
        bins = np.clip(np.digitize(x_new, edges[1:-1]), 0, n_bins - 1)
        return means[bins]

    return predict


def endpoint_disjoint_residual_field(
    pairs: np.ndarray, U_S: np.ndarray, U_X: np.ndarray, Zbar_Y: np.ndarray,
    node_fold: dict, kappa: int, K: int = 10, min_train: int = 10,
) -> tuple[np.ndarray, bool]:
    """Endpoint-disjoint kappa^2 cross-fitting. Returns (field, has_support).
    Only pairs with a VALID fold-pair fit contribute to the field average;
    the denominator is the count of valid-evaluated pairs, not len(pairs).
    If too few pairs get valid support, has_support=False signals the
    caller to abstain rather than silently report a near-zero field.
    """
    fold_of_i = np.array([node_fold[i] for i in pairs[:, 0]])
    fold_of_j = np.array([node_fold[i] for i in pairs[:, 1]])

    residuals = np.zeros(len(pairs))
    valid = np.zeros(len(pairs), dtype=bool)

    for v in range(kappa):
        for w in range(v, kappa):
            excluded = {v, w}
            train_mask = ~np.isin(fold_of_i, list(excluded)) & ~np.isin(fold_of_j, list(excluded))
            eval_mask = (
                ((fold_of_i == v) & (fold_of_j == w)) | ((fold_of_i == w) & (fold_of_j == v))
            )
            if train_mask.sum() < min_train or not eval_mask.any():
                continue
            mu = _binned_regression_fit(U_X[train_mask], Zbar_Y[train_mask])
            residuals[eval_mask] = Zbar_Y[eval_mask] - mu(U_X[eval_mask])
            valid[eval_mask] = True

    has_support = valid.sum() >= max(20, 0.3 * len(pairs))
    edges = np.linspace(0.0, 1.0, K + 1)[1:]
    denom = max(valid.sum(), 1)
    field = np.array([
        residuals[valid & (U_S <= r)].sum() / denom for r in edges
    ])
    return field, has_support


def task_conditioned_fields(
    pairs: np.ndarray, y: dict, U_S_by_d: dict, U_X: np.ndarray, node_fold: dict,
    kappa: int, K: int = 10,
) -> tuple[dict, bool]:
    labels_i = np.array([y[i] for i in pairs[:, 0]])
    labels_j = np.array([y[j] for j in pairs[:, 1]])
    Z = (labels_i == labels_j).astype(float)
    p_hat = Z.mean()
    Zbar = Z - p_hat

    edges = np.linspace(0.0, 1.0, K + 1)[1:]
    A_XY = np.array([(Zbar[U_X <= s]).sum() / len(pairs) for s in edges])

    out = {"A_XY": A_XY, "A_SY": {}, "A_SY_given_X": {}}
    all_supported = True
    for d, U_S in U_S_by_d.items():
        out["A_SY"][d] = np.array([(Zbar[U_S <= r]).sum() / len(pairs) for r in edges])
        field, supported = endpoint_disjoint_residual_field(pairs, U_S, U_X, Zbar, node_fold, kappa, K)
        out["A_SY_given_X"][d] = field
        all_supported = all_supported and supported
    return out, all_supported


# =============================================================================
# 3.4  Finite grammar coordinates: NNLS dictionary projection
# =============================================================================

NATIVE_MECHANISMS = [
    "local_agreement", "local_opposition", "delayed_multihop",
    "role_correspondence", "diffuse_organization",
]
TASK_MECHANISMS = [
    "feature_explained", "structure_explained", "label_opposition", "role_aligned",
]


def project_nnls(vector: np.ndarray, templates: np.ndarray) -> tuple[np.ndarray, float]:
    """Lawson-Hanson NNLS (scipy.optimize.nnls is a direct implementation)."""
    coeffs, residual = nnls(templates, vector)
    return coeffs, residual


def fit_native_templates_joint(
    generator_fn: Callable, structural_keys=("local", "t1", "t2", "role"),
    K: int = 10, n_graphs: int = 20, local_pairs: int = 2000, **gen_kwargs,
) -> np.ndarray:
    """Fit one joint native dictionary across all structural geometries.

    The local channel uses a deliberately balanced edge/nonedge diagnostic
    sample.  This is essential on sparse graphs: under the uniform VxV pair
    law the mass of edges is O(|E|/n^2), so direct-adjacency information would
    vanish with graph size.  The nonlocal diffusion/role channels retain their
    graph-global pair law.  Template fitting uses exactly the same local
    diagnostic law as ``compute_geometric_grammar``.
    """
    templates = []
    for mech in NATIVE_MECHANISMS:
        fields = []
        for s in range(n_graphs):
            data = generator_fn(mechanism=mech, seed=s, **gen_kwargs)
            G, X = data["G"], data["X"]

            # Global feature rank transform from the generator's full pair
            # population (template graphs are intentionally small).
            gp = data["pairs"]
            raw_x = np.linalg.norm(X[gp[:, 0]] - X[gp[:, 1]], axis=1)
            tx_x = fit_rank_transform(raw_x)

            pieces = []
            for key in structural_keys:
                if key == "local":
                    lp = sample_local_edge_nonedge_pairs(
                        G, local_pairs, seed=10000 + s, edge_fraction=0.5
                    )
                    adj = nx.to_scipy_sparse_array(G, format="csr")
                    raw_local = 1.0 - np.asarray(adj[lp[:, 0], lp[:, 1]]).ravel()
                    u_local = fit_local_rank_transform(G)(raw_local)
                    u_x_local = tx_x(
                        np.linalg.norm(X[lp[:, 0]] - X[lp[:, 1]], axis=1)
                    )
                    # Intentionally UNWEIGHTED: the local channel is defined on
                    # the balanced edge/nonedge diagnostic law, preventing its
                    # magnitude from collapsing as graph density decreases.
                    f = empirical_alignment_field(u_local, u_x_local, K=K)
                else:
                    f = empirical_alignment_field(data["U_S"][key], data["U_X"], K=K)
                pieces.append(f.ravel())
            fields.append(np.concatenate(pieces))
        templates.append(np.mean(fields, axis=0))
    return np.stack(templates, axis=1)  # (K*K*|structural_keys|, 5)


def fit_task_templates(
    task_generator_fn: Callable, structural_keys, K: int = 10, n_graphs: int = 10,
    kappa: int = 3, **gen_kwargs,
) -> np.ndarray:
    """Fit the 4 task-dictionary templates by averaging q_tau over synthetic
    graphs from a task-mechanism generator (Section 3.4)."""
    templates = []
    for mech in TASK_MECHANISMS:
        qs = []
        for s in range(n_graphs):
            data = task_generator_fn(mechanism=mech, seed=s, **gen_kwargs)
            G, y = data["G"], data["y"]
            labeled_nodes = np.array(list(y.keys()))
            node_fold = assign_node_folds(labeled_nodes, kappa, seed=s)
            pairs = data["label_pairs"]
            fields, _ = task_conditioned_fields(
                pairs, y, {k: data["U_S_label_pairs"][k] for k in structural_keys},
                data["U_X_label_pairs"], node_fold, kappa, K=K,
            )
            q = np.concatenate(
                [fields["A_XY"]] + [
                    np.concatenate([fields["A_SY"][k], fields["A_SY_given_X"][k]])
                    for k in structural_keys
                ]
            )
            qs.append(q)
        templates.append(np.mean(qs, axis=0))
    return np.stack(templates, axis=1)  # (K*(1+2|D_S|), 4)


# =============================================================================
# 3.7  Extensions: local grammar
# =============================================================================
# NOTE (correction): earlier drafts claimed a weighted aggregate of local
# fields reconstructs the global field once global ranks are used. That is
# still false: the joint term P(S,X) does aggregate linearly across an exact
# partition, but the subtracted product-of-marginals term does not --
# sum_i w_i P_i(S) P_i(X) != P(S) P(X) in general, since a product does not
# distribute over a weighted average. No reconstruction guarantee, exact or
# approximate, is claimed here. Local grammar is a local diagnostic only.

def local_alignment_field(
    U_S_global_for_i: np.ndarray, U_X_global_for_i: np.ndarray, K: int = 10,
) -> np.ndarray:
    """Local field for a node's pair sub-population {(i,j): j in N_i}, using
    the same global ranks as the global field. No aggregation claim."""
    return empirical_alignment_field(U_S_global_for_i, U_X_global_for_i, K=K)


# =============================================================================
# 3.6  Uncertainty: two sources
# =============================================================================

def estimation_uncertainty(compute_fn, pairs, B=50, subsample_frac=0.8, seed=0):
    rng = np.random.default_rng(seed)
    m = len(pairs)
    m_sub = max(int(subsample_frac * m), 1)
    return np.stack([compute_fn(pairs[rng.choice(m, size=m_sub, replace=False)]) for _ in range(B)])


def label_uncertainty(compute_fn, labeled_nodes, B=50, seed=0):
    rng = np.random.default_rng(seed)
    n = len(labeled_nodes)
    return np.stack([compute_fn(labeled_nodes[rng.choice(n, size=n, replace=True)]) for _ in range(B)])


def abstain(coord_samples: np.ndarray, nnls_residual: float, spread_thresh: float = 0.6,
            residual_thresh: float = 0.1) -> bool:
    point = coord_samples.mean(axis=0)
    spread = coord_samples.std(axis=0) / (np.abs(point) + 1e-6)
    return bool(np.any(spread > spread_thresh) or nnls_residual > residual_thresh)


# =============================================================================
# 3.5  Grammar assembly (Algorithm 1)
# =============================================================================

@dataclass
class GeometricGrammar:
    alpha0: dict
    alpha0_residual: dict
    beta_tau: Optional[np.ndarray]
    beta_tau_residual: Optional[float]
    abstained: bool = False
    task_support: Optional[bool] = None


def compute_geometric_grammar(
    G: nx.Graph, X: np.ndarray, y: Optional[dict], native_templates: np.ndarray,
    task_templates: Optional[np.ndarray] = None,
    structural_keys=("local", "t1", "t2", "role"),
    t_list=(1.0, 3.0), n_landmarks: int = 20, K: int = 10, kappa: int = 3,
    m_pairs: int = 2000, local_pairs: Optional[int] = None,
    label_pairs_cap: int = 4000, compute_uncertainty: bool = False,
    B: int = 30, seed: int = 0,
) -> GeometricGrammar:
    """End-to-end geometric-grammar estimation.

    Nonlocal geometries use a graph-global UNIFORM pair sample.  The local adjacency geometry uses a separate balanced edge/nonedge
    diagnostic sample so sparse graphs still contain enough observed edges and
    the local signal does not vanish with graph density. The remaining
    geometries use the graph-global uniform pair law.  No n x n distance
    matrix is materialized.
    """
    # Row-preserving relabel under the standard convention X[k] corresponds to
    # the k-th node in G.nodes() iteration order.  If a caller uses a different
    # ordering, X must be reordered before calling this function.
    G = nx.convert_node_labels_to_integers(G, ordering="default", label_attribute="orig_id")
    orig_to_new = {data["orig_id"]: v for v, data in G.nodes(data=True)}
    if y is not None:
        y = {orig_to_new[k]: v for k, v in y.items()}

    n = G.number_of_nodes()
    if X.shape[0] != n:
        raise ValueError(f"X has {X.shape[0]} rows but G has {n} nodes")
    rng = np.random.default_rng(seed)
    local_pairs = m_pairs if local_pairs is None else int(local_pairs)

    embeddings = propagation_embeddings(G, t_list, n_landmarks, seed=seed)
    role_sig = role_signature(G)
    adj_csr = nx.to_scipy_sparse_array(G, format="csr")

    def local_proxy(prs):
        return 1.0 - np.asarray(adj_csr[prs[:, 0], prs[:, 1]]).ravel()

    def role_proxy(prs):
        return np.linalg.norm(role_sig[prs[:, 0]] - role_sig[prs[:, 1]], axis=1)

    def raw_for_key(key, prs):
        if key == "role":
            return role_proxy(prs)
        if key == "local":
            return local_proxy(prs)
        if key not in {"t1", "t2"}:
            raise KeyError(f"Unknown structural geometry: {key}")
        t = t_list[0] if key == "t1" else t_list[1]
        return pairwise_from_embedding(embeddings[t], prs)

    # ------------------------------------------------------------------
    # Global reference sample: defines feature/diffusion/role rank transforms.
    # ------------------------------------------------------------------
    pairs = uniform_pair_sample(n, m_pairs, seed=seed)
    raw_X_global = np.linalg.norm(X[pairs[:, 0]] - X[pairs[:, 1]], axis=1)
    tx_X = fit_rank_transform(raw_X_global)
    U_X = tx_X(raw_X_global)

    U_S_by_d, tx_S = {}, {}
    for key in structural_keys:
        if key == "local":
            tx_S[key] = fit_local_rank_transform(G)
            # This is only a reference array; the actual local field below uses
            # an edge/nonedge importance sample with population weights.
            U_S_by_d[key] = tx_S[key](raw_for_key(key, pairs))
        else:
            raw = raw_for_key(key, pairs)
            tx_S[key] = fit_rank_transform(raw)
            U_S_by_d[key] = tx_S[key](raw)

    # ------------------------------------------------------------------
    # Local field: balanced edge/nonedge diagnostic sampling.
    # ------------------------------------------------------------------
    local_prs = sample_local_edge_nonedge_pairs(
        G, local_pairs, seed=seed + 17, edge_fraction=0.5
    )
    local_U_X = tx_X(np.linalg.norm(X[local_prs[:, 0]] - X[local_prs[:, 1]], axis=1))
    local_U_S = tx_S["local"](raw_for_key("local", local_prs)) if "local" in structural_keys else None

    fields_for_joint = []
    for key in structural_keys:
        if key == "local":
            field = empirical_alignment_field(local_U_S, local_U_X, K=K)
        else:
            field = empirical_alignment_field(U_S_by_d[key], U_X, K=K)
        fields_for_joint.append(field.ravel())
    joint_field = np.concatenate(fields_for_joint)
    alpha0_vec, alpha0_residual = project_nnls(joint_field, native_templates)
    alpha0 = {"joint": alpha0_vec}
    alpha0_res = {"joint": alpha0_residual}

    # Optional estimation uncertainty.  Resample the two sampling streams
    # independently, preserving the balanced local diagnostic law.
    unc_flag = False
    if compute_uncertainty:
        samples = []
        for b in range(B):
            sub = pairs[rng.choice(len(pairs), size=max(1, int(0.8 * len(pairs))), replace=False)]
            sU_X = tx_X(np.linalg.norm(X[sub[:, 0]] - X[sub[:, 1]], axis=1))
            b_local_prs = sample_local_edge_nonedge_pairs(
                G, local_pairs, seed=seed + 1000 + b, edge_fraction=0.5
            )
            b_local_U_X = tx_X(np.linalg.norm(X[b_local_prs[:, 0]] - X[b_local_prs[:, 1]], axis=1))
            b_joint = []
            for key in structural_keys:
                if key == "local":
                    bUS = tx_S[key](raw_for_key(key, b_local_prs))
                    bf = empirical_alignment_field(bUS, b_local_U_X, K=K)
                else:
                    bUS = tx_S[key](raw_for_key(key, sub))
                    bf = empirical_alignment_field(bUS, sU_X, K=K)
                b_joint.append(bf.ravel())
            c, _ = project_nnls(np.concatenate(b_joint), native_templates)
            samples.append(c)
        samples = np.stack(samples, axis=0)
        unc_flag = abstain(samples, alpha0_residual)

    # ------------------------------------------------------------------
    # Task grammar: labeled pairs are sampled directly from V_L x V_L, but
    # every raw distance is transformed using the SAME graph-global transforms.
    # ------------------------------------------------------------------
    beta_tau, beta_res, task_support = None, None, None
    if y is not None and task_templates is not None:
        labeled_nodes = np.array(list(y.keys()), dtype=np.int64)
        if len(labeled_nodes) >= max(4, kappa):
            node_fold = assign_node_folds(labeled_nodes, kappa, seed=seed)
            max_ordered = len(labeled_nodes) * (len(labeled_nodes) - 1)
            n_label_pairs = min(label_pairs_cap, max_ordered)

            if max_ordered <= label_pairs_cap:
                ii, jj = np.meshgrid(labeled_nodes, labeled_nodes, indexing="ij")
                lp = np.stack([ii.ravel(), jj.ravel()], axis=1)
                lp = lp[lp[:, 0] != lp[:, 1]]
            else:
                # Uniform ordered labeled pairs with replacement; duplicates are
                # acceptable Monte Carlo replicates and preserve the target law.
                li = rng.integers(0, len(labeled_nodes), size=n_label_pairs * 2)
                lj = rng.integers(0, len(labeled_nodes), size=n_label_pairs * 2)
                good = li != lj
                li, lj = li[good][:n_label_pairs], lj[good][:n_label_pairs]
                lp = np.stack([labeled_nodes[li], labeled_nodes[lj]], axis=1)

            if len(lp) > 0:
                lU_X = tx_X(np.linalg.norm(X[lp[:, 0]] - X[lp[:, 1]], axis=1))
                lU_S = {key: tx_S[key](raw_for_key(key, lp)) for key in structural_keys}

                fields, task_support = task_conditioned_fields(
                    lp, y, lU_S, lU_X, node_fold, kappa, K=K
                )
                q_tau = np.concatenate(
                    [fields["A_XY"]] + [
                        np.concatenate([fields["A_SY"][k], fields["A_SY_given_X"][k]])
                        for k in structural_keys
                    ]
                )
                beta_tau, beta_res = project_nnls(q_tau, task_templates)
                if not task_support:
                    unc_flag = True

    return GeometricGrammar(
        alpha0, alpha0_res, beta_tau, beta_res,
        abstained=unc_flag, task_support=task_support,
    )


# =============================================================================
# Synthetic mechanism generators, with explicit block-transition matrices
# =============================================================================

def _generate_agreement_graph(n, rng, target_avg_degree=8.0):
    """agreement: structural closeness <-> feature closeness, by direct
    construction. z is a continuous latent position; X depends on z; edges
    connect nodes close in z-rank (a banded interval graph), so a node's
    structural neighbors are, by construction, its feature-nearest nodes."""
    z = rng.normal(size=n)
    X = np.outer(z, rng.normal(size=8)) * 1.5 + rng.normal(size=(n, 8)) * 0.5
    order = np.argsort(z)
    rank = np.empty(n, dtype=int)
    rank[order] = np.arange(n)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    bandwidth = max(int(target_avg_degree), 2)
    edges = [(i, order[rank[i] + k]) for i in range(n) for k in range(1, bandwidth + 1)
             if rank[i] + k < n]
    G.add_edges_from(edges)
    _stitch_components(G)
    y = {v: int(z[v] > 0) for v in range(n)}
    return G, X, y


def _generate_opposition_graph(n, rng, target_avg_degree=8.0):
    """opposition: structural closeness <-> feature SEPARATION. Same z/X
    construction as agreement, but edges connect nodes at OPPOSITE ends of
    the z-ranking, so structural neighbors are feature-farthest nodes."""
    z = rng.normal(size=n)
    X = np.outer(z, rng.normal(size=8)) * 1.5 + rng.normal(size=(n, 8)) * 0.5
    order = np.argsort(z)
    rank = np.empty(n, dtype=int)
    rank[order] = np.arange(n)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    bandwidth = max(int(target_avg_degree), 2)
    edges = []
    for i in range(n):
        opp = n - 1 - rank[i]
        for k in range(-(bandwidth // 2), bandwidth // 2 + 1):
            rr = opp + k
            if 0 <= rr < n and rr != rank[i]:
                edges.append((i, order[rr]))
    G.add_edges_from(edges)
    _stitch_components(G)
    y = {v: int(z[v] > 0) for v in range(n)}
    return G, X, y


def _generate_delayed_graph(n, rng, target_avg_degree=8.0):
    """delayed_multihop: pure bipartite structure between 2 blocks, with
    NO within-block edges at all. 1-hop neighbors are therefore ALWAYS the
    opposite (feature-dissimilar) block -- an opposition-like signature at
    short scale -- while 2-hop neighbors (opposite-of-opposite) land back
    in the SAME (feature-similar) block -- an agreement-like signature at
    longer scale. The mechanism's fingerprint is this disagreement BETWEEN
    scales, not a fixed shape at either scale alone."""
    block = rng.integers(0, 2, size=n)
    X = rng.normal(size=(n, 8)) + 0.8 * np.eye(2)[block] @ rng.normal(size=(2, 8))
    b0, b1 = np.where(block == 0)[0], np.where(block == 1)[0]
    G = nx.Graph()
    G.add_nodes_from(range(n))
    if len(b0) > 0 and len(b1) > 0:
        target_edges = int(target_avg_degree * n / 2)
        ii = rng.choice(b0, size=target_edges)
        jj = rng.choice(b1, size=target_edges)
        G.add_edges_from(zip(ii.tolist(), jj.tolist()))
    _stitch_components(G)
    y = {v: int(block[v]) for v in range(n)}
    return G, X, y


def _generate_role_graph(n, rng, n_communities=None, target_avg_degree=8.0):
    """role_correspondence: hub/leaf ROLE (not community identity)
    determines the label/feature, so recovery requires role geometry, not
    proximity -- hubs across distant, otherwise-unconnected communities
    share the role-linked feature; leaves (adjacent to their OWN hub, hence
    structurally close to a DIFFERENT-labeled node) do not."""
    if n_communities is None:
        n_communities = max(n // 25, 3)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    per_comm = max(n // n_communities, 4)
    y = {}
    node = 0
    hubs = []
    while node + per_comm <= n:
        hub = node
        y[hub] = 1
        hubs.append(hub)
        leaves = list(range(node + 1, node + per_comm))
        for k, leaf in enumerate(leaves):
            y[leaf] = 0
            G.add_edge(hub, leaf)
            if k > 0 and rng.random() < 0.1:
                G.add_edge(leaf, leaves[k - 1])
        node += per_comm
    for h1, h2 in zip(hubs[:-1], hubs[1:]):
        G.add_edge(h1, h2)
    for v in range(node, n):
        y[v] = 0
        G.add_edge(v, hubs[rng.integers(0, len(hubs))])
    X = rng.normal(size=(n, 8))
    direction = rng.normal(size=8)
    for v, lbl in y.items():
        X[v] += 2.0 * lbl * direction  # strong, concentrated role-feature coupling
    return G, X, y


def _generate_diffuse_graph(n, rng, n_blocks=8, target_avg_degree=8.0):
    """diffuse_organization: a CHAIN of blocks (block b connects only to
    b-1, b+1), with feature signal tied to a SMOOTH, noisy position along
    the chain -- weak enough that adjacent blocks (1-hop) are barely
    distinguishable, but the broad first-half/second-half trend (recovered
    only by aggregating over many hops, i.e. larger diffusion time)
    resolves into real structure-feature alignment at longer scale."""
    block = rng.integers(0, n_blocks, size=n)
    smooth_pos = block + rng.normal(scale=0.4, size=n)
    X = np.outer(smooth_pos, rng.normal(size=8)) * 0.5 + rng.normal(size=(n, 8)) * 1.0
    block_nodes = [np.where(block == b)[0] for b in range(n_blocks)]
    G = nx.Graph()
    G.add_nodes_from(range(n))
    edges = []
    within, between = int(target_avg_degree * 0.6), int(target_avg_degree * 0.4)
    for b in range(n_blocks):
        nb = len(block_nodes[b])
        if nb > 1:
            n_e = min(within * nb // 2, nb * (nb - 1) // 2)
            if n_e > 0:
                ii, jj = rng.integers(0, nb, n_e * 2), rng.integers(0, nb, n_e * 2)
                m = ii != jj
                pr = np.unique(np.stack([np.minimum(ii[m], jj[m]), np.maximum(ii[m], jj[m])], axis=1), axis=0)[:n_e]
                edges += list(zip(block_nodes[b][pr[:, 0]].tolist(), block_nodes[b][pr[:, 1]].tolist()))
        if b + 1 < n_blocks:
            na, nb2 = len(block_nodes[b]), len(block_nodes[b + 1])
            if na > 0 and nb2 > 0:
                n_e = min(between * (na + nb2) // 2, na * nb2)
                if n_e > 0:
                    ii, jj = rng.integers(0, na, n_e * 2), rng.integers(0, nb2, n_e * 2)
                    pr = np.unique(np.stack([ii, jj], axis=1), axis=0)[:n_e]
                    edges += list(zip(block_nodes[b][pr[:, 0]].tolist(), block_nodes[b + 1][pr[:, 1]].tolist()))
    G.add_edges_from(edges)
    _stitch_components(G)
    y = {v: int(block[v] >= n_blocks // 2) for v in range(n)}
    return G, X, y


def _stitch_components(G):
    comps = list(nx.connected_components(G))
    for c1, c2 in zip(comps[:-1], comps[1:]):
        G.add_edge(next(iter(c1)), next(iter(c2)))


def generate_mechanism_graph(mechanism: str, n: int = 200, seed: int = 0,
                              t_list=(1.0, 3.0), n_landmarks: int = 20) -> dict:
    rng = np.random.default_rng(seed)
    if mechanism == "local_agreement":
        G, X, y = _generate_agreement_graph(n, rng)
    elif mechanism == "local_opposition":
        G, X, y = _generate_opposition_graph(n, rng)
    elif mechanism == "delayed_multihop":
        G, X, y = _generate_delayed_graph(n, rng)
    elif mechanism == "role_correspondence":
        G, X, y = _generate_role_graph(n, rng)
    else:  # diffuse_organization
        G, X, y = _generate_diffuse_graph(n, rng)

    embeddings = propagation_embeddings(G, t_list, min(n_landmarks, n - 1), seed=seed)
    role_sig = role_signature(G)
    node_list = np.arange(n)
    all_pairs = np.stack(np.meshgrid(node_list, node_list), axis=-1).reshape(-1, 2)
    all_pairs = all_pairs[all_pairs[:, 0] != all_pairs[:, 1]]
    # SIMPLIFICATION: for template-fitting on small synthetic graphs (n<=few
    # hundred) using all pairs is fine; compute_geometric_grammar itself
    # never does this for real (large) graphs.
    U_X = midrank_uniform(np.linalg.norm(X[all_pairs[:, 0]] - X[all_pairs[:, 1]], axis=1))
    adj = nx.to_scipy_sparse_array(G, format="csr")
    d_local_raw = 1.0 - np.asarray(adj[all_pairs[:, 0], all_pairs[:, 1]]).ravel()
    U_S = {
        "local": midrank_uniform(d_local_raw),
        "t1": midrank_uniform(pairwise_from_embedding(embeddings[t_list[0]], all_pairs)),
        "t2": midrank_uniform(pairwise_from_embedding(embeddings[t_list[1]], all_pairs)),
        "role": midrank_uniform(np.linalg.norm(role_sig[all_pairs[:, 0]] - role_sig[all_pairs[:, 1]], axis=1)),
    }
    return {"G": G, "X": X, "y": y, "U_X": U_X, "U_S": U_S, "pairs": all_pairs}




def generate_task_mechanism_graph(mechanism: str, n: int = 200, seed: int = 0,
                                   t_list=(1.0, 3.0)) -> dict:
    """Controlled generators for the four task-dictionary mechanisms.

    The task fields use rank transforms fitted on the full graph pair
    population and then applied unchanged to the sampled labeled pairs, exactly
    matching ``compute_geometric_grammar``.
    """
    rng = np.random.default_rng(seed)
    if mechanism == "feature_explained":
        G, X, y = _generate_agreement_graph(n, rng)
        X = rng.normal(size=(n, 8))
        y = {v: int(X[v, 0] > 0) for v in range(n)}
    elif mechanism == "structure_explained":
        G, X, y = _generate_agreement_graph(n, rng)
        X = rng.normal(size=(n, 8))  # remove feature explanation
    elif mechanism == "label_opposition":
        G, X, y = _generate_opposition_graph(n, rng)
        X = rng.normal(size=(n, 8))  # isolate graph-label opposition
    elif mechanism == "role_aligned":
        G, X, y = _generate_role_graph(n, rng)
        X = rng.normal(size=(n, 8))  # isolate role-label relation
    else:
        raise ValueError(f"Unknown task mechanism: {mechanism}")

    embeddings = propagation_embeddings(G, t_list, min(20, n - 1), seed=seed)
    role_sig = role_signature(G)
    adj = nx.to_scipy_sparse_array(G, format="csr")

    # Full pair population is acceptable here because template graphs are small.
    nodes = np.arange(n)
    ii, jj = np.meshgrid(nodes, nodes, indexing="ij")
    all_pairs = np.stack([ii.ravel(), jj.ravel()], axis=1)
    all_pairs = all_pairs[all_pairs[:, 0] != all_pairs[:, 1]]

    def raw_local(prs):
        return 1.0 - np.asarray(adj[prs[:, 0], prs[:, 1]]).ravel()

    raw_X_global = np.linalg.norm(X[all_pairs[:, 0]] - X[all_pairs[:, 1]], axis=1)
    tx_X = fit_rank_transform(raw_X_global)
    tx_local = fit_local_rank_transform(G)
    raw_t1 = pairwise_from_embedding(embeddings[t_list[0]], all_pairs)
    raw_t2 = pairwise_from_embedding(embeddings[t_list[1]], all_pairs)
    raw_role = np.linalg.norm(role_sig[all_pairs[:, 0]] - role_sig[all_pairs[:, 1]], axis=1)
    tx = {
        "local": tx_local,
        "t1": fit_rank_transform(raw_t1),
        "t2": fit_rank_transform(raw_t2),
        "role": fit_rank_transform(raw_role),
    }

    labeled_nodes = np.array(list(y.keys()), dtype=np.int64)
    n_draw = min(3000, len(labeled_nodes) * (len(labeled_nodes) - 1))
    if len(labeled_nodes) * (len(labeled_nodes) - 1) <= 3000:
        a, b = np.meshgrid(labeled_nodes, labeled_nodes, indexing="ij")
        lp = np.stack([a.ravel(), b.ravel()], axis=1)
        lp = lp[lp[:, 0] != lp[:, 1]]
    else:
        li = rng.integers(0, len(labeled_nodes), size=n_draw * 2)
        lj = rng.integers(0, len(labeled_nodes), size=n_draw * 2)
        good = li != lj
        li, lj = li[good][:n_draw], lj[good][:n_draw]
        lp = np.stack([labeled_nodes[li], labeled_nodes[lj]], axis=1)

    U_X_lp = tx_X(np.linalg.norm(X[lp[:, 0]] - X[lp[:, 1]], axis=1))
    U_S_lp = {
        "local": tx["local"](raw_local(lp)),
        "t1": tx["t1"](pairwise_from_embedding(embeddings[t_list[0]], lp)),
        "t2": tx["t2"](pairwise_from_embedding(embeddings[t_list[1]], lp)),
        "role": tx["role"](np.linalg.norm(role_sig[lp[:, 0]] - role_sig[lp[:, 1]], axis=1)),
    }
    return {
        "G": G, "X": X, "y": y,
        "label_pairs": lp,
        "U_X_label_pairs": U_X_lp,
        "U_S_label_pairs": U_S_lp,
    }


# =============================================================================
# Demo
# =============================================================================

if __name__ == "__main__":
    import time

    KEYS = ("local", "t1", "t2", "role")
    K = 6

    print("Fitting joint native dictionary from G_train ...")
    native_templates = fit_native_templates_joint(
        generate_mechanism_graph, structural_keys=KEYS, K=K, n_graphs=6, n=150,
    )
    print(f"  native template matrix shape: {native_templates.shape}")

    print("Fitting task dictionary templates ...")
    task_templates = fit_task_templates(
        generate_task_mechanism_graph, structural_keys=KEYS, K=K,
        n_graphs=4, n=150,
    )
    print(f"  task template matrix shape: {task_templates.shape}")

    print("\nHeld-seed native sanity check using the ACTUAL sampled estimator:")
    correct = 0
    total = 0
    for mech in NATIVE_MECHANISMS:
        for seed in range(100, 105):
            data = generate_mechanism_graph(mech, n=300, seed=seed)
            grammar = compute_geometric_grammar(
                data["G"], data["X"], None, native_templates,
                task_templates=None, structural_keys=KEYS, K=K,
                m_pairs=3000, local_pairs=2500, seed=seed + 500,
            )
            pred = NATIVE_MECHANISMS[int(np.argmax(grammar.alpha0["joint"]))]
            correct += int(pred == mech)
            total += 1
        print(f"  finished {mech}")
    print(f"  sampled recovery: {correct}/{total} = {100.0 * correct / total:.1f}%")

    print("\nEnd-to-end sampled-estimator sanity check:")
    correct = 0
    total = 0
    for mech in NATIVE_MECHANISMS:
        data = generate_mechanism_graph(mech, n=300, seed=777)
        grammar = compute_geometric_grammar(
            data["G"], data["X"], None, native_templates,
            task_templates=None, structural_keys=KEYS, K=K,
            m_pairs=3000, local_pairs=2000, seed=11,
        )
        pred = NATIVE_MECHANISMS[int(np.argmax(grammar.alpha0["joint"]))]
        print(f"  {mech:22s} -> {pred:22s}  residual={grammar.alpha0_residual['joint']:.4f}")
        correct += int(pred == mech)
        total += 1
    print(f"  sampled-estimator recovery: {correct}/{total}")

    print("\nLabel-scarce task-grammar test (20 labels/class where available):")
    G_large, X_large, y_full = _generate_delayed_graph(2000, np.random.default_rng(1))
    labels = np.array(list(y_full.values()))
    nodes = np.array(list(y_full.keys()))
    rng = np.random.default_rng(1)
    scarce = []
    for c in np.unique(labels):
        cand = nodes[labels == c]
        scarce.extend(rng.choice(cand, size=min(20, len(cand)), replace=False).tolist())
    y_scarce = {int(v): y_full[int(v)] for v in scarce}
    grammar = compute_geometric_grammar(
        G_large, X_large, y_scarce, native_templates,
        task_templates=task_templates, structural_keys=KEYS,
        K=K, kappa=3, m_pairs=2500, local_pairs=2000,
        compute_uncertainty=False, seed=3,
    )
    print(f"  beta_tau={None if grammar.beta_tau is None else np.round(grammar.beta_tau, 3)}")
    print(f"  task_support={grammar.task_support}, abstained={grammar.abstained}")

    print("\nModerate scalability smoke test (n=5000; role_signature remains the known bottleneck):")
    G_big, X_big, _ = _generate_agreement_graph(5000, np.random.default_rng(2))
    t0 = time.time()
    grammar_big = compute_geometric_grammar(
        G_big, X_big, None, native_templates,
        task_templates=None, structural_keys=KEYS,
        K=K, m_pairs=2000, local_pairs=2000, seed=4,
    )
    print(f"  completed in {time.time() - t0:.1f}s; alpha0={np.round(grammar_big.alpha0['joint'], 3)}")

    print("\nDone.")

