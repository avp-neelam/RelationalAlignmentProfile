"""
Experiments for Geometric Grammar (E1--E4)
===========================================

Reference implementation matching the paper's experiment section:

E1  Matched-summary diagnosis / counterfactual tasks.
E2  Cross-generator A->B generalization + held-out mechanism mixtures.
E3  Frozen synthetic->real architecture-selection transfer.
E4  Representation and low-label ablations.

This file intentionally delegates geometric-grammar estimation primitives to
    ``utils.grammar.core``.  The runner is kept separate from the numerical
    grammar implementation so each part can be tested independently.

Important experimental safeguards implemented here
---------------------------------------------------
* No task-label leakage: beta^(tau), label-aware baselines, and all other task
  statistics see TRAIN labels only.  Validation/test labels are never passed to
  the problem representation.
* G_train and G_test are structurally distinct generator families.
* The same six architecture implementations are used on synthetic and real
  graphs by default, with fixed PER-ARCHITECTURE hyperparameters across all
  problems. The sixth candidate is a GPS-style local-global Graph Transformer
  with Performer attention. Use --no-gt to reproduce the original five-model bank.
* Architecture performance is averaged over multiple training seeds before it
  becomes the target for the grammar->performance predictor.
* Grammar, conventional statistics, raw fields, PCA, and signed-NMF are all fed
  to the same standardized Ridge meta-predictor.
* Real graphs are symmetrized once and that SAME graph is used by both grammar
  and the architecture bank.
* Expensive architecture training is cached to disk.

Dependencies
------------
numpy scipy scikit-learn networkx torch torch-geometric
Optional for ogbn-arxiv: ogb

Example
-------
python -m utils.experiments.runner --exp e2 --out results/e2
python -m utils.experiments.runner --exp e3 --out results/e3 --e2-state results/e2/e2_state.pkl
python -m utils.experiments.runner --exp all --out results/full_6arch
python -m utils.experiments.runner --exp all --no-gt --out results/full_5arch

The default configuration is intended for final runs and can be expensive.
Use ``--quick`` only as a pipeline smoke test; never report quick-mode numbers.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import pickle
import random
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse
from scipy.sparse.linalg import eigsh
from scipy.stats import spearmanr, wilcoxon
from sklearn.decomposition import NMF, PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils.grammar import core as gg

try:
    from torch_geometric.data import Data
    from torch_geometric.nn import APPNP as APPNPProp
    from torch_geometric.nn import FAConv, GCNConv, SAGEConv
    from torch_geometric.utils import remove_self_loops, to_undirected
    PYG_AVAILABLE = True
    try:
        from torch_geometric.nn import GPSConv
    except Exception:  # Older PyG can still run the five-model --no-gt bank.
        GPSConv = None
except Exception:  # pragma: no cover - allows static import without PyG
    Data = Any
    APPNPProp = FAConv = GCNConv = SAGEConv = GPSConv = None
    remove_self_loops = to_undirected = None
    PYG_AVAILABLE = False


# -----------------------------------------------------------------------------
# Global conventions
# -----------------------------------------------------------------------------

STRUCTURAL_KEYS: Tuple[str, ...] = ("local", "t1", "t2", "role")
NATIVE_MECHANISMS = tuple(gg.NATIVE_MECHANISMS)
TASK_MECHANISMS = tuple(gg.TASK_MECHANISMS)
BASE_ARCHITECTURE_NAMES: Tuple[str, ...] = (
    "feature_only",
    "local_lowpass",
    "high_pass",
    "multihop",
    "role_structural",
)
GT_ARCHITECTURE_NAME = "graph_transformer"
ARCHITECTURE_NAMES: Tuple[str, ...] = BASE_ARCHITECTURE_NAMES + (GT_ARCHITECTURE_NAME,)


def active_architectures(cfg: "ExperimentConfig") -> Tuple[str, ...]:
    """Candidate architecture bank for this run.

    The six-model bank is the primary setting. ``include_gt=False`` reproduces
    the original five-model bank on exactly the same problem corpus.
    """
    return ARCHITECTURE_NAMES if cfg.include_gt else BASE_ARCHITECTURE_NAMES
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _require_pyg() -> None:
    if not PYG_AVAILABLE:
        raise ImportError(
            "torch_geometric is required for architecture training / real-data "
            "experiments. Install a PyTorch-Geometric build compatible with your "
            "PyTorch version."
        )


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class ArchConfig:
    hidden: int = 64
    lr: float = 0.01
    weight_decay: float = 5e-4
    epochs: int = 300
    patience: int = 50
    dropout: float = 0.5
    # architecture-specific values; ignored when irrelevant
    hops: int = 10
    alpha: float = 0.1
    layers: int = 2
    heads: int = 4
    attn_type: str = "performer"


DEFAULT_ARCH_CONFIG: Dict[str, ArchConfig] = {
    "feature_only": ArchConfig(hidden=64, lr=0.01, weight_decay=5e-4),
    "local_lowpass": ArchConfig(hidden=64, lr=0.01, weight_decay=5e-4),
    "high_pass": ArchConfig(hidden=64, lr=0.01, weight_decay=5e-4, layers=2),
    "multihop": ArchConfig(hidden=64, lr=0.01, weight_decay=5e-4, hops=10, alpha=0.1),
    "role_structural": ArchConfig(hidden=64, lr=0.005, weight_decay=5e-4),
    # GPS-style local-global Graph Transformer. Performer attention is linear
    # in the number of nodes and avoids the O(n^2) full-attention bottleneck.
    # No extra handcrafted PE/SE is injected here, keeping the input information
    # budget matched to the other graph models.
    "graph_transformer": ArchConfig(
        hidden=64, lr=0.003, weight_decay=1e-4, dropout=0.3,
        layers=2, heads=4, attn_type="performer",
    ),
}


@dataclass
class ExperimentConfig:
    # Grammar
    K: int = 6
    t_list: Tuple[float, float] = (1.0, 3.0)
    n_landmarks: int = 20
    m_pairs: int = 3000
    local_pairs: int = 2500
    label_pairs_cap: int = 5000
    kappa: int = 3

    # Template fitting / synthetic corpus
    template_graphs_per_mechanism: int = 20
    task_template_graphs_per_mechanism: int = 12
    template_n: int = 220
    synthetic_n: int = 300
    train_problems_per_mechanism: int = 30
    test_problems_per_mechanism: int = 15
    mixture_problems: int = 30

    # Architecture targets
    model_seeds: Tuple[int, ...] = (0, 1, 2, 3, 4)
    meta_ridge_alpha: float = 1.0

    # E1
    e1_triples: int = 30
    e1_n: int = 240
    e1_target_adjusted_homophily: float = 0.0
    e1_homophily_tolerance: float = 0.08
    e1_anneal_steps: int = 3500

    # E4
    low_labels_per_class: int = 20

    # Evaluation
    real_split_seed: int = 0
    random_baseline_draws: int = 500

    # Files / caching
    data_root: str = "./data"
    cache_dir: str = "./cache"

    # Candidate bank: six architectures by default; set False for the
    # pre-specified five-architecture robustness analysis.
    include_gt: bool = True

    # Per-architecture fixed configs
    arch: Dict[str, ArchConfig] = field(
        default_factory=lambda: copy.deepcopy(DEFAULT_ARCH_CONFIG)
    )

    @classmethod
    def quick(cls) -> "ExperimentConfig":
        c = cls()
        c.template_graphs_per_mechanism = 3
        c.task_template_graphs_per_mechanism = 2
        c.template_n = 100
        c.synthetic_n = 120
        c.train_problems_per_mechanism = 2
        c.test_problems_per_mechanism = 1
        c.mixture_problems = 3
        c.model_seeds = (0,)
        c.m_pairs = 800
        c.local_pairs = 600
        c.label_pairs_cap = 1000
        c.e1_triples = 4
        c.e1_n = 120
        c.e1_anneal_steps = 500
        for a in c.arch.values():
            a.epochs = 25
            a.patience = 6
            a.hidden = 32
        return c


# =============================================================================
# Problem container and graph preprocessing
# =============================================================================

@dataclass
class Problem:
    name: str
    G: nx.Graph
    X: np.ndarray
    y: np.ndarray
    train_mask: np.ndarray
    val_mask: np.ndarray
    test_mask: np.ndarray
    metric: str = "accuracy"  # accuracy | roc_auc

    def y_dict(self, mask: Optional[np.ndarray] = None) -> Dict[int, int]:
        if mask is None:
            idx = np.arange(len(self.y))
        else:
            idx = np.flatnonzero(mask)
        return {int(i): int(self.y[i]) for i in idx}

    @property
    def n_classes(self) -> int:
        return int(np.max(self.y)) + 1


def stratified_masks(y: np.ndarray, seed: int, train_frac: float = 0.6,
                     val_frac: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Class-stratified random split, 60/20/20 by default.

    This is the repo-wide default split for every real dataset EXCEPT
    ogbn-arxiv (see load_real_problem): even datasets that ship their own
    masks (Planetoid's 20-per-class semi-supervised "public" split,
    HeterophilousGraphDataset's 10 predefined 50/25/25 folds) are re-split
    with this function rather than using what they shipped with.
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    train = np.zeros(n, dtype=bool)
    val = np.zeros(n, dtype=bool)
    test = np.zeros(n, dtype=bool)
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        idx = rng.permutation(idx)
        nt = max(1, int(round(train_frac * len(idx))))
        nv = max(1, int(round(val_frac * len(idx)))) if len(idx) >= 5 else 0
        if nt + nv >= len(idx):
            nt = max(1, len(idx) - 2)
            nv = 1 if len(idx) - nt >= 2 else 0
        train[idx[:nt]] = True
        val[idx[nt:nt + nv]] = True
        test[idx[nt + nv:]] = True
    return train, val, test


def same_split_masks(n: int, seed: int, train_frac: float = 0.6,
                     val_frac: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unstratified but identical node split for E1 counterfactual tasks."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    ntr = int(train_frac * n)
    nva = int(val_frac * n)
    tr = np.zeros(n, dtype=bool)
    va = np.zeros(n, dtype=bool)
    te = np.zeros(n, dtype=bool)
    tr[perm[:ntr]] = True
    va[perm[ntr:ntr+nva]] = True
    te[perm[ntr+nva:]] = True
    return tr, va, te


def nx_graph_from_edge_index(edge_index: torch.Tensor, n: int) -> nx.Graph:
    ei = edge_index.detach().cpu().numpy()
    G = nx.Graph()
    G.add_nodes_from(range(n))
    if ei.size:
        G.add_edges_from(zip(ei[0].tolist(), ei[1].tolist()))
    G.remove_edges_from(nx.selfloop_edges(G))
    return G


def sanitize_edge_index(edge_index: torch.Tensor, n: int) -> torch.Tensor:
    _require_pyg()
    edge_index, _ = remove_self_loops(edge_index)
    edge_index = to_undirected(edge_index, num_nodes=n)
    return edge_index


def problem_to_pyg(problem: Problem, role_x: Optional[np.ndarray] = None) -> Data:
    _require_pyg()
    edges = np.asarray(list(problem.G.edges()), dtype=np.int64)
    if len(edges) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edges.T, dtype=torch.long)
        edge_index = to_undirected(edge_index, num_nodes=len(problem.y))
    d = Data(
        x=torch.tensor(problem.X, dtype=torch.float32),
        edge_index=edge_index,
        y=torch.tensor(problem.y, dtype=torch.long),
        train_mask=torch.tensor(problem.train_mask, dtype=torch.bool),
        val_mask=torch.tensor(problem.val_mask, dtype=torch.bool),
        test_mask=torch.tensor(problem.test_mask, dtype=torch.bool),
    )
    if role_x is not None:
        d.role_x = torch.tensor(role_x, dtype=torch.float32)
    return d


# =============================================================================
# G_train: SBM-based planted mechanism family
# =============================================================================


def _stitch(G: nx.Graph) -> None:
    comps = list(nx.connected_components(G))
    for a, b in zip(comps[:-1], comps[1:]):
        G.add_edge(next(iter(a)), next(iter(b)))


def _sbm_from_assignments(assign: np.ndarray, P: np.ndarray, rng: np.random.Generator) -> nx.Graph:
    n = len(assign)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    blocks = [np.flatnonzero(assign == b) for b in range(P.shape[0])]
    for a in range(P.shape[0]):
        aa = blocks[a]
        if len(aa) > 1 and P[a, a] > 0:
            for ii in range(len(aa)):
                # n is only a few hundred in controlled experiments; this clear
                # implementation is preferable to a generator-dependent shortcut.
                js = aa[ii + 1:]
                if len(js):
                    keep = rng.random(len(js)) < P[a, a]
                    G.add_edges_from((int(aa[ii]), int(v)) for v in js[keep])
        for b in range(a + 1, P.shape[0]):
            bb = blocks[b]
            if not len(aa) or not len(bb) or P[a, b] <= 0:
                continue
            for u in aa:
                keep = rng.random(len(bb)) < P[a, b]
                G.add_edges_from((int(u), int(v)) for v in bb[keep])
    _stitch(G)
    return G


def generate_train_problem(mechanism: str, n: int, seed: int) -> Tuple[nx.Graph, np.ndarray, Dict[int, int]]:
    """SBM-based G_train. All mechanisms are binary node-classification tasks."""
    rng = np.random.default_rng(seed)
    d = 8

    if mechanism == "local_agreement":
        z = rng.integers(0, 2, n)
        P = np.array([[0.09, 0.008], [0.008, 0.09]])
        G = _sbm_from_assignments(z, P, rng)
        direction = rng.normal(size=d)
        X = rng.normal(scale=1.0, size=(n, d)) + 0.8 * z[:, None] * direction
        y = z

    elif mechanism == "local_opposition":
        z = rng.integers(0, 2, n)
        P = np.array([[0.006, 0.095], [0.095, 0.006]])
        G = _sbm_from_assignments(z, P, rng)
        direction = rng.normal(size=d)
        X = rng.normal(scale=1.0, size=(n, d)) + 0.8 * z[:, None] * direction
        y = z

    elif mechanism == "delayed_multihop":
        # Four-block cycle. Parity is the task: one hop crosses parity, two hops
        # return to the same parity through a different block.
        block = rng.integers(0, 4, n)
        P = np.full((4, 4), 0.002)
        for b in range(4):
            P[b, (b + 1) % 4] = 0.075
            P[(b + 1) % 4, b] = 0.075
        np.fill_diagonal(P, 0.003)
        G = _sbm_from_assignments(block, P, rng)
        y = block % 2
        direction = rng.normal(size=d)
        X = rng.normal(scale=1.0, size=(n, d)) + 0.7 * y[:, None] * direction

    elif mechanism == "role_correspondence":
        # community x role blocks: role-1 nodes are core-like, role-0 nodes are
        # peripheral, while communities remain separated. Roles are balanced.
        community = rng.integers(0, 3, n)
        role = rng.integers(0, 2, n)
        block = 2 * community + role
        B = 6
        P = np.full((B, B), 0.001)
        for c in range(3):
            p0, p1 = 2*c, 2*c + 1
            P[p1, p1] = 0.11
            P[p0, p1] = P[p1, p0] = 0.075
            P[p0, p0] = 0.008
        G = _sbm_from_assignments(block, P, rng)
        y = role
        direction = rng.normal(size=d)
        X = rng.normal(scale=1.0, size=(n, d)) + 0.8 * role[:, None] * direction

    elif mechanism == "diffuse_organization":
        B = 8
        block = rng.integers(0, B, n)
        P = np.full((B, B), 0.0005)
        for b in range(B):
            P[b, b] = 0.028
            if b + 1 < B:
                P[b, b + 1] = P[b + 1, b] = 0.055
        G = _sbm_from_assignments(block, P, rng)
        pos = (block + rng.normal(scale=0.35, size=n)) / (B - 1)
        direction = rng.normal(size=d)
        X = rng.normal(scale=1.0, size=(n, d)) + 0.85 * pos[:, None] * direction
        y = (block >= B // 2).astype(int)

    else:
        raise ValueError(f"Unknown mechanism: {mechanism}")

    return G, X.astype(np.float32), {int(i): int(y[i]) for i in range(n)}


# =============================================================================
# G_test: structurally distinct geometric / motif family
# =============================================================================


def _knn_edges(pos: np.ndarray, k: int, allowed: Optional[np.ndarray] = None) -> List[Tuple[int, int]]:
    """Naive kNN helper for controlled n~few hundred graphs."""
    n = len(pos)
    edges: set[Tuple[int, int]] = set()
    for i in range(n):
        dist = np.linalg.norm(pos - pos[i], axis=1)
        dist[i] = np.inf
        order = np.argsort(dist)
        count = 0
        for j in order:
            if allowed is not None and not allowed[i, j]:
                continue
            a, b = sorted((i, int(j)))
            edges.add((a, b))
            count += 1
            if count >= k:
                break
    return list(edges)


def generate_test_problem(mechanism: str, n: int, seed: int) -> Tuple[nx.Graph, np.ndarray, Dict[int, int]]:
    """G_test: geometric / motif constructions sharing mechanisms but not topology code."""
    rng = np.random.default_rng(seed)
    d = 8
    G = nx.Graph()
    G.add_nodes_from(range(n))

    if mechanism == "local_agreement":
        pos = rng.uniform(0, 1, size=(n, 2))
        y = (pos[:, 0] > 0.5).astype(int)
        G.add_edges_from(_knn_edges(pos, k=7))
        direction = rng.normal(size=d)
        X = rng.normal(size=(n, d)) + 0.9 * pos[:, [0]] * direction

    elif mechanism == "local_opposition":
        pos = rng.uniform(0, 1, size=(n, 2))
        y = rng.integers(0, 2, size=n)
        allowed = y[:, None] != y[None, :]
        G.add_edges_from(_knn_edges(pos, k=7, allowed=allowed))
        direction = rng.normal(size=d)
        X = rng.normal(size=(n, d)) + 0.9 * y[:, None] * direction

    elif mechanism == "delayed_multihop":
        # Nodes around a ring, labels alternate in short arcs; edges connect
        # nearest opposite-label nodes, so two-hop walks recover same label.
        theta = np.sort(rng.uniform(0, 2*np.pi, size=n))
        pos = np.c_[np.cos(theta), np.sin(theta)]
        y = (np.arange(n) % 2).astype(int)
        # permute node ids so label isn't an index artifact
        perm = rng.permutation(n)
        pos = pos[perm]
        y = y[perm]
        allowed = y[:, None] != y[None, :]
        G.add_edges_from(_knn_edges(pos, k=6, allowed=allowed))
        direction = rng.normal(size=d)
        X = rng.normal(size=(n, d)) + 0.75 * y[:, None] * direction

    elif mechanism == "role_correspondence":
        # Repeated P4 motifs: two endpoints (role 0), two interiors (role 1),
        # exactly balanced within each motif. Sparse bridges join motifs.
        y = np.zeros(n, dtype=int)
        motif_nodes = []
        node = 0
        while node + 4 <= n:
            a, b, c, d0 = node, node+1, node+2, node+3
            G.add_edges_from([(a, b), (b, c), (c, d0)])
            y[[a, d0]] = 0
            y[[b, c]] = 1
            motif_nodes.append((a, b, c, d0))
            node += 4
        for v in range(node, n):
            G.add_edge(v, rng.integers(0, max(node, 1)))
            y[v] = int(G.degree(v) > 1)
        for m1, m2 in zip(motif_nodes[:-1], motif_nodes[1:]):
            G.add_edge(m1[-1], m2[0])
        direction = rng.normal(size=d)
        X = rng.normal(size=(n, d)) + 0.85 * y[:, None] * direction

    elif mechanism == "diffuse_organization":
        # Random geometric graph with a smooth low-frequency coordinate.
        pos = rng.uniform(0, 1, size=(n, 2))
        G.add_edges_from(_knn_edges(pos, k=7))
        y = (pos[:, 0] > 0.5).astype(int)
        direction = rng.normal(size=d)
        smooth = 0.5 * pos[:, 0] + 0.5 * pos[:, 1]
        X = rng.normal(scale=1.0, size=(n, d)) + 0.9 * smooth[:, None] * direction

    else:
        raise ValueError(f"Unknown mechanism: {mechanism}")

    _stitch(G)
    return G, X.astype(np.float32), {int(i): int(y[i]) for i in range(n)}


def generate_mixture_problem(n: int, seed: int,
                             mechanisms: Optional[Tuple[str, str]] = None) -> Tuple[nx.Graph, np.ndarray, Dict[int, int], Tuple[str, str]]:
    """Two-region held-out mixture, never used as a labeled meta-training condition."""
    rng = np.random.default_rng(seed)
    if mechanisms is None:
        mechanisms = tuple(rng.choice(NATIVE_MECHANISMS, size=2, replace=False).tolist())  # type: ignore
    n1 = n // 2
    n2 = n - n1
    G1, X1, y1 = generate_test_problem(mechanisms[0], n1, seed + 11)
    G2, X2, y2 = generate_test_problem(mechanisms[1], n2, seed + 29)
    G = nx.disjoint_union(G1, G2)
    for _ in range(3):
        G.add_edge(int(rng.integers(0, n1)), int(rng.integers(n1, n)))
    X = np.vstack([X1, X2])
    y = {**y1, **{n1 + k: v for k, v in y2.items()}}
    return G, X, y, mechanisms


# =============================================================================
# Generator records used ONLY to fit grammar dictionaries on G_train
# =============================================================================


def _all_ordered_pairs(n: int) -> np.ndarray:
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    p = np.stack([ii.ravel(), jj.ravel()], axis=1)
    return p[p[:, 0] != p[:, 1]]


def _native_generator_record(mechanism: str, seed: int, n: int = 220,
                             t_list: Tuple[float, float] = (1.0, 3.0),
                             n_landmarks: int = 20) -> Dict[str, Any]:
    G, X, y = generate_train_problem(mechanism, n=n, seed=seed)
    pairs = _all_ordered_pairs(n)
    emb = gg.propagation_embeddings(G, t_list, min(n_landmarks, n-1), seed=seed)
    role = gg.role_signature(G)
    adj = nx.to_scipy_sparse_array(G, format="csr")
    U_X = gg.midrank_uniform(np.linalg.norm(X[pairs[:, 0]] - X[pairs[:, 1]], axis=1))
    U_S = {
        "local": gg.midrank_uniform(1.0 - np.asarray(adj[pairs[:, 0], pairs[:, 1]]).ravel()),
        "t1": gg.midrank_uniform(gg.pairwise_from_embedding(emb[t_list[0]], pairs)),
        "t2": gg.midrank_uniform(gg.pairwise_from_embedding(emb[t_list[1]], pairs)),
        "role": gg.midrank_uniform(np.linalg.norm(role[pairs[:, 0]] - role[pairs[:, 1]], axis=1)),
    }
    return {"G": G, "X": X, "y": y, "pairs": pairs, "U_X": U_X, "U_S": U_S}


def _task_generator_record(mechanism: str, seed: int, n: int = 220,
                           t_list: Tuple[float, float] = (1.0, 3.0),
                           n_landmarks: int = 20) -> Dict[str, Any]:
    """Task-template problems drawn from the same SBM-based G_train family."""
    rng = np.random.default_rng(seed)
    if mechanism == "feature_explained":
        G, _, _ = generate_train_problem("diffuse_organization", n, seed)
        X = rng.normal(size=(n, 8)).astype(np.float32)
        y = {i: int(X[i, 0] > 0) for i in range(n)}
    elif mechanism == "structure_explained":
        G, _, y = generate_train_problem("local_agreement", n, seed)
        X = rng.normal(size=(n, 8)).astype(np.float32)
    elif mechanism == "label_opposition":
        G, _, y = generate_train_problem("local_opposition", n, seed)
        X = rng.normal(size=(n, 8)).astype(np.float32)
    elif mechanism == "role_aligned":
        G, _, y = generate_train_problem("role_correspondence", n, seed)
        X = rng.normal(size=(n, 8)).astype(np.float32)
    else:
        raise ValueError(mechanism)

    pairs_all = _all_ordered_pairs(n)
    raw_x_all = np.linalg.norm(X[pairs_all[:, 0]] - X[pairs_all[:, 1]], axis=1)
    tx_x = gg.fit_rank_transform(raw_x_all)
    emb = gg.propagation_embeddings(G, t_list, min(n_landmarks, n-1), seed=seed)
    role = gg.role_signature(G)
    adj = nx.to_scipy_sparse_array(G, format="csr")

    raw_s_all = {
        "local": 1.0 - np.asarray(adj[pairs_all[:, 0], pairs_all[:, 1]]).ravel(),
        "t1": gg.pairwise_from_embedding(emb[t_list[0]], pairs_all),
        "t2": gg.pairwise_from_embedding(emb[t_list[1]], pairs_all),
        "role": np.linalg.norm(role[pairs_all[:, 0]] - role[pairs_all[:, 1]], axis=1),
    }
    tx_s = {
        "local": gg.fit_local_rank_transform(G),
        "t1": gg.fit_rank_transform(raw_s_all["t1"]),
        "t2": gg.fit_rank_transform(raw_s_all["t2"]),
        "role": gg.fit_rank_transform(raw_s_all["role"]),
    }

    labeled = np.arange(n, dtype=np.int64)
    cap = min(3000, n * (n - 1))
    if n * (n - 1) <= cap:
        lp = pairs_all
    else:
        li = rng.integers(0, n, size=cap * 2)
        lj = rng.integers(0, n, size=cap * 2)
        good = li != lj
        lp = np.stack([li[good][:cap], lj[good][:cap]], axis=1)

    U_X_lp = tx_x(np.linalg.norm(X[lp[:, 0]] - X[lp[:, 1]], axis=1))
    U_S_lp = {
        "local": tx_s["local"](1.0 - np.asarray(adj[lp[:, 0], lp[:, 1]]).ravel()),
        "t1": tx_s["t1"](gg.pairwise_from_embedding(emb[t_list[0]], lp)),
        "t2": tx_s["t2"](gg.pairwise_from_embedding(emb[t_list[1]], lp)),
        "role": tx_s["role"](np.linalg.norm(role[lp[:, 0]] - role[lp[:, 1]], axis=1)),
    }
    return {
        "G": G, "X": X, "y": y, "label_pairs": lp,
        "U_X_label_pairs": U_X_lp, "U_S_label_pairs": U_S_lp,
    }


def fit_dictionaries(cfg: ExperimentConfig) -> Tuple[np.ndarray, np.ndarray]:
    native = gg.fit_native_templates_joint(
        _native_generator_record,
        structural_keys=STRUCTURAL_KEYS,
        K=cfg.K,
        n_graphs=cfg.template_graphs_per_mechanism,
        local_pairs=cfg.local_pairs,
        n=cfg.template_n,
        t_list=cfg.t_list,
        n_landmarks=cfg.n_landmarks,
    )
    task = gg.fit_task_templates(
        _task_generator_record,
        STRUCTURAL_KEYS,
        K=cfg.K,
        n_graphs=cfg.task_template_graphs_per_mechanism,
        kappa=cfg.kappa,
        n=cfg.template_n,
        t_list=cfg.t_list,
        n_landmarks=cfg.n_landmarks,
    )
    return native, task


# =============================================================================
# Representation extraction: grammar, raw fields, conventional statistics
# =============================================================================

@dataclass
class RepresentationBundle:
    alpha: np.ndarray              # 5
    beta: np.ndarray               # 4 (zeros if unavailable)
    gamma: np.ndarray              # 9
    raw_native: np.ndarray         # 4*K*K
    raw_task: np.ndarray           # K*(1+2*4), zero if unavailable
    raw_full: np.ndarray           # raw_native || raw_task
    task_support: bool
    role_signature: np.ndarray


def _extract_raw_fields(
    G: nx.Graph,
    X: np.ndarray,
    y_observed: Optional[Dict[int, int]],
    cfg: ExperimentConfig,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, bool, np.ndarray]:
    """Mirror the grammar core's sampling/rank conventions exactly."""
    # Our Problems use 0..n-1, so no row-remapping ambiguity.
    n = G.number_of_nodes()
    emb = gg.propagation_embeddings(G, cfg.t_list, min(cfg.n_landmarks, n-1), seed=seed)
    role = gg.role_signature(G)
    adj = nx.to_scipy_sparse_array(G, format="csr")

    def raw_key(key: str, prs: np.ndarray) -> np.ndarray:
        if key == "local":
            return 1.0 - np.asarray(adj[prs[:, 0], prs[:, 1]]).ravel()
        if key == "role":
            return np.linalg.norm(role[prs[:, 0]] - role[prs[:, 1]], axis=1)
        t = cfg.t_list[0] if key == "t1" else cfg.t_list[1]
        return gg.pairwise_from_embedding(emb[t], prs)

    gp = gg.uniform_pair_sample(n, cfg.m_pairs, seed=seed)
    raw_x = np.linalg.norm(X[gp[:, 0]] - X[gp[:, 1]], axis=1)
    tx_x = gg.fit_rank_transform(raw_x)
    U_X = tx_x(raw_x)

    tx_s: Dict[str, Any] = {}
    U_s: Dict[str, np.ndarray] = {}
    for key in STRUCTURAL_KEYS:
        if key == "local":
            tx_s[key] = gg.fit_local_rank_transform(G)
        else:
            rr = raw_key(key, gp)
            tx_s[key] = gg.fit_rank_transform(rr)
            U_s[key] = tx_s[key](rr)

    local_prs = gg.sample_local_edge_nonedge_pairs(
        G, cfg.local_pairs, seed=seed + 17, edge_fraction=0.5
    )
    local_U_X = tx_x(np.linalg.norm(X[local_prs[:, 0]] - X[local_prs[:, 1]], axis=1))
    local_U_S = tx_s["local"](raw_key("local", local_prs))

    native_pieces = []
    for key in STRUCTURAL_KEYS:
        if key == "local":
            field = gg.empirical_alignment_field(local_U_S, local_U_X, K=cfg.K)
        else:
            field = gg.empirical_alignment_field(U_s[key], U_X, K=cfg.K)
        native_pieces.append(field.ravel())
    raw_native = np.concatenate(native_pieces)

    raw_task_dim = cfg.K * (1 + 2 * len(STRUCTURAL_KEYS))
    raw_task = np.zeros(raw_task_dim, dtype=float)
    support = False

    if y_observed is not None and len(y_observed) >= max(4, cfg.kappa):
        labeled = np.array(sorted(y_observed.keys()), dtype=np.int64)
        node_fold = gg.assign_node_folds(labeled, cfg.kappa, seed=seed)
        max_ordered = len(labeled) * (len(labeled) - 1)
        cap = min(cfg.label_pairs_cap, max_ordered)
        rng = np.random.default_rng(seed)
        if max_ordered <= cfg.label_pairs_cap:
            ii, jj = np.meshgrid(labeled, labeled, indexing="ij")
            lp = np.stack([ii.ravel(), jj.ravel()], axis=1)
            lp = lp[lp[:, 0] != lp[:, 1]]
        else:
            # Uniform ordered labeled-pair Monte Carlo sample.
            out = []
            while sum(len(x) for x in out) < cap:
                i = rng.choice(labeled, size=2*cap, replace=True)
                j = rng.choice(labeled, size=2*cap, replace=True)
                p = np.stack([i, j], axis=1)
                p = p[p[:, 0] != p[:, 1]]
                out.append(p[: max(0, cap - sum(len(x) for x in out))])
            lp = np.vstack(out)[:cap]

        if len(lp):
            lUx = tx_x(np.linalg.norm(X[lp[:, 0]] - X[lp[:, 1]], axis=1))
            lUs = {key: tx_s[key](raw_key(key, lp)) for key in STRUCTURAL_KEYS}
            fields, support = gg.task_conditioned_fields(
                lp, y_observed, lUs, lUx, node_fold, cfg.kappa, K=cfg.K
            )
            raw_task = np.concatenate(
                [fields["A_XY"]] + [
                    np.concatenate([fields["A_SY"][k], fields["A_SY_given_X"][k]])
                    for k in STRUCTURAL_KEYS
                ]
            )

    return raw_native, raw_task, bool(support), role


def extract_representations(problem: Problem, native_templates: np.ndarray,
                            task_templates: np.ndarray, cfg: ExperimentConfig,
                            seed: int, y_observed: Optional[Dict[int, int]] = None) -> RepresentationBundle:
    if y_observed is None:
        y_observed = problem.y_dict(problem.train_mask)
    raw_native, raw_task, support, role = _extract_raw_fields(
        problem.G, problem.X, y_observed, cfg, seed
    )
    alpha, _ = gg.project_nnls(raw_native, native_templates)
    if support:
        beta, _ = gg.project_nnls(raw_task, task_templates)
    else:
        beta = np.zeros(task_templates.shape[1], dtype=float)
    gamma = np.concatenate([alpha, beta])
    return RepresentationBundle(
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        raw_native=raw_native,
        raw_task=raw_task,
        raw_full=np.concatenate([raw_native, raw_task]),
        task_support=support,
        role_signature=role,
    )


# -----------------------------------------------------------------------------
# Conventional-statistics baseline
# -----------------------------------------------------------------------------


def edge_label_counts(G: nx.Graph, labels: Mapping[int, int]) -> Tuple[int, int]:
    same = total = 0
    for u, v in G.edges():
        if u in labels and v in labels:
            total += 1
            same += int(labels[u] == labels[v])
    return same, total


def adjusted_homophily(G: nx.Graph, labels: Mapping[int, int]) -> float:
    """Degree-adjusted edge homophily on the labeled induced edge sample."""
    nodes = set(labels)
    edges = [(u, v) for u, v in G.edges() if u in nodes and v in nodes]
    if not edges:
        return 0.0
    h = np.mean([labels[u] == labels[v] for u, v in edges])
    # Endpoint class frequencies (degree-weighted on the observed labeled edges).
    endpoint = []
    for u, v in edges:
        endpoint.extend([labels[u], labels[v]])
    vals, counts = np.unique(endpoint, return_counts=True)
    p = counts / counts.sum()
    expected = float(np.sum(p ** 2))
    return float((h - expected) / max(1.0 - expected, 1e-12))


def label_informativeness(G: nx.Graph, labels: Mapping[int, int]) -> float:
    """Normalized mutual information between labels at the two ends of an edge.

    This is a fixed low-capacity label-informativeness statistic; all labels used
    here come from the training-label set only.
    """
    nodes = set(labels)
    pairs = []
    for u, v in G.edges():
        if u in nodes and v in nodes:
            pairs.extend([(labels[u], labels[v]), (labels[v], labels[u])])
    if not pairs:
        return 0.0
    a = np.array([p[0] for p in pairs], dtype=int)
    b = np.array([p[1] for p in pairs], dtype=int)
    classes = np.unique(np.r_[a, b])
    idx = {c: i for i, c in enumerate(classes)}
    joint = np.zeros((len(classes), len(classes)), dtype=float)
    for x, y in zip(a, b):
        joint[idx[x], idx[y]] += 1
    joint /= joint.sum()
    pa = joint.sum(1)
    pb = joint.sum(0)
    nz = joint > 0
    denom = pa[:, None] * pb[None, :]
    mi = float(np.sum(joint[nz] * np.log((joint[nz] + 1e-15) / (denom[nz] + 1e-15))))
    H = float(-np.sum(pa[pa > 0] * np.log(pa[pa > 0])))
    return mi / max(H, 1e-12)


def _spectral_summaries(G: nx.Graph, k: int = 4) -> np.ndarray:
    n = G.number_of_nodes()
    if n < 4 or G.number_of_edges() == 0:
        return np.zeros(2 * k)
    A = nx.to_scipy_sparse_array(G, format="csr", dtype=float)
    deg = np.asarray(A.sum(1)).ravel()
    dinv = np.zeros_like(deg)
    nz = deg > 0
    dinv[nz] = deg[nz] ** -0.5
    S = sparse.diags(dinv) @ A @ sparse.diags(dinv)
    kk = min(k, max(1, n - 2))
    try:
        large = np.sort(eigsh(S, k=kk, which="LA", return_eigenvectors=False))[::-1]
        small = np.sort(eigsh(S, k=kk, which="SA", return_eigenvectors=False))
    except Exception:
        large = np.zeros(kk)
        small = np.zeros(kk)
    return np.pad(np.r_[large, small], (0, 2*k - 2*kk))


def conventional_statistics(problem: Problem, y_observed: Optional[Dict[int, int]]) -> np.ndarray:
    G, X = problem.G, problem.X
    n, m = G.number_of_nodes(), G.number_of_edges()
    deg = np.array([d for _, d in G.degree()], dtype=float)
    if len(deg) == 0:
        deg = np.zeros(1)
    degree_stats = np.array([
        np.mean(deg), np.std(deg), np.quantile(deg, 0.25),
        np.quantile(deg, 0.5), np.quantile(deg, 0.75), np.max(deg),
    ])
    # Feature smoothness on edges, sampled if needed.
    edges = np.asarray(list(G.edges()), dtype=np.int64)
    if len(edges):
        if len(edges) > 5000:
            rng = np.random.default_rng(0)
            edges = edges[rng.choice(len(edges), 5000, replace=False)]
        feat_edge_dist = float(np.mean(np.linalg.norm(X[edges[:, 0]] - X[edges[:, 1]], axis=1)))
    else:
        feat_edge_dist = 0.0

    label_stats = np.zeros(3)
    if y_observed:
        label_stats[0] = adjusted_homophily(G, y_observed)
        label_stats[1] = label_informativeness(G, y_observed)
        vals = np.array(list(y_observed.values()), dtype=int)
        _, cnt = np.unique(vals, return_counts=True)
        p = cnt / cnt.sum()
        label_stats[2] = -np.sum(p * np.log(p + 1e-15))

    return np.r_[
        np.log1p(n), np.log1p(m), 2*m / max(n*(n-1), 1),
        degree_stats, feat_edge_dist, _spectral_summaries(G), label_stats,
    ].astype(float)


# =============================================================================
# Architecture bank
# =============================================================================

class FeatureOnlyMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float):
        super().__init__()
        self.l1 = nn.Linear(in_dim, hidden)
        self.l2 = nn.Linear(hidden, out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index=None, role_x=None):
        h = F.relu(self.l1(x))
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.l2(h)


class GCNModel(nn.Module):
    """Stacked GCN with configurable depth (default 2 layers).

    ``layers`` lets the propagation-depth ablation (see
    scripts/experiments/hop_sweep_from_tstar.py) read a stacked-layer count
    directly off a problem's diffusion peak t*, instead of always using the
    fixed 2-layer default; every other caller keeps layers=2 (identical to
    the previous hardcoded architecture) unless it opts in explicitly.
    """
    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float,
                 layers: int = 2):
        super().__init__()
        _require_pyg()
        if layers < 1:
            raise ValueError("GCNModel needs at least one layer")
        self.dropout = dropout
        if layers == 1:
            self.convs = nn.ModuleList([GCNConv(in_dim, out_dim)])
        else:
            convs = [GCNConv(in_dim, hidden)]
            convs += [GCNConv(hidden, hidden) for _ in range(layers - 2)]
            convs += [GCNConv(hidden, out_dim)]
            self.convs = nn.ModuleList(convs)

    def forward(self, x, edge_index, role_x=None):
        h = x
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            if i < len(self.convs) - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        return h


class FAGCNModel(nn.Module):
    """Frequency-adaptive / heterophily-aware architecture using PyG FAConv."""
    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float, layers: int = 2):
        super().__init__()
        _require_pyg()
        self.lin_in = nn.Linear(in_dim, hidden)
        self.convs = nn.ModuleList([FAConv(hidden, eps=0.1, dropout=dropout) for _ in range(layers)])
        self.lin_out = nn.Linear(hidden, out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index, role_x=None):
        h = F.dropout(x, p=self.dropout, training=self.training)
        h = F.relu(self.lin_in(h))
        h0 = h
        for conv in self.convs:
            h = conv(h, h0, edge_index)
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.lin_out(h)


class APPNPModel(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float,
                 hops: int, alpha: float):
        super().__init__()
        _require_pyg()
        self.l1 = nn.Linear(in_dim, hidden)
        self.l2 = nn.Linear(hidden, out_dim)
        self.prop = APPNPProp(K=hops, alpha=alpha)
        self.dropout = dropout

    def forward(self, x, edge_index, role_x=None):
        h = F.relu(self.l1(x))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.l2(h)
        return self.prop(h, edge_index)


class RoleAugmentedSAGE(nn.Module):
    def __init__(self, in_dim: int, role_dim: int, hidden: int, out_dim: int, dropout: float):
        super().__init__()
        _require_pyg()
        self.c1 = SAGEConv(in_dim + role_dim, hidden)
        self.c2 = SAGEConv(hidden, out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index, role_x=None):
        if role_x is None:
            raise ValueError("role_structural architecture requires role_x")
        h = torch.cat([x, role_x], dim=1)
        h = F.relu(self.c1(h, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.c2(h, edge_index)


class GPSGraphTransformer(nn.Module):
    """a6: GPS-style local-global Graph Transformer.

    Each layer combines a local GCN branch with global Performer attention via
    PyG ``GPSConv``. Performer is used instead of quadratic full attention so
    the same architecture definition can be attempted on the larger real
    graphs. We intentionally do not add an extra handcrafted positional or
    structural encoding here: the GT receives the same node features and graph
    edges as the ordinary graph models, making this a conservative stress test
    of whether a flexible local-global architecture dominates the bank.
    """
    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float,
                 layers: int = 2, heads: int = 4, attn_type: str = "performer"):
        super().__init__()
        _require_pyg()
        if GPSConv is None:
            raise ImportError(
                "graph_transformer requires a PyTorch-Geometric version with GPSConv. "
                "Upgrade PyG or run the five-model robustness bank with --no-gt."
            )
        if hidden % heads != 0:
            raise ValueError(f"GPS hidden={hidden} must be divisible by heads={heads}")
        self.input_proj = nn.Linear(in_dim, hidden)
        self.dropout = dropout
        self.convs = nn.ModuleList()
        for _ in range(layers):
            local_conv = GCNConv(hidden, hidden)
            attn_kwargs: Dict[str, Any] = {"dropout": dropout}
            if attn_type == "performer":
                # Keep the total attention width comparable to ``hidden``.
                attn_kwargs["head_channels"] = hidden // heads
            self.convs.append(
                GPSConv(
                    channels=hidden,
                    conv=local_conv,
                    heads=heads,
                    dropout=dropout,
                    norm="batch_norm",
                    attn_type=attn_type,
                    attn_kwargs=attn_kwargs,
                )
            )
        self.output = nn.Linear(hidden, out_dim)

    def forward(self, x, edge_index, role_x=None):
        h = F.relu(self.input_proj(x))
        h = F.dropout(h, p=self.dropout, training=self.training)
        # Node classification here is full-batch on one graph. Explicitly mark
        # every node as belonging to graph 0 for the global attention branch.
        batch = torch.zeros(h.size(0), dtype=torch.long, device=h.device)
        for conv in self.convs:
            h = conv(h, edge_index, batch=batch)
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.output(h)


def make_model(name: str, in_dim: int, role_dim: int, n_classes: int,
               cfg: ArchConfig) -> nn.Module:
    if name == "feature_only":
        return FeatureOnlyMLP(in_dim, cfg.hidden, n_classes, cfg.dropout)
    if name == "local_lowpass":
        return GCNModel(in_dim, cfg.hidden, n_classes, cfg.dropout, cfg.layers)
    if name == "high_pass":
        return FAGCNModel(in_dim, cfg.hidden, n_classes, cfg.dropout, cfg.layers)
    if name == "multihop":
        return APPNPModel(in_dim, cfg.hidden, n_classes, cfg.dropout, cfg.hops, cfg.alpha)
    if name == "role_structural":
        return RoleAugmentedSAGE(in_dim, role_dim, cfg.hidden, n_classes, cfg.dropout)
    if name == "graph_transformer":
        return GPSGraphTransformer(
            in_dim, cfg.hidden, n_classes, cfg.dropout,
            layers=cfg.layers, heads=cfg.heads, attn_type=cfg.attn_type,
        )
    raise ValueError(name)


def _metric_from_logits(logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor,
                        metric: str) -> float:
    yy = y[mask].detach().cpu().numpy()
    out = logits[mask].detach().cpu()
    if len(yy) == 0:
        return float("nan")
    if metric == "roc_auc":
        if out.shape[1] != 2 or len(np.unique(yy)) < 2:
            return float("nan")
        prob = torch.softmax(out, dim=1)[:, 1].numpy()
        return float(roc_auc_score(yy, prob))
    pred = out.argmax(dim=1).numpy()
    return float(np.mean(pred == yy))


def train_one_seed(problem: Problem, role_x: np.ndarray, arch_name: str,
                   arch_cfg: ArchConfig, seed: int) -> float:
    _require_pyg()
    set_all_seeds(seed)
    data = problem_to_pyg(problem, role_x=role_x).to(DEVICE)
    model = make_model(
        arch_name, data.x.shape[1], data.role_x.shape[1], problem.n_classes, arch_cfg
    ).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=arch_cfg.lr, weight_decay=arch_cfg.weight_decay)

    best_val = -np.inf
    best_state = None
    bad = 0
    for _ in range(arch_cfg.epochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        out = model(data.x, data.edge_index, data.role_x)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            out_val = model(data.x, data.edge_index, data.role_x)
            val = _metric_from_logits(out_val, data.y, data.val_mask, problem.metric)
        if np.isfinite(val) and val > best_val + 1e-8:
            best_val = val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= arch_cfg.patience:
                break

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index, data.role_x)
        test = _metric_from_logits(out, data.y, data.test_mask, problem.metric)
    return float(test)


class PerformanceCache:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self.db = json.loads(self.path.read_text())
        else:
            self.db: Dict[str, float] = {}

    def get(self, key: str) -> Optional[float]:
        return self.db.get(key)

    def put(self, key: str, value: float) -> None:
        self.db[key] = float(value)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.db, indent=2, sort_keys=True))
        tmp.replace(self.path)


class JsonCheckpoint:
    """Append-only, resumable per-item checkpoint backed by a single JSON file.

    Each item is identified by a stable string key (e.g. a dataset name, or
    "{family}_{mechanism}_{index}" for a synthetic corpus slot). ``load(key)``
    returns the saved payload for a key already present, or ``None``, so a
    caller can skip redoing expensive work (representation extraction,
    architecture training) for items that already finished on a prior,
    interrupted run. ``save(key, payload)`` merges one item's result in and
    rewrites the file atomically (write-tmp + replace, same pattern as
    PerformanceCache above), so a crash after N/M items keeps all N results
    on disk instead of losing the whole run.

    Pass ``resume=False`` to ignore whatever is already on disk (start this
    run from scratch) while still recording fresh results as they complete,
    e.g. for ``--fresh`` reruns that should overwrite a stale checkpoint
    rather than trust it.
    """

    def __init__(self, path: str | Path, resume: bool = True):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if resume and self.path.exists():
            self.items: Dict[str, Any] = json.loads(self.path.read_text())
        else:
            self.items = {}

    def load(self, key: str) -> Optional[Dict[str, Any]]:
        return self.items.get(key)

    def save(self, key: str, payload: Dict[str, Any]) -> None:
        self.items[key] = payload
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.items, indent=2, sort_keys=True))
        tmp.replace(self.path)

    def __len__(self) -> int:
        return len(self.items)


def _problem_fingerprint(problem: Problem) -> str:
    """Compact fingerprint preventing silent cache reuse across changed splits/data.

    The fingerprint is memoized on the ``Problem`` instance because hashing a
    large real graph once is useful; hashing it once per architecture/seed is
    unnecessary overhead.
    """
    cached = getattr(problem, "_performance_cache_fingerprint", None)
    if cached is not None:
        return cached
    h = hashlib.sha1()
    h.update(problem.name.encode())
    edges = np.asarray(
        sorted((min(u, v), max(u, v)) for u, v in problem.G.edges()),
        dtype=np.int64,
    )
    for arr in (
        edges, np.asarray(problem.X, dtype=np.float32), np.asarray(problem.y, dtype=np.int64),
        np.asarray(problem.train_mask, dtype=np.uint8),
        np.asarray(problem.val_mask, dtype=np.uint8),
        np.asarray(problem.test_mask, dtype=np.uint8),
    ):
        a = np.ascontiguousarray(arr)
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    h.update(problem.metric.encode())
    digest = h.hexdigest()
    setattr(problem, "_performance_cache_fingerprint", digest)
    return digest


def _perf_cache_key(problem: Problem, arch: str, seed: int, cfg: ArchConfig) -> str:
    payload = json.dumps({
        "problem_fingerprint": _problem_fingerprint(problem),
        "arch": arch,
        "seed": seed,
        "cfg": asdict(cfg),
    }, sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()


def performance_vector(problem: Problem, role_x: np.ndarray, cfg: ExperimentConfig,
                       cache: PerformanceCache) -> Tuple[np.ndarray, np.ndarray]:
    means, stds = [], []
    for arch in active_architectures(cfg):
        vals = []
        ac = cfg.arch[arch]
        for seed in cfg.model_seeds:
            key = _perf_cache_key(problem, arch, seed, ac)
            val = cache.get(key)
            if val is None:
                val = train_one_seed(problem, role_x, arch, ac, seed)
                cache.put(key, val)
            vals.append(val)
        means.append(np.nanmean(vals))
        stds.append(np.nanstd(vals))
    return np.asarray(means), np.asarray(stds)


# =============================================================================
# Meta-predictor and metrics
# =============================================================================

class PerformancePredictor:
    """Identical low-capacity downstream machinery for every representation."""
    def __init__(self, alpha: float = 1.0):
        self.pipe = Pipeline([
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ])

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "PerformancePredictor":
        self.pipe.fit(X, Y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.pipe.predict(np.asarray(x).reshape(1, -1))[0])

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.pipe.predict(X))


@dataclass
class SelectionMetrics:
    spearman: float
    top1: float
    top3: float
    regret: float


def selection_metrics(true: np.ndarray, pred: np.ndarray) -> SelectionMetrics:
    rho = spearmanr(true, pred).statistic
    if not np.isfinite(rho):
        rho = 0.0
    best = int(np.argmax(true))
    chosen = int(np.argmax(pred))
    top3 = set(np.argsort(pred)[-3:].tolist())
    return SelectionMetrics(
        spearman=float(rho),
        top1=float(best == chosen),
        top3=float(best in top3),
        regret=float(np.max(true) - true[chosen]),
    )


def random_selection_metrics(true: np.ndarray, draws: int, seed: int) -> SelectionMetrics:
    rng = np.random.default_rng(seed)
    ms = []
    for _ in range(draws):
        ms.append(selection_metrics(true, rng.normal(size=len(true))))
    return aggregate_selection(ms)


def aggregate_selection(ms: Sequence[SelectionMetrics]) -> SelectionMetrics:
    if not ms:
        return SelectionMetrics(np.nan, np.nan, np.nan, np.nan)
    return SelectionMetrics(**{
        k: float(np.nanmean([getattr(m, k) for m in ms]))
        for k in SelectionMetrics.__annotations__
    })


def metrics_mean_std(ms: Sequence[SelectionMetrics]) -> Dict[str, Dict[str, float]]:
    return {
        k: {
            "mean": float(np.nanmean([getattr(m, k) for m in ms])),
            "std": float(np.nanstd([getattr(m, k) for m in ms])),
        }
        for k in SelectionMetrics.__annotations__
    }


def paired_regret_tests(results: Mapping[str, Sequence[SelectionMetrics]], ours: str) -> Dict[str, float]:
    out = {}
    a = np.array([x.regret for x in results[ours]])
    for b, vals in results.items():
        if b == ours or b in {"random", "oracle"}:
            continue
        bb = np.array([x.regret for x in vals])
        if len(a) != len(bb) or len(a) < 5:
            out[b] = np.nan
            continue
        try:
            out[b] = float(wilcoxon(a, bb, alternative="less").pvalue)
        except ValueError:
            out[b] = 1.0
    return out


# =============================================================================
# Corpus records shared by E2--E4
# =============================================================================

@dataclass
class CorpusItem:
    problem: Problem
    rep: RepresentationBundle
    conventional: np.ndarray
    perf: np.ndarray
    perf_std: np.ndarray
    mechanism: str


def _corpus_item_checkpoint(item: CorpusItem) -> Dict[str, Any]:
    """JSON-safe snapshot of everything downstream code reads off a CorpusItem.

    Deliberately excludes ``item.problem`` (the graph/X/y/masks): rebuilding a
    Problem is cheap for every corpus in this file (real datasets hit PyG's
    own on-disk processed-data cache; synthetic/mixture problems are
    regenerated in O(ms) from their draw seed), so callers always reconstruct
    a real Problem on resume rather than persisting/restoring one. What *is*
    expensive -- representation extraction and architecture training -- is
    exactly what this payload lets a resumed run skip.
    """
    return {
        "mechanism": item.mechanism,
        "conventional": np.asarray(item.conventional, dtype=float).tolist(),
        "perf": np.asarray(item.perf, dtype=float).tolist(),
        "perf_std": np.asarray(item.perf_std, dtype=float).tolist(),
        "rep": {
            "alpha": np.asarray(item.rep.alpha, dtype=float).tolist(),
            "beta": np.asarray(item.rep.beta, dtype=float).tolist(),
            "gamma": np.asarray(item.rep.gamma, dtype=float).tolist(),
            "raw_native": np.asarray(item.rep.raw_native, dtype=float).tolist(),
            "raw_task": np.asarray(item.rep.raw_task, dtype=float).tolist(),
            "raw_full": np.asarray(item.rep.raw_full, dtype=float).tolist(),
            "task_support": bool(item.rep.task_support),
        },
    }


def _corpus_item_from_checkpoint(problem: Problem, ck: Dict[str, Any]) -> CorpusItem:
    """Inverse of ``_corpus_item_checkpoint``, given a freshly-built Problem.

    ``rep.role_signature`` is left empty: it is only ever consumed while
    *computing* perf (to train the role_structural architecture), and perf is
    already frozen in the checkpoint, so nothing downstream of this call
    needs it again.
    """
    rep = RepresentationBundle(
        alpha=np.asarray(ck["rep"]["alpha"], dtype=np.float64),
        beta=np.asarray(ck["rep"]["beta"], dtype=np.float64),
        gamma=np.asarray(ck["rep"]["gamma"], dtype=np.float64),
        raw_native=np.asarray(ck["rep"]["raw_native"], dtype=np.float64),
        raw_task=np.asarray(ck["rep"]["raw_task"], dtype=np.float64),
        raw_full=np.asarray(ck["rep"]["raw_full"], dtype=np.float64),
        task_support=bool(ck["rep"]["task_support"]),
        role_signature=np.zeros(0, dtype=np.float64),
    )
    return CorpusItem(
        problem=problem,
        rep=rep,
        conventional=np.asarray(ck["conventional"], dtype=np.float64),
        perf=np.asarray(ck["perf"], dtype=np.float64),
        perf_std=np.asarray(ck["perf_std"], dtype=np.float64),
        mechanism=ck["mechanism"],
    )


def _problem_from_generated(name: str, G: nx.Graph, X: np.ndarray, y_dict: Dict[int, int],
                            seed: int) -> Problem:
    y = np.array([y_dict[i] for i in range(len(y_dict))], dtype=int)
    tr, va, te = stratified_masks(y, seed)
    return Problem(name=name, G=G, X=np.asarray(X, dtype=np.float32), y=y,
                   train_mask=tr, val_mask=va, test_mask=te, metric="accuracy")


def build_synthetic_corpus(
    family: str,
    per_mechanism: int,
    native_templates: np.ndarray,
    task_templates: np.ndarray,
    cfg: ExperimentConfig,
    cache: PerformanceCache,
    seed: int,
    checkpoint_path: Optional[str | Path] = None,
    resume: bool = True,
) -> List[CorpusItem]:
    """Build the (family, mechanism, replicate) synthetic corpus.

    If ``checkpoint_path`` is given, each (mechanism, replicate) slot is
    checkpointed under the key ``"{family}_{mechanism}_{j}"`` after it
    finishes. On a later call with the same path (and ``resume=True``), a
    slot whose checkpointed draw seed matches the freshly-drawn ``s`` for
    that slot is loaded from disk instead of recomputed -- the raw problem
    is still cheaply regenerated from ``s`` (so ``item.problem`` is always a
    real Problem, never a stub), only representation extraction and
    architecture training are skipped. A seed mismatch (e.g. ``cfg`` or
    ``per_mechanism`` changed since the checkpoint was written) is treated
    as a miss and recomputed, with a warning, rather than silently reused.
    """
    rng = np.random.default_rng(seed)
    ckpt = JsonCheckpoint(checkpoint_path, resume=resume) if checkpoint_path is not None else None
    items: List[CorpusItem] = []
    for mech in NATIVE_MECHANISMS:
        for j in range(per_mechanism):
            s = int(rng.integers(1, 2**31 - 1))
            if family == "train":
                G, X, y = generate_train_problem(mech, cfg.synthetic_n, s)
            elif family == "test":
                G, X, y = generate_test_problem(mech, cfg.synthetic_n, s)
            else:
                raise ValueError(family)
            p = _problem_from_generated(f"{family}_{mech}_{j}_{s}", G, X, y, seed=s)

            key = f"{family}_{mech}_{j}"
            saved = ckpt.load(key) if ckpt is not None else None
            if saved is not None and saved.get("seed") == s:
                print(f"[{family}] {key} (resumed from checkpoint)")
                items.append(_corpus_item_from_checkpoint(p, saved))
                continue
            if saved is not None:
                warnings.warn(
                    f"[{family}] checkpoint seed mismatch for {key} "
                    "(cfg/per_mechanism changed?); recomputing"
                )

            y_train = p.y_dict(p.train_mask)
            rep = extract_representations(p, native_templates, task_templates, cfg, seed=s, y_observed=y_train)
            conv = conventional_statistics(p, y_train)
            perf, perf_std = performance_vector(p, rep.role_signature, cfg, cache)
            item = CorpusItem(p, rep, conv, perf, perf_std, mech)
            items.append(item)
            if ckpt is not None:
                payload = _corpus_item_checkpoint(item)
                payload["seed"] = s
                ckpt.save(key, payload)
    return items


def build_mixture_corpus(
    n_problems: int,
    native_templates: np.ndarray,
    task_templates: np.ndarray,
    cfg: ExperimentConfig,
    cache: PerformanceCache,
    seed: int,
    checkpoint_path: Optional[str | Path] = None,
    resume: bool = True,
) -> List[CorpusItem]:
    """Build the held-out generator-mixture corpus; see build_synthetic_corpus
    for the checkpoint/resume contract (same idea, keyed by "mixture_{i}")."""
    rng = np.random.default_rng(seed)
    ckpt = JsonCheckpoint(checkpoint_path, resume=resume) if checkpoint_path is not None else None
    out = []
    for i in range(n_problems):
        s = int(rng.integers(1, 2**31 - 1))
        G, X, y, pair = generate_mixture_problem(cfg.synthetic_n, s)
        p = _problem_from_generated(f"mixture_{pair[0]}_{pair[1]}_{i}_{s}", G, X, y, s)

        key = f"mixture_{i}"
        saved = ckpt.load(key) if ckpt is not None else None
        if saved is not None and saved.get("seed") == s:
            print(f"[mixture] {key} (resumed from checkpoint)")
            out.append(_corpus_item_from_checkpoint(p, saved))
            continue
        if saved is not None:
            warnings.warn(f"[mixture] checkpoint seed mismatch for {key} (cfg changed?); recomputing")

        y_train = p.y_dict(p.train_mask)
        rep = extract_representations(p, native_templates, task_templates, cfg, s, y_train)
        conv = conventional_statistics(p, y_train)
        perf, perf_std = performance_vector(p, rep.role_signature, cfg, cache)
        item = CorpusItem(p, rep, conv, perf, perf_std, f"{pair[0]}+{pair[1]}")
        out.append(item)
        if ckpt is not None:
            payload = _corpus_item_checkpoint(item)
            payload["seed"] = s
            ckpt.save(key, payload)
    return out


def representation_matrix(items: Sequence[CorpusItem], kind: str) -> np.ndarray:
    if kind == "conventional":
        return np.vstack([x.conventional for x in items])
    if kind == "raw":
        return np.vstack([x.rep.raw_full for x in items])
    if kind == "raw_native":
        return np.vstack([x.rep.raw_native for x in items])
    if kind == "alpha":
        return np.vstack([x.rep.alpha for x in items])
    if kind == "gamma":
        return np.vstack([x.rep.gamma for x in items])
    if kind == "task_raw":
        return np.vstack([x.rep.raw_task for x in items])
    raise ValueError(kind)


def performance_matrix(items: Sequence[CorpusItem]) -> np.ndarray:
    return np.vstack([x.perf for x in items])


# =============================================================================
# E1: matched-summary counterfactual diagnosis
# =============================================================================


def _feature_near_edge_pairs(G: nx.Graph, X: np.ndarray) -> np.ndarray:
    edges = np.asarray(list(G.edges()), dtype=np.int64)
    if len(edges) == 0:
        return edges.reshape(0, 2)
    d = np.linalg.norm(X[edges[:, 0]] - X[edges[:, 1]], axis=1)
    return edges[d <= np.median(d)]


def _two_hop_nonedge_pairs(G: nx.Graph, max_pairs: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pairs: set[Tuple[int, int]] = set()
    nodes = list(G.nodes())
    rng.shuffle(nodes)
    for u in nodes:
        one = set(G.neighbors(u))
        two = set()
        for v in one:
            two.update(G.neighbors(v))
        two.discard(u)
        two -= one
        for v in two:
            a, b = sorted((u, v))
            pairs.add((a, b))
            if len(pairs) >= max_pairs:
                return np.asarray(list(pairs), dtype=np.int64)
    return np.asarray(list(pairs), dtype=np.int64) if pairs else np.empty((0, 2), dtype=np.int64)


def _pair_agreement(y: np.ndarray, pairs: np.ndarray) -> float:
    if len(pairs) == 0:
        return 0.5
    return float(np.mean(y[pairs[:, 0]] == y[pairs[:, 1]]))


def _adjusted_h_full(G: nx.Graph, y: np.ndarray) -> float:
    return adjusted_homophily(G, {i: int(y[i]) for i in range(len(y))})


def optimize_counterfactual_labels(
    G: nx.Graph,
    X: np.ndarray,
    mechanism: str,
    target_adj_h: float,
    tolerance: float,
    steps: int,
    seed: int,
) -> np.ndarray:
    """Balanced binary labels with matched global adjusted homophily but
    different conditional relational organization.

    Because global homophily is constrained to be the same, the mechanism score
    is conditional: agreement/opposition are measured on FEATURE-NEAR edges;
    delayed dependence is measured on two-hop nonedges. This makes the
    matched-summary counterfactual scientifically possible rather than trying
    to demand simultaneously high and low global homophily.
    """
    rng = np.random.default_rng(seed)
    n = G.number_of_nodes()
    y = np.zeros(n, dtype=int)
    y[rng.choice(n, size=n//2, replace=False)] = 1
    near_edges = _feature_near_edge_pairs(G, X)
    twohop = _two_hop_nonedge_pairs(G, max_pairs=max(1500, 6*n), seed=seed)

    def score(z: np.ndarray) -> float:
        h = _adjusted_h_full(G, z)
        near = _pair_agreement(z, near_edges)
        two = _pair_agreement(z, twohop)
        if mechanism == "one_hop_agreement":
            mech = near
        elif mechanism == "opposition":
            mech = 1.0 - near
        elif mechanism == "delayed_two_hop":
            mech = two - 0.35 * near
        else:
            raise ValueError(mechanism)
        # Very strong penalty keeps adjusted homophily matched; balance is
        # exactly preserved by opposite-label swaps below.
        return mech - 30.0 * (h - target_adj_h) ** 2

    cur = score(y)
    best = y.copy()
    best_s = cur
    T0 = 0.05
    for t in range(steps):
        zero = np.flatnonzero(y == 0)
        one = np.flatnonzero(y == 1)
        if not len(zero) or not len(one):
            break
        a = int(rng.choice(zero))
        b = int(rng.choice(one))
        y[a], y[b] = 1, 0
        new = score(y)
        temp = T0 * (1 - t / max(steps, 1)) + 1e-4
        if new >= cur or rng.random() < math.exp((new - cur) / temp):
            cur = new
            if new > best_s:
                best_s = new
                best = y.copy()
        else:
            y[a], y[b] = 0, 1

    achieved = _adjusted_h_full(G, best)
    if abs(achieved - target_adj_h) > tolerance:
        warnings.warn(
            f"E1 {mechanism}: adjusted homophily {achieved:.3f} misses target "
            f"{target_adj_h:.3f} by > {tolerance:.3f}. Increase anneal_steps or "
            "regenerate the base graph before reporting this triple."
        )
    return best


def make_e1_triple(base_id: int, cfg: ExperimentConfig, seed: int) -> List[Tuple[str, Problem]]:
    # Neutral base: use a diffuse G_train graph only to provide a nontrivial G,X;
    # task labels below are independently optimized and are the only thing varied.
    G, X, _ = generate_train_problem("diffuse_organization", cfg.e1_n, seed)
    tr, va, te = same_split_masks(cfg.e1_n, seed + 1)
    out = []
    for j, mech in enumerate(("one_hop_agreement", "opposition", "delayed_two_hop")):
        y = optimize_counterfactual_labels(
            G, X, mech,
            target_adj_h=cfg.e1_target_adjusted_homophily,
            tolerance=cfg.e1_homophily_tolerance,
            steps=cfg.e1_anneal_steps,
            seed=seed + 100*j,
        )
        p = Problem(
            name=f"e1_{base_id}_{mech}", G=G.copy(), X=X.copy(), y=y,
            train_mask=tr.copy(), val_mask=va.copy(), test_mask=te.copy(), metric="accuracy"
        )
        out.append((mech, p))
    return out


def _group_cv_id_accuracy(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> float:
    uniq = np.unique(groups)
    n_splits = min(5, len(uniq))
    if n_splits < 2:
        return float("nan")
    cv = GroupKFold(n_splits=n_splits)
    correct = total = 0
    for tr, te in cv.split(X, y, groups):
        clf = Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, C=1.0)),
        ]).fit(X[tr], y[tr])
        pred = clf.predict(X[te])
        correct += int(np.sum(pred == y[te]))
        total += len(te)
    return correct / max(total, 1)


def _group_excluded_nn_accuracy(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> float:
    correct = total = 0
    for g in np.unique(groups):
        tr = groups != g
        te = groups == g
        scaler = StandardScaler().fit(X[tr])
        A = scaler.transform(X[tr])
        B = scaler.transform(X[te])
        ytr = y[tr]
        for row, yy in zip(B, y[te]):
            d = np.linalg.norm(A - row[None, :], axis=1)
            pred = ytr[int(np.argmin(d))]
            correct += int(pred == yy)
            total += 1
    return correct / max(total, 1)


def run_e1(native_templates: np.ndarray, task_templates: np.ndarray,
           cfg: ExperimentConfig, out_dir: Path,
           e2_state: Optional[Dict[str, Any]] = None,
           cache: Optional[PerformanceCache] = None,
           seed: int = 10) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    reps = {"conventional": [], "raw": [], "grammar": []}
    labels, groups = [], []
    first_triple_payload = []

    mech_to_id = {m: i for i, m in enumerate(("one_hop_agreement", "opposition", "delayed_two_hop"))}
    for base_id in range(cfg.e1_triples):
        triple = make_e1_triple(base_id, cfg, int(rng.integers(1, 2**31-1)))
        for mech, p in triple:
            y_train = p.y_dict(p.train_mask)
            rep = extract_representations(p, native_templates, task_templates, cfg, seed=seed+base_id, y_observed=y_train)
            # E1 table specifically names adjusted homophily + label informativeness.
            conv2 = np.array([adjusted_homophily(p.G, y_train), label_informativeness(p.G, y_train)])
            reps["conventional"].append(conv2)
            reps["raw"].append(rep.raw_full)
            reps["grammar"].append(rep.gamma)
            labels.append(mech_to_id[mech])
            groups.append(base_id)

            if base_id == 0:
                row = {
                    "mechanism": mech,
                    "adjusted_homophily": float(conv2[0]),
                    "label_informativeness": float(conv2[1]),
                    "alpha": rep.alpha.tolist(),
                    "beta": rep.beta.tolist(),
                }
                if e2_state is not None:
                    row["predicted_performance"] = e2_state["predictors"]["gamma"].predict(rep.gamma).tolist()
                    arch_names = tuple(e2_state.get("architecture_names", active_architectures(cfg)))
                    row["predicted_best"] = arch_names[int(np.argmax(row["predicted_performance"]))]
                if cache is not None:
                    perf, _ = performance_vector(p, rep.role_signature, cfg, cache)
                    row["actual_performance"] = perf.tolist()
                    arch_names = tuple(e2_state.get("architecture_names", active_architectures(cfg))) if e2_state is not None else active_architectures(cfg)
                    row["actual_best"] = arch_names[int(np.argmax(perf))]
                first_triple_payload.append(row)

    y = np.asarray(labels)
    g = np.asarray(groups)
    table = {}
    for name, rows in reps.items():
        X = np.vstack(rows)
        table[name] = {
            "mechanism_id_accuracy": _group_cv_id_accuracy(X, y, g),
            "nn_retrieval_accuracy": _group_excluded_nn_accuracy(X, y, g),
        }

    out = {"table": table, "counterfactual": first_triple_payload}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "e1_results.json").write_text(json.dumps(out, indent=2))
    return out


# =============================================================================
# E2: cross-generator A->B
# =============================================================================


def fit_representation_predictors(train: Sequence[CorpusItem], cfg: ExperimentConfig) -> Dict[str, PerformancePredictor]:
    Y = performance_matrix(train)
    predictors = {}
    for kind in ("conventional", "raw", "alpha", "gamma", "task_raw"):
        predictors[kind] = PerformancePredictor(cfg.meta_ridge_alpha).fit(
            representation_matrix(train, kind), Y
        )
    return predictors


def evaluate_predictors(items: Sequence[CorpusItem], predictors: Mapping[str, PerformancePredictor],
                        cfg: ExperimentConfig, kinds: Sequence[str]) -> Dict[str, List[SelectionMetrics]]:
    out: Dict[str, List[SelectionMetrics]] = {k: [] for k in kinds}
    out["random"] = []
    out["oracle"] = []
    for i, item in enumerate(items):
        for kind in kinds:
            x = representation_matrix([item], kind)[0]
            out[kind].append(selection_metrics(item.perf, predictors[kind].predict(x)))
        out["random"].append(random_selection_metrics(item.perf, cfg.random_baseline_draws, seed=1000+i))
        out["oracle"].append(SelectionMetrics(1.0, 1.0, 1.0, 0.0))
    return out


def summarize_eval(results: Mapping[str, Sequence[SelectionMetrics]], ours: str = "gamma") -> Dict[str, Any]:
    return {
        "metrics": {k: metrics_mean_std(v) for k, v in results.items()},
        "paired_regret_p_less": paired_regret_tests(results, ours=ours),
    }


def run_e2(cfg: ExperimentConfig, out_dir: Path, seed: int = 20, resume: bool = True) -> Dict[str, Any]:
    """Run E2. Each of the three synthetic corpora (train/test/mixture) is
    checkpointed to its own JSON file under ``out_dir`` (train_progress.json /
    test_progress.json / mixture_progress.json). If a prior run of this same
    ``--out`` was interrupted partway through, rerunning with ``resume=True``
    (the default; ``--fresh`` on the CLI sets ``resume=False``) picks back up
    at the next un-checkpointed (mechanism, replicate) slot instead of
    redoing already-finished ones -- see build_synthetic_corpus/
    build_mixture_corpus for exactly what is/isn't recomputed on resume.
    """
    _require_pyg()
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = PerformanceCache(Path(cfg.cache_dir) / "architecture_perf.json")
    native_templates, task_templates = fit_dictionaries(cfg)

    train = build_synthetic_corpus(
        "train", cfg.train_problems_per_mechanism, native_templates, task_templates,
        cfg, cache, seed=seed,
        checkpoint_path=out_dir / "train_progress.json", resume=resume,
    )
    test = build_synthetic_corpus(
        "test", cfg.test_problems_per_mechanism, native_templates, task_templates,
        cfg, cache, seed=seed+1,
        checkpoint_path=out_dir / "test_progress.json", resume=resume,
    )
    mixtures = build_mixture_corpus(
        cfg.mixture_problems, native_templates, task_templates, cfg, cache, seed=seed+2,
        checkpoint_path=out_dir / "mixture_progress.json", resume=resume,
    )
    predictors = fit_representation_predictors(train, cfg)

    kinds = ("conventional", "raw", "alpha", "gamma")
    test_results = evaluate_predictors(test, predictors, cfg, kinds)
    mixture_results = evaluate_predictors(mixtures, predictors, cfg, ("conventional", "gamma"))

    state = {
        "cfg": cfg,
        "architecture_names": active_architectures(cfg),
        "native_templates": native_templates,
        "task_templates": task_templates,
        "predictors": predictors,
        "train": train,
    }
    with open(out_dir / "e2_state.pkl", "wb") as f:
        pickle.dump(state, f)

    summary = {
        "architecture_names": list(active_architectures(cfg)),
        "heldout_generator": summarize_eval(test_results),
        "heldout_mixtures": summarize_eval(mixture_results),
    }
    (out_dir / "e2_results.json").write_text(json.dumps(summary, indent=2))
    return {"state": state, "summary": summary, "test": test, "mixtures": mixtures}


# =============================================================================
# Real datasets for E3/E4
# =============================================================================

REAL_DATASET_SPECS = [
    ("planetoid", "Cora", "accuracy"),
    ("planetoid", "CiteSeer", "accuracy"),
    ("planetoid", "PubMed", "accuracy"),
    ("heterophilous", "Roman-empire", "accuracy"),
    ("heterophilous", "Amazon-ratings", "accuracy"),
    ("heterophilous", "Minesweeper", "roc_auc"),
    ("heterophilous", "Tolokers", "roc_auc"),
    ("heterophilous", "Questions", "roc_auc"),
    ("amazon", "Photo", "accuracy"),
    ("amazon", "Computers", "accuracy"),
    ("coauthor", "CS", "accuracy"),
    # struc2vec-style role-recovery benchmarks (Ribeiro et al. 2017). Node
    # "features" as shipped by PyG are a one-hot node-identity matrix (no
    # real attributes) -- see load_real_problem's Airports branch below for
    # what that means for the native/feature-alignment channel.
    ("airports", "USA", "accuracy"),
    ("airports", "Brazil", "accuracy"),
    ("airports", "Europe", "accuracy"),
    # WebKB (Pei et al. 2020 / Craven et al. hyperlink webpage-classification
    # graphs): small, heavily heterophilous, directed hyperlink graphs
    # (symmetrized like every other source -- see sanitize_edge_index above),
    # sparse bag-of-words page features, 5 classes (course/faculty/student/
    # project/staff), badly class-imbalanced. Re-split 60/20/20 same as
    # everything else (stratified_masks handles the tiny classes fine).
    ("webkb", "Texas", "accuracy"),
    ("webkb", "Cornell", "accuracy"),
    ("webkb", "Wisconsin", "accuracy"),
    # Actor: same Geom-GCN paper as WebKB above (film/actor co-occurrence
    # network); see load_real_problem's Actor branch.
    ("actor", "Actor", "accuracy"),
    # BlogCatalog / Flickr: small attributed social networks (Wang et al.,
    # "Scaling Attributed Network Embedding to Massive Graphs"); see
    # load_real_problem's attributed branch, including the multi-label ->
    # single-label collapse caveat if it applies to these two names.
    ("attributed", "BlogCatalog", "accuracy"),
    ("attributed", "Flickr", "accuracy"),
    # Chameleon-Filtered / Squirrel-filtered: de-duplicated/leakage-fixed
    # versions of the original geom-gcn Chameleon/Squirrel (Platonov et al.
    # 2023, same paper/repo as the 5 "heterophilous" datasets above); see
    # load_real_problem's heterophilous_filtered branch.
    ("heterophilous_filtered", "Chameleon-Filtered", "accuracy"),
    ("heterophilous_filtered", "Squirrel-filtered", "accuracy"),
    ("ogb", "ogbn-arxiv", "accuracy"),
]


def _mask_np(mask: torch.Tensor) -> np.ndarray:
    return mask.detach().cpu().numpy().astype(bool)


def load_real_problem(source: str, name: str, metric: str,
                      cfg: ExperimentConfig, split_id: int = 0) -> Problem:
    _require_pyg()
    root = Path(cfg.data_root)
    if source == "planetoid":
        from torch_geometric.datasets import Planetoid
        data = Planetoid(root=str(root / "Planetoid" / name), name=name)[0]
    elif source == "heterophilous":
        from torch_geometric.datasets import HeterophilousGraphDataset
        data = HeterophilousGraphDataset(root=str(root / "Heterophilous" / name), name=name)[0]
    elif source == "amazon":
        from torch_geometric.datasets import Amazon
        data = Amazon(root=str(root / "Amazon" / name), name=name)[0]
    elif source == "coauthor":
        from torch_geometric.datasets import Coauthor
        data = Coauthor(root=str(root / "Coauthor" / name), name=name)[0]
    elif source == "airports":
        # USA / Brazil / Europe airport-traffic graphs (struc2vec). Ships
        # x = identity(n) (a one-hot node id, not a real attribute vector)
        # and no train/val/test masks, so this always falls through to the
        # stratified 60/20/20 split below. The raw edge list is directed
        # (one row per edge, no reverse row); sanitize_edge_index's
        # to_undirected() below symmetrizes it, same as every other source.
        from torch_geometric.datasets import Airports
        data = Airports(root=str(root / "Airports" / name), name=name)[0]
    elif source == "webkb":
        # Texas / Cornell / Wisconsin. Ships x = sparse bag-of-words page
        # features (real attributes, unlike Airports) and directed
        # hyperlink edges; sanitize_edge_index below symmetrizes them, and
        # the shipped 10-fold train/val/test masks are ignored in favor of
        # the repo-wide stratified 60/20/20 split, same as every other
        # non-ogb source.
        from torch_geometric.datasets import WebKB
        data = WebKB(root=str(root / "WebKB" / name), name=name)[0]
    elif source == "actor":
        # Film/actor co-occurrence network, same Geom-GCN paper/split style
        # as WebKB above: nodes are actors, edges are co-occurrence on a
        # Wikipedia page, x = sparse bag-of-words features (932-dim), 5
        # classes. Ships 10 pre-defined splits; ignored in favor of the
        # repo-wide stratified 60/20/20 split, same as every other source.
        from torch_geometric.datasets import Actor
        data = Actor(root=str(root / "Actor"))[0]
    elif source == "attributed":
        # BlogCatalog / Flickr: small attributed social networks from Wang
        # et al., "Scaling Attributed Network Embedding to Massive Graphs"
        # (the same PyG loader also serves Wiki/PPI/Facebook/etc., but only
        # these two names are wired into REAL_DATASET_SPECS below).
        # Downloaded from a Google Drive mirror -- if that's unreachable or
        # rate-limited, this raises before touching anything else.
        #
        # Some datasets in this family ship multi-label y (multi-hot, e.g.
        # PPI); this repo's Problem/metric machinery is single-label only
        # (see Problem.y_dict / n_classes above), so a multi-label y here is
        # collapsed to argmax-over-labels as a pragmatic single-label
        # stand-in, with a printed count of how many nodes had 0 or >1
        # label (i.e. how lossy the collapse is). Treat results on such a
        # dataset as approximate until that collapse is reviewed.
        from torch_geometric.datasets import AttributedGraphDataset
        data = AttributedGraphDataset(root=str(root / "Attributed"), name=name)[0]
        if data.x.layout != torch.strided:
            data.x = data.x.to_dense()
        if data.y.dim() == 2:
            counts = data.y.sum(dim=1)
            n_multi = int((counts > 1).sum())
            n_zero = int((counts == 0).sum())
            print(
                f"[attributed] {name}: multi-label y collapsed to argmax "
                f"({n_multi} nodes with >1 label, {n_zero} with 0 labels, "
                f"of {data.y.size(0)})"
            )
            data.y = data.y.argmax(dim=1)
    elif source == "heterophilous_filtered":
        # Chameleon-Filtered / Squirrel-filtered: the de-duplicated /
        # leakage-fixed versions of the original geom-gcn Chameleon/Squirrel
        # (Platonov et al. 2023, "A Critical Look at the Evaluation of GNNs
        # under Heterophily" -- same paper/repo/npz format as the 5
        # "heterophilous" datasets above), just not in PyG's
        # HeterophilousGraphDataset name whitelist, so downloaded and parsed
        # here directly instead of going through that class. Like every
        # other non-ogb source, no shipped split is used -- the repo-wide
        # stratified 60/20/20 split applies below.
        from torch_geometric.data import download_url
        fname = name.lower().replace("-", "_").replace(" ", "_")
        raw_dir = root / "HeterophilousFiltered" / name / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        raw_path = raw_dir / f"{fname}.npz"
        if not raw_path.exists():
            download_url(
                f"https://github.com/yandex-research/heterophilous-graphs/raw/main/data/{fname}.npz",
                str(raw_dir),
            )
        raw = np.load(raw_path, "r")
        data = Data(
            x=torch.from_numpy(raw["node_features"]),
            y=torch.from_numpy(raw["node_labels"]),
            edge_index=torch.from_numpy(raw["edges"]).t().contiguous(),
        )
    elif source == "ogb":
        try:
            from ogb.nodeproppred import PygNodePropPredDataset
        except Exception as e:
            raise ImportError("Install `ogb` for ogbn-arxiv") from e
        ds = PygNodePropPredDataset(name=name, root=str(root / "OGB"))
        data = ds[0]
        split = ds.get_idx_split()
        n = data.num_nodes
        tr = np.zeros(n, dtype=bool); tr[split["train"].cpu().numpy()] = True
        va = np.zeros(n, dtype=bool); va[split["valid"].cpu().numpy()] = True
        te = np.zeros(n, dtype=bool); te[split["test"].cpu().numpy()] = True
        data.train_mask = torch.tensor(tr)
        data.val_mask = torch.tensor(va)
        data.test_mask = torch.tensor(te)
        data.y = data.y.view(-1)
    else:
        raise ValueError(source)

    n = data.num_nodes
    data.edge_index = sanitize_edge_index(data.edge_index, n)
    y = data.y.view(-1).cpu().numpy().astype(int)

    if source == "ogb" and hasattr(data, "train_mask") and data.train_mask is not None:
        # OGB's split is the point of using OGB: it is the leaderboard-
        # standard (temporal, non-random) protocol, so it is honored as-is
        # rather than randomized. Every other source ignores whatever split
        # it ships with (if any) and gets a fresh stratified split instead --
        # see the comment above REAL_DATASET_SPECS / stratified_masks.
        trm, vam, tem = data.train_mask, data.val_mask, data.test_mask
        if trm.ndim == 2:
            col = split_id % trm.shape[1]
            tr, va, te = _mask_np(trm[:, col]), _mask_np(vam[:, col]), _mask_np(tem[:, col])
        else:
            tr, va, te = _mask_np(trm), _mask_np(vam), _mask_np(tem)
    else:
        # Repo-wide default: class-stratified 60/20/20 (stratified_masks'
        # own train_frac=0.6/val_frac=0.2 defaults), not any split a loader
        # happens to ship with.
        tr, va, te = stratified_masks(y, seed=cfg.real_split_seed + split_id)

    G = nx_graph_from_edge_index(data.edge_index, n)
    X = data.x.cpu().numpy().astype(np.float32)
    return Problem(name=name, G=G, X=X, y=y, train_mask=tr, val_mask=va, test_mask=te, metric=metric)


def build_real_corpus(native_templates: np.ndarray, task_templates: np.ndarray,
                      cfg: ExperimentConfig, cache: PerformanceCache,
                      specs: Sequence[Tuple[str, str, str]] = REAL_DATASET_SPECS,
                      seed: int = 30,
                      checkpoint_path: Optional[str | Path] = None,
                      resume: bool = True) -> List[CorpusItem]:
    """Build the real-dataset corpus (E3/E4).

    If ``checkpoint_path`` is given, each dataset is checkpointed under its
    own name after it finishes (representations + true architecture
    performance). On a later call with the same path (and ``resume=True``,
    the default), a dataset already present in the checkpoint is loaded from
    disk instead of recomputed: ``load_real_problem`` still runs for it (a
    cheap disk read once PyG has the processed dataset cached -- no network
    round-trip, no re-download), but representation extraction and the full
    architecture-training sweep are skipped. This is the key piece for "ran
    11/17 datasets, process died, pick up the other 6": rerun with the same
    ``--out`` and the 11 finished ones are skipped entirely rather than
    redownloaded/re-embedded/retrained from scratch.

    Note this is on top of, not instead of, PerformanceCache: even a
    from-scratch (no checkpoint) rerun already gets cached architecture
    training per (dataset, arch, seed) via ``cache``. What the checkpoint
    additionally buys is skipping representation extraction and every
    disk/network step for datasets that are already fully done, and -- more
    importantly -- guaranteeing that a crash on dataset i doesn't cost you
    the i-1 datasets that already finished, which the previous
    all-or-nothing (only write e3_results.json at the very end) behavior did.
    """
    ckpt = JsonCheckpoint(checkpoint_path, resume=resume) if checkpoint_path is not None else None
    items = []
    for i, (source, name, metric) in enumerate(specs):
        p = load_real_problem(source, name, metric, cfg, split_id=seed+i)
        fp = _problem_fingerprint(p)

        saved = ckpt.load(name) if ckpt is not None else None
        if saved is not None and saved.get("fingerprint") == fp:
            print(f"[real] {name} (resumed from checkpoint)")
            items.append(_corpus_item_from_checkpoint(p, saved))
            continue
        if saved is not None:
            warnings.warn(
                f"[real] checkpoint fingerprint mismatch for {name} "
                "(cfg/split/data changed since it was written?); recomputing"
            )

        print(f"[real] {name}")
        y_train = p.y_dict(p.train_mask)  # CRITICAL: no validation/test labels
        rep = extract_representations(p, native_templates, task_templates, cfg, seed=seed+i, y_observed=y_train)
        conv = conventional_statistics(p, y_train)
        perf, perf_std = performance_vector(p, rep.role_signature, cfg, cache)
        item = CorpusItem(p, rep, conv, perf, perf_std, mechanism="real")
        items.append(item)
        if ckpt is not None:
            payload = _corpus_item_checkpoint(item)
            payload["fingerprint"] = fp
            ckpt.save(name, payload)
    return items


# =============================================================================
# E3: frozen synthetic -> real transfer
# =============================================================================


def leave_one_real_out_reference(real: Sequence[CorpusItem], cfg: ExperimentConfig,
                                 kind: str = "gamma") -> List[SelectionMetrics]:
    """Upper reference: meta-predictor trained directly on other real datasets."""
    out = []
    for i in range(len(real)):
        tr = [x for j, x in enumerate(real) if j != i]
        te = real[i]
        pred = PerformancePredictor(cfg.meta_ridge_alpha).fit(
            representation_matrix(tr, kind), performance_matrix(tr)
        )
        out.append(selection_metrics(te.perf, pred.predict(representation_matrix([te], kind)[0])))
    return out


def run_e3(e2_state: Dict[str, Any], cfg: ExperimentConfig, out_dir: Path,
           specs: Sequence[Tuple[str, str, str]] = REAL_DATASET_SPECS,
           seed: int = 30, resume: bool = True) -> Dict[str, Any]:
    """Run E3. The real-dataset corpus is checkpointed to
    ``out_dir/real_progress.json`` as each dataset finishes (see
    build_real_corpus). If this run is interrupted after N/len(specs)
    datasets, rerunning with the same ``--out`` (``resume=True``, the
    default; ``--fresh`` on the CLI disables it) skips the N already-done
    datasets and continues with the rest -- ``e3_results.json`` itself is
    still only written once, at the end, with the *complete* set of
    datasets, so its schema/consumers (e.g. existing plotting/table code)
    are unchanged; real_progress.json is the resumable checkpoint, not a
    replacement for e3_results.json.
    """
    _require_pyg()
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = PerformanceCache(Path(cfg.cache_dir) / "architecture_perf.json")
    native = e2_state["native_templates"]
    task = e2_state["task_templates"]
    predictors = e2_state["predictors"]  # FROZEN synthetic-trained predictors
    real = build_real_corpus(
        native, task, cfg, cache, specs=specs, seed=seed,
        checkpoint_path=out_dir / "real_progress.json", resume=resume,
    )

    kinds = ("conventional", "raw", "alpha", "gamma")
    results = evaluate_predictors(real, predictors, cfg, kinds)
    results["real_fit_reference"] = leave_one_real_out_reference(real, cfg, "gamma")
    summary = summarize_eval(results)

    per_dataset = {}
    for i, item in enumerate(real):
        per_dataset[item.problem.name] = {
            kind: asdict(results[kind][i]) for kind in results
        }
        per_dataset[item.problem.name]["true_performance"] = item.perf.tolist()
        per_dataset[item.problem.name]["true_performance_std"] = item.perf_std.tolist()

    payload = {
        "architecture_names": list(active_architectures(cfg)),
        "aggregate": summary,
        "per_dataset": per_dataset,
    }
    (out_dir / "e3_results.json").write_text(json.dumps(payload, indent=2))
    with open(out_dir / "e3_real_corpus.pkl", "wb") as f:
        pickle.dump(real, f)
    return {"summary": summary, "real": real, "results": results}


# =============================================================================
# E4a: raw fields vs PCA vs signed-NMF vs grammar
# =============================================================================

class SignedNMFCompressor:
    """NMF baseline that preserves sign by splitting positive/negative channels."""
    def __init__(self, n_components: int, seed: int):
        self.model = NMF(n_components=n_components, init="nndsvda", random_state=seed, max_iter=1500)

    @staticmethod
    def lift(X: np.ndarray) -> np.ndarray:
        return np.concatenate([np.maximum(X, 0), np.maximum(-X, 0)], axis=1)

    def fit(self, X: np.ndarray) -> "SignedNMFCompressor":
        self.model.fit(self.lift(X))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.model.transform(self.lift(X))


def run_e4_representation(e2_state: Dict[str, Any], real: Sequence[CorpusItem],
                          cfg: ExperimentConfig, out_dir: Path,
                          seed: int = 40) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    train = e2_state["train"]
    Y = performance_matrix(train)
    Xtr_raw = representation_matrix(train, "raw")
    Xre_raw = representation_matrix(real, "raw")
    dim = representation_matrix(train, "gamma").shape[1]
    dim = min(dim, Xtr_raw.shape[0] - 1, Xtr_raw.shape[1])
    if dim < 2:
        raise ValueError("Need more synthetic training problems for PCA/NMF ablation")

    pca = PCA(n_components=dim, random_state=seed).fit(Xtr_raw)
    nmf = SignedNMFCompressor(dim, seed).fit(Xtr_raw)

    reps_train = {
        "raw": Xtr_raw,
        "pca": pca.transform(Xtr_raw),
        "nmf": nmf.transform(Xtr_raw),
        "grammar": representation_matrix(train, "gamma"),
    }
    reps_real = {
        "raw": Xre_raw,
        "pca": pca.transform(Xre_raw),
        "nmf": nmf.transform(Xre_raw),
        "grammar": representation_matrix(real, "gamma"),
    }

    results: Dict[str, List[SelectionMetrics]] = {}
    for kind in reps_train:
        f = PerformancePredictor(cfg.meta_ridge_alpha).fit(reps_train[kind], Y)
        results[kind] = [
            selection_metrics(item.perf, f.predict(reps_real[kind][i]))
            for i, item in enumerate(real)
        ]

    payload = {
        "dimensions": {k: int(v.shape[1]) for k, v in reps_real.items()},
        "metrics": {k: metrics_mean_std(v) for k, v in results.items()},
        "paired_regret_p_less": paired_regret_tests(results, ours="grammar"),
        "pca_explained_variance": pca.explained_variance_ratio_.tolist(),
    }
    (out_dir / "e4a_results.json").write_text(json.dumps(payload, indent=2))
    return payload


# =============================================================================
# E4c: 5D native-field ablation (unsupervised compression vs. the alpha
# dictionary), matched at alpha's native dimensionality (5D) rather than at
# gamma's combined dimensionality (9D) as in E4a. This isolates whether the
# mechanism-aware NNLS dictionary Phi does anything a generic 5D compression
# of the same raw native field A_SX could not, independent of any
# task-conditioned (beta) signal.
# =============================================================================


def run_e4_native_ablation(e2_state: Dict[str, Any], real: Sequence[CorpusItem],
                           cfg: ExperimentConfig, out_dir: Path,
                           seed: int = 45) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    train = e2_state["train"]
    Y = performance_matrix(train)
    Xtr_native = representation_matrix(train, "raw_native")
    Xre_native = representation_matrix(real, "raw_native")
    dim = representation_matrix(train, "alpha").shape[1]
    dim = min(dim, Xtr_native.shape[0] - 1, Xtr_native.shape[1])
    if dim < 2:
        raise ValueError("Need more synthetic training problems for the native PCA/NMF ablation")

    pca = PCA(n_components=dim, random_state=seed).fit(Xtr_native)
    nmf = SignedNMFCompressor(dim, seed).fit(Xtr_native)

    reps_train = {
        "raw_native": Xtr_native,
        "pca_native": pca.transform(Xtr_native),
        "nmf_native": nmf.transform(Xtr_native),
        "alpha": representation_matrix(train, "alpha"),
    }
    reps_real = {
        "raw_native": Xre_native,
        "pca_native": pca.transform(Xre_native),
        "nmf_native": nmf.transform(Xre_native),
        "alpha": representation_matrix(real, "alpha"),
    }

    results: Dict[str, List[SelectionMetrics]] = {}
    for kind in reps_train:
        f = PerformancePredictor(cfg.meta_ridge_alpha).fit(reps_train[kind], Y)
        results[kind] = [
            selection_metrics(item.perf, f.predict(reps_real[kind][i]))
            for i, item in enumerate(real)
        ]

    payload = {
        "dimensions": {k: int(v.shape[1]) for k, v in reps_real.items()},
        "metrics": {k: metrics_mean_std(v) for k, v in results.items()},
        "paired_regret_p_less": paired_regret_tests(results, ours="alpha"),
        "pca_explained_variance": pca.explained_variance_ratio_.tolist(),
    }
    (out_dir / "e4c_results.json").write_text(json.dumps(payload, indent=2))
    return payload


# =============================================================================
# E4b: 20 labels/class, same supervision for all label-aware representations
# =============================================================================


def limited_label_dict(problem: Problem, labels_per_class: int, seed: int) -> Dict[int, int]:
    """Sample only from the official/TRAIN pool, never val/test."""
    rng = np.random.default_rng(seed)
    out: Dict[int, int] = {}
    train_idx = np.flatnonzero(problem.train_mask)
    for c in np.unique(problem.y):
        idx = train_idx[problem.y[train_idx] == c]
        if len(idx) == 0:
            continue
        take = min(labels_per_class, len(idx))
        for v in rng.choice(idx, take, replace=False):
            out[int(v)] = int(c)
    return out


def rebuild_with_label_budget(items: Sequence[CorpusItem], native_templates: np.ndarray,
                              task_templates: np.ndarray, cfg: ExperimentConfig,
                              labels_per_class: int, seed: int) -> List[CorpusItem]:
    out = []
    for i, item in enumerate(items):
        p = item.problem
        y_lim = limited_label_dict(p, labels_per_class, seed+i)
        rep = extract_representations(p, native_templates, task_templates, cfg, seed+i, y_observed=y_lim)
        conv = conventional_statistics(p, y_lim)
        out.append(CorpusItem(p, rep, conv, item.perf, item.perf_std, item.mechanism))
    return out


def run_e4_labels(e2_state: Dict[str, Any], real: Sequence[CorpusItem],
                  cfg: ExperimentConfig, out_dir: Path,
                  seed: int = 50) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    native, task = e2_state["native_templates"], e2_state["task_templates"]
    # Rebuild BOTH synthetic meta-training problems and real targets with the
    # identical low-label budget. Performance targets remain unchanged.
    train_lim = rebuild_with_label_budget(
        e2_state["train"], native, task, cfg, cfg.low_labels_per_class, seed
    )
    real_lim = rebuild_with_label_budget(
        real, native, task, cfg, cfg.low_labels_per_class, seed + 10000
    )
    Y = performance_matrix(train_lim)

    # (1) alpha only (label-free)
    # (2) conventional label-aware summaries at the same budget
    # (3) direct low-capacity selector on RAW task fields (same labels, no gamma)
    # (4) alpha+beta grammar
    kinds = {
        "alpha_only": "alpha",
        "conventional_label_aware": "conventional",
        "direct_task_field_selector": "task_raw",
        "alpha_plus_beta": "gamma",
    }
    results: Dict[str, List[SelectionMetrics]] = {}
    for display, kind in kinds.items():
        f = PerformancePredictor(cfg.meta_ridge_alpha).fit(representation_matrix(train_lim, kind), Y)
        results[display] = [
            selection_metrics(item.perf, f.predict(representation_matrix([item], kind)[0]))
            for item in real_lim
        ]

    payload = {
        "labels_per_class": cfg.low_labels_per_class,
        "metrics": {k: metrics_mean_std(v) for k, v in results.items()},
        "paired_regret_p_less": paired_regret_tests(results, ours="alpha_plus_beta"),
    }
    (out_dir / "e4b_results.json").write_text(json.dumps(payload, indent=2))
    return payload


# =============================================================================
# CLI / orchestration
# =============================================================================


def load_e2_state(path: str | Path) -> Dict[str, Any]:
    with open(path, "rb") as f:
        return pickle.load(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", choices=["e1", "e2", "e3", "e4", "all"], default="all")
    ap.add_argument("--out", type=str, default="./results")
    ap.add_argument("--data-root", type=str, default="./data")
    ap.add_argument("--cache-dir", type=str, default="./cache")
    ap.add_argument("--e2-state", type=str, default=None)
    ap.add_argument("--quick", action="store_true", help="Smoke test only; never report these numbers")
    ap.add_argument("--skip-ogb", action="store_true", help="Useful while optimizing the role-signature path")
    ap.add_argument(
        "--no-gt", action="store_true",
        help="Use the original five-architecture bank (robustness analysis); default includes Graph Transformer",
    )
    ap.add_argument(
        "--fresh", action="store_true",
        help=(
            "Ignore any existing *_progress.json checkpoints under --out and recompute "
            "every E2/E3 corpus item from scratch (still writes fresh checkpoints as it "
            "goes). Default is to resume: skip items already checkpointed from a prior, "
            "possibly-interrupted run with this same --out."
        ),
    )
    args = ap.parse_args()

    cfg = ExperimentConfig.quick() if args.quick else ExperimentConfig()
    cfg.data_root = args.data_root
    cfg.cache_dir = args.cache_dir
    cfg.include_gt = not args.no_gt
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2))

    print(
        f"device={DEVICE}; quick={args.quick}; "
        f"architectures={','.join(active_architectures(cfg))}"
    )
    if args.quick:
        warnings.warn("QUICK MODE is only a pipeline smoke test. Do not report these numbers.")

    e2_state = load_e2_state(args.e2_state) if args.e2_state else None
    if e2_state is not None:
        state_arch = tuple(e2_state.get("architecture_names", ()))
        requested_arch = active_architectures(cfg)
        if state_arch and state_arch != requested_arch:
            raise ValueError(
                "Architecture-bank mismatch: the loaded E2 state was fit for "
                f"{state_arch}, but this run requests {requested_arch}. "
                "Use --no-gt consistently for the five-model analysis, or load "
                "the six-model E2 state for the primary GT analysis."
            )
    e2_result = None
    e3_result = None

    if args.exp in {"e2", "all"}:
        print("\n=== E2: cross-generator A->B ===")
        e2_result = run_e2(cfg, out / "e2", resume=not args.fresh)
        e2_state = e2_result["state"]

    if e2_state is None and args.exp in {"e1", "e3", "e4"}:
        raise ValueError("E1 counterfactual prediction / E3 / E4 require --e2-state, or run --exp all")

    specs = REAL_DATASET_SPECS
    if args.skip_ogb:
        specs = [s for s in specs if s[1] != "ogbn-arxiv"]

    if args.exp in {"e3", "all"}:
        print("\n=== E3: frozen synthetic -> real ===")
        e3_result = run_e3(e2_state, cfg, out / "e3", specs=specs, resume=not args.fresh)

    if args.exp in {"e1", "all"}:
        print("\n=== E1: matched-summary diagnosis ===")
        if e2_state is None:
            raise RuntimeError("Need E2 state for the counterfactual predicted-best payload")
        cache = PerformanceCache(Path(cfg.cache_dir) / "architecture_perf.json")
        run_e1(
            e2_state["native_templates"], e2_state["task_templates"], cfg,
            out / "e1", e2_state=e2_state, cache=cache,
        )

    if args.exp in {"e4", "all"}:
        print("\n=== E4: representation + label ablations ===")
        if e3_result is None:
            # Load/build real corpus once if E3 was not run in this invocation.
            # Reuses E3's checkpoint file/convention (out/e3/real_progress.json)
            # so an E4-only run benefits from a prior E3 run's progress too.
            cache = PerformanceCache(Path(cfg.cache_dir) / "architecture_perf.json")
            real = build_real_corpus(
                e2_state["native_templates"], e2_state["task_templates"],
                cfg, cache, specs=specs, seed=30,
                checkpoint_path=out / "e3" / "real_progress.json", resume=not args.fresh,
            )
        else:
            real = e3_result["real"]
        run_e4_representation(e2_state, real, cfg, out / "e4")
        run_e4_native_ablation(e2_state, real, cfg, out / "e4")
        run_e4_labels(e2_state, real, cfg, out / "e4")

    print("\nDone.")


if __name__ == "__main__":
    main()
