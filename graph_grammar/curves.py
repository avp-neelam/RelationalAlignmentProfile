"""Relational-alignment-curve API (rho_align(t), rho_task(t)).

Companion to grammar.py: that module exposes the compressed gamma
coordinate (alpha, beta); this one exposes the dictionary-free curves that
sit alongside it. Same convention -- the implementation lives in
utils.grammar.curves, this is the preferred import location for new
analysis code.
"""

from utils.grammar.curves import (
    DEFAULT_T_GRID,
    AlignmentCurve,
    RAPCurves,
    compute_alignment_curves,
    plot_alignment_curve,
    structural_alignment_curve,
    task_alignment_curve,
)

__all__ = [
    "DEFAULT_T_GRID", "AlignmentCurve", "RAPCurves",
    "structural_alignment_curve", "task_alignment_curve",
    "compute_alignment_curves", "plot_alignment_curve",
]
