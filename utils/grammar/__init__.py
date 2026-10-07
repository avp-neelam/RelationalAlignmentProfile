"""Graph geometry, grammar estimation, and synthetic mechanisms."""

from .core import (
    GeometricGrammar,
    compute_geometric_grammar,
    fit_native_templates_joint,
    fit_task_templates,
    generate_mechanism_graph,
    generate_task_mechanism_graph,
)
from .curves import (
    AlignmentCurve,
    RAPCurves,
    compute_alignment_curves,
    structural_alignment_curve,
    task_alignment_curve,
)

__all__ = [
    "GeometricGrammar", "compute_geometric_grammar",
    "fit_native_templates_joint", "fit_task_templates",
    "generate_mechanism_graph", "generate_task_mechanism_graph",
    "AlignmentCurve", "RAPCurves", "compute_alignment_curves",
    "structural_alignment_curve", "task_alignment_curve",
]
