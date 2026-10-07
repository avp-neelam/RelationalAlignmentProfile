"""Geometric-grammar API.

This module intentionally exposes the geometry implementation independently of
the experiment runner.  It is the preferred import location for new analysis
code; the legacy module remains the implementation source until the numerical
sections are split into smaller files.
"""

from utils.grammar.core import (
    GeometricGrammar,
    compute_geometric_grammar,
    fit_native_templates_joint,
    fit_task_templates,
    generate_mechanism_graph,
    generate_task_mechanism_graph,
)

__all__ = [
    "GeometricGrammar", "compute_geometric_grammar",
    "fit_native_templates_joint", "fit_task_templates",
    "generate_mechanism_graph", "generate_task_mechanism_graph",
]
