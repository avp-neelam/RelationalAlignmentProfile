"""Public package interface for the Graph Grammar experiments.

The original scripts remain supported as command-line entry points.  The
package modules provide smaller, discoverable import surfaces for notebooks,
tests, and future refactoring.
"""

__all__ = [
    "ArchConfig", "ExperimentConfig", "Problem",
    "compute_geometric_grammar", "fit_native_templates_joint", "fit_task_templates",
    "compute_alignment_curves", "structural_alignment_curve", "task_alignment_curve",
]


def __getattr__(name):
    """Load optional, dependency-heavy submodules only when requested."""
    if name in {"ArchConfig", "ExperimentConfig"}:
        from .config import ArchConfig, ExperimentConfig
        return {"ArchConfig": ArchConfig, "ExperimentConfig": ExperimentConfig}[name]
    if name == "Problem":
        from .problems import Problem
        return Problem
    if name in {"compute_geometric_grammar", "fit_native_templates_joint", "fit_task_templates"}:
        from .grammar import compute_geometric_grammar, fit_native_templates_joint, fit_task_templates
        return {
            "compute_geometric_grammar": compute_geometric_grammar,
            "fit_native_templates_joint": fit_native_templates_joint,
            "fit_task_templates": fit_task_templates,
        }[name]
    if name in {"compute_alignment_curves", "structural_alignment_curve", "task_alignment_curve"}:
        from .curves import compute_alignment_curves, structural_alignment_curve, task_alignment_curve
        return {
            "compute_alignment_curves": compute_alignment_curves,
            "structural_alignment_curve": structural_alignment_curve,
            "task_alignment_curve": task_alignment_curve,
        }[name]
    raise AttributeError(name)
