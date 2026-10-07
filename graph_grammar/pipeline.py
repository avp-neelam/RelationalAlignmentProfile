"""Composable experiment-stage entry points.

Each function delegates to the tested reference stage.  Keeping this thin
layer separate makes it possible to replace a stage without changing callers.
"""

from utils.experiments.runner import (
    build_mixture_corpus,
    build_real_corpus,
    build_synthetic_corpus,
    fit_dictionaries,
    run_e1,
    run_e2,
    run_e3,
    run_e4_labels,
    run_e4_representation,
)

__all__ = [
    "fit_dictionaries", "build_synthetic_corpus", "build_mixture_corpus",
    "build_real_corpus", "run_e1", "run_e2", "run_e3",
    "run_e4_labels", "run_e4_representation",
]
