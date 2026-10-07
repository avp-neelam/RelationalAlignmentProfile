"""Composable entry points for the E1--E4 experiment stages."""

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
