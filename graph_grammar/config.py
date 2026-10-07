"""Experiment configuration and architecture settings.

The dataclasses are re-exported from the reference implementation for now so
existing serialized E2 state and command-line behavior remain compatible.
They are isolated behind this module as the first step toward making config a
dependency of the individual pipeline stages rather than the whole script.
"""

from utils.experiments.runner import ArchConfig, ExperimentConfig

__all__ = ["ArchConfig", "ExperimentConfig"]
