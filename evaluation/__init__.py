"""
This package contains modules for evaluating the latent space of neural network models,
including extracting embeddings, applying dimensionality reduction, computing metrics,
and visualizing results.

Modules:
    - latent_space_evaluator: Contains the LatentSpaceEvaluator class for evaluating latent space.
    - get_predictions: Contains functions for obtaining model predictions.
    - save_results: Contains the ResultsSaver class for saving classification results.
"""

from .get_predictions import get_predictions
from .latent_space_evaluator import LatentSpaceEvaluator
from .save_results import ResultsSaver

__all__ = [
    "get_predictions",
    "LatentSpaceEvaluator",
    "ResultsSaver",
]
