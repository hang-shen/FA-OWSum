"""Final locally adaptive FA-OWSum prediction package."""

from .final_model import fit, make_fingerprints, make_similarity_caches, predict
from .model import ModelConfig

__all__ = [
    "ModelConfig",
    "fit",
    "make_fingerprints",
    "make_similarity_caches",
    "predict",
]
