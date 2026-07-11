"""OnePass-STAMP: query-driven single-forward referring segmentation."""

from .model import OnePassOutput, OnePassQueryModule, OnePassSTAMP, onepass_mask_loss

__all__ = [
    "OnePassOutput",
    "OnePassQueryModule",
    "OnePassSTAMP",
    "onepass_mask_loss",
]
