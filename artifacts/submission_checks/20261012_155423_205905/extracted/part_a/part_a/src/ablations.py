"""A2 models: remove categorical inputs or replace the hidden ReLU with Sigmoid."""

from __future__ import annotations

import math

import torch
from torch import nn

from part_a.src.model import HousingMLP

VARIANTS = ("baseline", "continuous_only", "sigmoid")
VARIANT_LABELS = {"baseline": "Baseline (ReLU)", "continuous_only": "Continuous only",
                  "sigmoid": "All inputs (Sigmoid)"}


class SigmoidHousingMLP(HousingMLP):
    """The baseline's initialization and parameters are preserved exactly."""

    def __init__(self, cardinalities, target_mean, target_std, *, hidden_width=256, embedding_dimension=4):
        super().__init__(cardinalities, target_mean, target_std,
                         hidden_width=hidden_width, embedding_dimension=embedding_dimension)
        # Sigmoid has no parameters and consumes no initialization randomness.
        self.mlp[2] = nn.Sigmoid()

    def metadata(self):
        return {**super().metadata(), "architecture": "four_embedding_mlp_sigmoid",
                "activation": "Sigmoid", "ablation": "replace_only_hidden_relu"}


class ContinuousOnlyMLP(nn.Module):
    """Six continuous inputs, the same hidden width/LayerNorm, no embedding weights."""

    def __init__(self, target_mean, target_std, *, hidden_width=256):
        super().__init__()
        if type(hidden_width) is not int or hidden_width < 1:
            raise ValueError("Hidden width must be a positive integer")
        if not math.isfinite(target_mean) or not math.isfinite(target_std) or target_std <= 0:
            raise ValueError("Invalid target scaling")
        self.hidden_width = hidden_width
        self.mlp = nn.Sequential(nn.Linear(6, hidden_width), nn.LayerNorm(hidden_width),
                                 nn.ReLU(), nn.Linear(hidden_width, 1))
        self.target_mean, self.target_std = target_mean, target_std

    def forward(self, categorical: torch.Tensor, continuous: torch.Tensor) -> torch.Tensor:
        # Retain the common loader interface, but never read categorical values.
        return self.mlp(continuous).squeeze(1)

    def predict_price(self, categorical, continuous):
        return self(categorical, continuous) * self.target_std + self.target_mean

    def metadata(self):
        return {"architecture": "continuous_only_mlp", "hidden_width": self.hidden_width,
                "embedding_dimension": None, "cardinalities": [], "continuous_features": 6,
                "input_width": 6, "normalization": "LayerNorm", "activation": "ReLU",
                "output": "standardized_price", "trainable_parameters": 10 * self.hidden_width + 1,
                "ablation": "remove_all_embeddings_and_categorical_inputs"}


def build_a2_model(variant, cardinalities, target_mean, target_std, *, hidden_width, embedding_dimension):
    if variant == "continuous_only":
        return ContinuousOnlyMLP(target_mean, target_std, hidden_width=hidden_width)
    model_type = {"baseline": HousingMLP, "sigmoid": SigmoidHousingMLP}.get(variant)
    if model_type is None:
        raise ValueError(f"Unknown A2 variant: {variant}")
    return model_type(cardinalities, target_mean, target_std,
                      hidden_width=hidden_width, embedding_dimension=embedding_dimension)


def model_from_a2_metadata(metadata):
    architecture = metadata["architecture"]
    if architecture == "continuous_only_mlp":
        return ContinuousOnlyMLP(0., 1., hidden_width=metadata["hidden_width"])
    variants = {"four_embedding_mlp": "baseline", "four_embedding_mlp_sigmoid": "sigmoid"}
    if architecture not in variants:
        raise ValueError(f"Unsupported A2 architecture: {architecture}")
    return build_a2_model(variants[architecture], metadata["cardinalities"], 0., 1.,
                          hidden_width=metadata["hidden_width"], embedding_dimension=metadata["embedding_dimension"])
