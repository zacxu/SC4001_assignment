"""Independent implementation of the supplied four-embedding housing baseline."""

from __future__ import annotations

import math

import torch
from torch import nn


def parameter_count(cardinalities, hidden_width: int, embedding_dimension: int) -> int:
    if (len(cardinalities) != 4 or any(type(value) is not int or value < 1 for value in cardinalities)
            or type(hidden_width) is not int or hidden_width < 1
            or type(embedding_dimension) is not int or embedding_dimension < 1):
        raise ValueError("Use four positive cardinalities and positive integer model dimensions")
    return embedding_dimension * sum(cardinalities) + (4 * embedding_dimension + 10) * hidden_width + 1


class HousingMLP(nn.Module):
    """Four independent embeddings, Linear/LayerNorm/ReLU/Linear, raw standardized output."""

    def __init__(self, cardinalities, target_mean: float, target_std: float,
                 *, hidden_width: int = 64, embedding_dimension: int = 8):
        super().__init__()
        parameter_count(cardinalities, hidden_width, embedding_dimension)
        if not math.isfinite(target_mean) or not math.isfinite(target_std) or target_std <= 0:
            raise ValueError("Invalid target scaling")
        self.cardinalities = list(cardinalities)
        self.hidden_width = hidden_width
        self.embedding_dimension = embedding_dimension
        self.embeddings = nn.ModuleList(nn.Embedding(size, embedding_dimension) for size in cardinalities)
        input_width = 4 * embedding_dimension + 6
        self.mlp = nn.Sequential(nn.Linear(input_width, hidden_width), nn.LayerNorm(hidden_width),
                                 nn.ReLU(), nn.Linear(hidden_width, 1))
        self.target_mean, self.target_std = target_mean, target_std

    def encode_features(self, categorical: torch.Tensor, continuous: torch.Tensor) -> torch.Tensor:
        embedded = [embedding(categorical[:, index]) for index, embedding in enumerate(self.embeddings)]
        return torch.cat(embedded + [continuous], dim=1)

    def forward(self, categorical: torch.Tensor, continuous: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.encode_features(categorical, continuous)).squeeze(1)

    def predict_price(self, categorical: torch.Tensor, continuous: torch.Tensor) -> torch.Tensor:
        return self(categorical, continuous) * self.target_std + self.target_mean

    def metadata(self) -> dict:
        return {"architecture": "four_embedding_mlp", "hidden_width": self.hidden_width,
                "embedding_dimension": self.embedding_dimension, "cardinalities": self.cardinalities,
                "continuous_features": 6, "input_width": 4 * self.embedding_dimension + 6,
                "normalization": "LayerNorm", "activation": "ReLU", "output": "standardized_price",
                "trainable_parameters": parameter_count(self.cardinalities, self.hidden_width, self.embedding_dimension)}

    @classmethod
    def from_metadata(cls, metadata: dict, *, target_mean=0., target_std=1.) -> HousingMLP:
        if metadata["architecture"] != "four_embedding_mlp":
            raise ValueError("Checkpoint architecture is not the A1 baseline")
        return cls(metadata["cardinalities"], target_mean, target_std,
                   hidden_width=metadata["hidden_width"], embedding_dimension=metadata["embedding_dimension"])
