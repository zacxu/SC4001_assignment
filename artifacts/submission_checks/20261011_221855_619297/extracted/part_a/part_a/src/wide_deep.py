"""Figure A: shared features, selected MLP, and one weighted linear branch."""

from __future__ import annotations

import math

from torch import nn

from part_a.src.model import HousingMLP


class WideDeepHousingModel(HousingMLP):
    def __init__(self, cardinalities, target_mean, target_std, *, hidden_width=256,
                 embedding_dimension=4, lambda_value=1.):
        if isinstance(lambda_value, bool) or not isinstance(lambda_value, (int, float)) or not math.isfinite(lambda_value) or lambda_value < 0:
            raise ValueError("Lambda must be a finite nonnegative fixed scalar")
        # Construct the whole baseline first: adding a branch must not change
        # the baseline's same-seed embeddings/MLP initialization.
        super().__init__(cardinalities, target_mean, target_std,
                         hidden_width=hidden_width, embedding_dimension=embedding_dimension)
        self.lambda_value = float(lambda_value)
        self.wide = nn.Linear(4 * embedding_dimension + 6, 1)

    def forward(self, categorical, continuous):
        features = self.encode_features(categorical, continuous)
        deep = self.mlp(features).squeeze(1)
        # Exact baseline control, with no gradients or Adam state for the
        # inactive branch. Avoid multiplying a possibly nonfinite value by 0.
        if self.lambda_value == 0:
            return deep
        return deep + self.lambda_value * self.wide(features).squeeze(1)

    def metadata(self):
        baseline = super().metadata()
        extra = baseline["input_width"] + 1
        return {**baseline, "architecture": "four_embedding_wide_and_deep",
                "lambda": self.lambda_value, "shared_embeddings": True,
                "wide_branch": "single_linear_output_in_standardized_price_space",
                "baseline_parameters": baseline["trainable_parameters"], "wide_parameters": extra,
                "trainable_parameters": baseline["trainable_parameters"] + extra,
                "active_parameters": baseline["trainable_parameters"] + (extra if self.lambda_value else 0),
                "prediction": "target_mean + target_std * (MLP(features) + lambda * Linear(features))"}

    @classmethod
    def from_metadata(cls, metadata, *, target_mean=0., target_std=1.):
        if metadata["architecture"] != "four_embedding_wide_and_deep":
            raise ValueError("Checkpoint architecture is not wide-and-deep")
        return cls(metadata["cardinalities"], target_mean, target_std,
                   hidden_width=metadata["hidden_width"], embedding_dimension=metadata["embedding_dimension"],
                   lambda_value=metadata["lambda"])
