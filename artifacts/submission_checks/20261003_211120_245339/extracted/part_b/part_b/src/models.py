"""The three compact CNNs drawn in assignment Figure B, not library presets."""

from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn


MODEL_NAMES = ("b1", "b2", "b3")
EXPECTED_PARAMETERS = {"b1": 99569, "b2": 18185, "b3": 77169}
PADDING_RULE = "symmetric_zero_padding_kernel_half_center_aligned_stride_two"


class ConvBNReLU(nn.Sequential):
    """Explicit odd-kernel padding keeps both residual paths on the same centers.

    Output size is ceil(input/stride). Kernel 3/stride 2 uses padding 1;
    the projection's kernel 1/stride 2 uses padding 0. Both sample centers
    0, 2, ... . This records the brief's otherwise unspecified SAME alignment.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1):
        if kernel_size not in (1, 3, 5) or stride not in (1, 2):
            raise ValueError("Figure B uses odd kernels 1/3/5 and strides 1/2")
        super().__init__(OrderedDict([
            ("conv", nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                               padding=kernel_size // 2, bias=False)),
            ("bn", nn.BatchNorm2d(out_channels)),
            ("relu", nn.ReLU()),
        ]))


class ResidualBlock(nn.Module):
    """ReLU follows both main convolutions, projection (if used), and addition."""

    def __init__(self, in_channels: int, out_channels: int, *, stride: int = 1):
        super().__init__()
        if (in_channels, out_channels, stride) not in ((16, 16, 1), (16, 32, 2), (32, 64, 2)):
            raise ValueError("Only the three Figure B3 residual blocks are supported")
        self.conv1 = ConvBNReLU(in_channels, out_channels, 3, stride)
        self.conv2 = ConvBNReLU(out_channels, out_channels, 3)
        self.shortcut = (nn.Identity() if stride == 1 else ConvBNReLU(in_channels, out_channels, 1, stride))
        self.add_relu = nn.ReLU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        main = self.conv2(self.conv1(inputs))
        shortcut = self.shortcut(inputs)
        if main.shape != shortcut.shape:
            raise RuntimeError(f"Residual shapes disagree: {main.shape} versus {shortcut.shape}")
        return self.add_relu(main + shortcut)


class XRayCNN(nn.Module):
    architecture: str

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or tuple(images.shape[1:]) != (1, 64, 64):
            raise ValueError("Models require normalized N x 1 x 64 x 64 input")
        return self.classifier(self.flatten(self.features(images))).squeeze(1)

    def metadata(self) -> dict:
        return {"format_version": 1, "architecture": self.architecture,
                "input_shape": [1, 64, 64], "convolution_bias": False,
                "fully_connected_bias": True, "convolution_postprocessing": ["BatchNorm2d", "ReLU"],
                "batchnorm_eps": 1e-5, "batchnorm_momentum": 0.1, "padding": PADDING_RULE,
                "residual_projection_activation": "ReLU" if self.architecture == "b3" else None,
                "residual_addition_activation": "ReLU" if self.architecture == "b3" else None,
                "hidden_fc_dropout": 0.5 if self.architecture == "b1" else None,
                "average_pool_kernel_and_stride": 2 if self.architecture == "b1" else 8,
                "average_pool_output": [4, 4] if self.architecture == "b1" else [1, 1],
                "output": "unrestricted_binary_logit", "output_shape": ["N"],
                "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad)}

    @property
    def last_convolution(self) -> nn.Conv2d:
        """Main-path spatial convolution available for later B4 Grad-CAM."""
        raise NotImplementedError


class AlexNetLike(XRayCNN):
    architecture = "b1"

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(OrderedDict([
            ("conv1", ConvBNReLU(1, 16, 5)), ("pool1", nn.MaxPool2d(2, 2)),
            ("conv2", ConvBNReLU(16, 32, 5)), ("pool2", nn.MaxPool2d(2, 2)),
            ("conv3", ConvBNReLU(32, 64, 3)), ("pool3", nn.MaxPool2d(2, 2)),
            # 8x8 -> 4x4, equivalent to adaptive averaging for the fixed input size.
            # AvgPool2d also avoids adaptive_avg_pool2d_backward_cuda's warning.
            ("average_pool", nn.AvgPool2d(2, 2)),
        ]))
        self.flatten = nn.Flatten(1)
        self.classifier = nn.Sequential(OrderedDict([
            ("fc1", nn.Linear(64 * 4 * 4, 64)), ("relu1", nn.ReLU()), ("dropout1", nn.Dropout(0.5)),
            ("fc2", nn.Linear(64, 32)), ("relu2", nn.ReLU()), ("dropout2", nn.Dropout(0.5)),
            ("output", nn.Linear(32, 1)),
        ]))

    @property
    def last_convolution(self):
        return self.features.conv3.conv


class VGGLike(XRayCNN):
    architecture = "b2"

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(OrderedDict([
            ("conv1", ConvBNReLU(1, 8, 3)), ("conv2", ConvBNReLU(8, 8, 3)),
            ("pool1", nn.MaxPool2d(2, 2)),
            ("conv3", ConvBNReLU(8, 16, 3)), ("conv4", ConvBNReLU(16, 16, 3)),
            ("pool2", nn.MaxPool2d(2, 2)),
            ("conv5", ConvBNReLU(16, 32, 3)), ("conv6", ConvBNReLU(32, 32, 3)),
            ("pool3", nn.MaxPool2d(2, 2)), ("average_pool", nn.AvgPool2d(8, 8)),
        ]))
        self.flatten = nn.Flatten(1)
        self.classifier = nn.Sequential(OrderedDict([("output", nn.Linear(32, 1))]))

    @property
    def last_convolution(self):
        return self.features.conv6.conv


class ResNetLike(XRayCNN):
    architecture = "b3"

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(OrderedDict([
            ("stem", ConvBNReLU(1, 16, 3)), ("pool", nn.MaxPool2d(2, 2)),
            ("block16", ResidualBlock(16, 16)),
            ("block32", ResidualBlock(16, 32, stride=2)),
            ("block64", ResidualBlock(32, 64, stride=2)),
            ("average_pool", nn.AvgPool2d(8, 8)),
        ]))
        self.flatten = nn.Flatten(1)
        self.classifier = nn.Sequential(OrderedDict([("output", nn.Linear(64, 1))]))

    @property
    def last_convolution(self):
        return self.features.block64.conv2.conv


def build_model(architecture: str) -> XRayCNN:
    constructors = {"b1": AlexNetLike, "b2": VGGLike, "b3": ResNetLike}
    if architecture not in constructors:
        raise ValueError(f"Unknown Part B architecture: {architecture}")
    return constructors[architecture]()


def model_from_metadata(metadata: dict) -> XRayCNN:
    model = build_model(metadata["architecture"])
    if metadata != model.metadata():
        raise ValueError("Checkpoint model metadata disagrees with assignment Figure B")
    return model
