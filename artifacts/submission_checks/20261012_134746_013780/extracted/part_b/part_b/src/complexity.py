"""Shape-checked inference FLOPs from the installed PyTorch profiler.

One multiply-add is two FLOPs. Count convolution/linear products and residual
additions; explicitly exclude operations for which the profiler reports no FLOPs.
"""

from __future__ import annotations

import copy
import math

import torch
from torch import nn
from torch.profiler import ProfilerActivity, profile

from part_b.src.models import ResidualBlock


FLOPS_CONVENTION = "one float32 1x1x64x64 image, eval mode; 2 FLOPs per conv/linear MAC plus residual additions"
EXCLUDED_OPERATIONS = ["BatchNorm", "ReLU", "max/average pooling", "bias additions in Linear",
                       "flatten/views", "data normalization/transfers", "sigmoid/threshold", "backward/optimizer"]


def parameter_breakdown(model):
    counts = {"convolution": 0, "batchnorm_affine": 0, "linear": 0, "other": 0}
    seen = set()
    for module in model.modules():
        category = ("convolution" if isinstance(module, nn.Conv2d) else "batchnorm_affine"
                    if isinstance(module, nn.BatchNorm2d) else "linear" if isinstance(module, nn.Linear) else "other")
        for parameter in module.parameters(recurse=False):
            if parameter.requires_grad and id(parameter) not in seen:
                counts[category] += parameter.numel()
                seen.add(id(parameter))
    counts["total"] = sum(counts.values())
    return counts


def profile_model(model):
    """Profile a CPU eval copy on synthetic zeros, preserving the caller's state.

    Counts depend on shapes, not learned weights. No dataset is accessed.
    Unknown leaf modules or nonzero profiler operators fail rather than silently
    broadening the reported scope. Pool/BN/activation execution is recorded.
    """
    candidate = copy.deepcopy(model).cpu().eval()
    allowed = (nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.ReLU, nn.MaxPool2d,
               nn.AvgPool2d, nn.Flatten, nn.Dropout, nn.Identity)
    layers, additions, handles = [], [], []
    for name, module in candidate.named_modules():
        if isinstance(module, ResidualBlock):
            def residual_hook(module, inputs, output, name=name):
                additions.append({"module": name, "output_shape": list(output.shape), "flops": output.numel()})
            handles.append(module.register_forward_hook(residual_hook))
        if any(module.children()):
            continue
        if not isinstance(module, allowed):
            raise NotImplementedError(f"Unspecified FLOPs scope for leaf {name}: {type(module).__name__}")
        def leaf_hook(module, inputs, output, name=name):
            record = {"module": name, "type": type(module).__name__,
                      "input_shape": list(inputs[0].shape), "output_shape": list(output.shape), "macs": 0, "flops": 0}
            if isinstance(module, nn.Conv2d):
                record.update(in_channels=module.in_channels, out_channels=module.out_channels,
                    groups=module.groups, kernel_size=list(module.kernel_size), stride=list(module.stride),
                    padding=list(module.padding), bias=module.bias is not None)
                record["macs"] = output.numel() * (module.in_channels // module.groups) * math.prod(module.kernel_size)
            elif isinstance(module, nn.Linear):
                record.update(in_features=module.in_features, out_features=module.out_features, bias=module.bias is not None)
                record["macs"] = output.numel() * module.in_features
            record["flops"] = 2 * record["macs"]
            layers.append(record)
        handles.append(module.register_forward_hook(leaf_hook))
    try:
        with torch.inference_mode(), profile(activities=[ProfilerActivity.CPU], record_shapes=True,
                                            with_flops=True, acc_events=True) as profiler:
            logits = candidate(torch.zeros(1, 1, 64, 64, dtype=torch.float32))
    finally:
        for handle in handles:
            handle.remove()
    events = [{"operator": event.key, "flops": int(event.flops), "calls": event.count}
              for event in profiler.key_averages() if event.flops]
    if any(event["operator"] not in ("aten::conv2d", "aten::addmm", "aten::mm", "aten::add") for event in events):
        raise RuntimeError("Profiler counted an operation outside the documented scope")
    core = sum(row["flops"] for row in layers)
    residual = sum(row["flops"] for row in additions)
    profiled_core = sum(row["flops"] for row in events if row["operator"] != "aten::add")
    profiled_additions = sum(row["flops"] for row in events if row["operator"] == "aten::add")
    if core != profiled_core or residual != profiled_additions:
        raise RuntimeError(f"Profiler/shape formula disagreement: {(profiled_core, profiled_additions)} != {(core, residual)}")
    return {"architecture": getattr(model, "architecture", type(model).__name__),
        "input_shape": [1, 1, 64, 64], "output_shape": list(logits.shape), "dtype": "float32",
        "mode": "eval", "profiling_device": "cpu", "torch_version": str(torch.__version__),
        "profiler": "torch.profiler.profile(with_flops=True, record_shapes=True, acc_events=True)",
        "convention": FLOPS_CONVENTION, "excluded_operations": EXCLUDED_OPERATIONS,
        "trainable_parameters": parameter_breakdown(model)["total"], "parameter_breakdown": parameter_breakdown(model),
        "convolution_and_linear_macs": core // 2, "convolution_and_linear_flops": core,
        "residual_addition_flops": residual, "inference_flops": core + residual,
        "profiler_flops": sum(row["flops"] for row in events), "shape_formula_matches_profiler": True,
        "nonzero_profiler_operators": events, "layers": layers, "residual_additions": additions,
        "uncounted_executed_leaf_modules": sorted({row["type"] for row in layers if row["type"] not in ("Conv2d", "Linear")}),
        "zero_flop_profiler_operators": sorted(event.key for event in profiler.key_averages() if not event.flops),
        "synthetic_input_only": True,
        "source": "https://docs.pytorch.org/docs/2.11/profiler.html"}
