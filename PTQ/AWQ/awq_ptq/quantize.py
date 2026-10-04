from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn

from PTQ.common.data import TokenBlocks

from .bit_width import (
    BitWidth,
    canonical_bit_width,
    quantization_levels,
)


@dataclass(frozen=True)
class ModuleQuantization:
    name: str
    module_type: str
    input_features: int
    output_features: int
    parameters: int
    alpha: float
    clip_ratio: float
    calibration_output_reconstruction_mse: float


def target_modules(
    model: nn.Module, excluded: Iterable[str]
) -> list[tuple[str, nn.Module]]:
    excluded_set = frozenset(str(value) for value in excluded)
    result: list[tuple[str, nn.Module]] = []
    for name, module in model.named_modules():
        if not name or not isinstance(module, nn.Linear):
            continue
        parts = name.split(".")
        in_opt_block = (
            len(parts) >= 4
            and parts[:3] == ["model", "decoder", "layers"]
            and parts[3].isdigit()
        )
        if not in_opt_block:
            continue
        if any(name == value or name.endswith("." + value) for value in excluded_set):
            continue
        result.append((name, module))
    if not result:
        raise RuntimeError("model exposes no non-excluded OPT Linear modules")
    return result


@torch.no_grad()
def collect_activation_second_moments(
    model: nn.Module,
    calibration: TokenBlocks,
    *,
    device: torch.device,
    micro_batch_size: int,
    excluded_modules: Iterable[str],
    precision: str,
    reconstruction_tokens: int,
) -> dict[str, dict[str, torch.Tensor]]:
    modules = target_modules(model, excluded_modules)
    sums: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = {}
    samples: dict[str, list[torch.Tensor]] = {}
    hooks = []

    def hook_for(name: str):
        def hook(_module: nn.Module, inputs: tuple[Any, ...], _output: Any) -> None:
            if not inputs or not torch.is_tensor(inputs[0]):
                raise RuntimeError(f"AWQ target {name} received no tensor input")
            values = inputs[0].detach().float().reshape(-1, inputs[0].shape[-1])
            batch_sum = values.square().sum(dim=0).cpu().double()
            if name not in sums:
                sums[name] = batch_sum
                counts[name] = int(values.shape[0])
                samples[name] = []
            else:
                sums[name].add_(batch_sum)
                counts[name] += int(values.shape[0])
            sampled = sum(item.shape[0] for item in samples[name])
            remaining = int(reconstruction_tokens) - sampled
            if remaining > 0:
                samples[name].append(values[:remaining].cpu())

        return hook

    for name, module in modules:
        hooks.append(module.register_forward_hook(hook_for(name)))
    try:
        model.eval()
        for start in range(0, calibration.num_blocks, micro_batch_size):
            count = min(micro_batch_size, calibration.num_blocks - start)
            input_ids = calibration.batch(start, count, device)
            with _autocast(device, precision):
                model(input_ids=input_ids, use_cache=False)
    finally:
        for hook in hooks:
            hook.remove()
    moments: dict[str, dict[str, torch.Tensor]] = {}
    for name, module in modules:
        if name not in sums or counts[name] <= 0:
            raise RuntimeError(f"calibration did not execute AWQ target module: {name}")
        moment = (sums[name] / counts[name]).float()
        expected = _logical_weight(module).shape[1]
        if moment.numel() != expected or not torch.isfinite(moment).all().item():
            raise RuntimeError(f"invalid activation statistic for {name}")
        reconstruction = torch.cat(samples[name], dim=0)
        if reconstruction.shape != (reconstruction_tokens, expected):
            raise RuntimeError(f"insufficient reconstruction inputs for {name}")
        moments[name] = {
            "second_moment": moment,
            "reconstruction_inputs": reconstruction,
        }
    return moments


@torch.no_grad()
def apply_awq_fake_dequant(
    model: nn.Module,
    activation_second_moments: dict[str, dict[str, torch.Tensor]],
    *,
    bits: BitWidth,
    config: dict[str, Any],
) -> dict[str, Any]:
    bits = canonical_bit_width(bits)
    group_size = int(config["group_size"])
    results: list[ModuleQuantization] = []
    for name, module in target_modules(model, config["exclude_modules"]):
        logical = _logical_weight(module)
        original_dtype = logical.dtype
        weight = logical.detach().float()
        calibration = activation_second_moments[name]
        second_moment = calibration["second_moment"].to(weight.device, torch.float32)
        reconstruction_inputs = calibration["reconstruction_inputs"].to(
            weight.device, torch.float32
        )
        selection = _search_module_protocol(
            weight,
            second_moment,
            bits=bits,
            group_size=group_size,
            alpha_grid=[float(value) for value in config["alpha_grid"]],
            clip_ratios=[float(value) for value in config["clip_ratios"]],
            max_rows=int(config["search_max_rows"]),
            reconstruction_inputs=reconstruction_inputs,
        )
        dequantized = torch.empty_like(weight)
        row_chunk = int(config["row_chunk_size"])
        scales = _awq_input_scales(weight, second_moment, selection["alpha"])
        for start in range(0, weight.shape[0], row_chunk):
            stop = min(start + row_chunk, weight.shape[0])
            scaled = weight[start:stop] * scales.unsqueeze(0)
            dequantized[start:stop] = _group_fake_quant(
                scaled,
                bits=bits,
                group_size=group_size,
                clip_ratio=selection["clip_ratio"],
            ) / scales.unsqueeze(0)
        if not torch.isfinite(dequantized).all().item():
            raise RuntimeError(
                f"AWQ fake-dequant produced non-finite weights for {name}"
            )
        _write_logical_weight(module, dequantized.to(original_dtype))
        results.append(
            ModuleQuantization(
                name=name,
                module_type=type(module).__name__,
                input_features=int(weight.shape[1]),
                output_features=int(weight.shape[0]),
                parameters=int(weight.numel()),
                alpha=float(selection["alpha"]),
                clip_ratio=float(selection["clip_ratio"]),
                calibration_output_reconstruction_mse=float(selection["output_mse"]),
            )
        )
    total_parameters = sum(item.parameters for item in results)
    return {
        "schema_version": 2,
        "algorithm": "awq_reference_v1",
        "algorithm_components": [
            "activation_second_moment",
            "activation_aware_input_channel_scaling_search",
            "range_clipping_search",
            "groupwise_affine_weight_fake_quantization",
        ],
        "nominal_weight_bits": bits,
        "quantization_levels": quantization_levels(bits),
        "quantization_scheme": "groupwise_affine",
        "group_size": group_size,
        "symmetric": False,
        "zero_point": True,
        "artifact_mode": "fake_dequant",
        "packed": False,
        "runtime_weight_storage": str(next(model.parameters()).dtype).replace(
            "torch.", ""
        ),
        "target_module_count": len(results),
        "quantized_parameter_count": int(total_parameters),
        "reconstruction_sample_tokens_per_module": int(config["reconstruction_tokens"]),
        "target_roster": [item.name for item in results],
        "modules": [asdict(item) for item in results],
    }


def _search_module_protocol(
    weight: torch.Tensor,
    second_moment: torch.Tensor,
    *,
    bits: BitWidth,
    group_size: int,
    alpha_grid: list[float],
    clip_ratios: list[float],
    max_rows: int,
    reconstruction_inputs: torch.Tensor,
) -> dict[str, float]:
    if weight.shape[0] <= max_rows:
        sample = weight
    else:
        indices = (
            torch.linspace(0, weight.shape[0] - 1, steps=max_rows, device=weight.device)
            .round()
            .long()
        )
        sample = weight.index_select(0, indices)
    reference_output = reconstruction_inputs @ sample.transpose(0, 1)
    best: dict[str, float] | None = None
    for alpha in alpha_grid:
        scales = _awq_input_scales(weight, second_moment, alpha)
        scaled_sample = sample * scales.unsqueeze(0)
        for clip_ratio in clip_ratios:
            quantized = _group_fake_quant(
                scaled_sample,
                bits=bits,
                group_size=group_size,
                clip_ratio=clip_ratio,
            ) / scales.unsqueeze(0)
            quantized_output = reconstruction_inputs @ quantized.transpose(0, 1)
            value = float((quantized_output - reference_output).square().mean())
            if not math.isfinite(value):
                raise RuntimeError("AWQ calibration search produced non-finite error")
            if best is None or value < best["output_mse"]:
                best = {"alpha": alpha, "clip_ratio": clip_ratio, "output_mse": value}
    if best is None:
        raise RuntimeError("AWQ calibration search evaluated no candidates")
    return best


def _awq_input_scales(
    weight: torch.Tensor, second_moment: torch.Tensor, alpha: float
) -> torch.Tensor:
    epsilon = torch.finfo(torch.float32).eps
    activation = second_moment.clamp_min(0).sqrt().clamp_min(epsilon)
    weight_scale = weight.abs().mean(dim=0).clamp_min(epsilon)
    scales = activation.pow(float(alpha)) / weight_scale.pow(1.0 - float(alpha))
    scales = scales.clamp(min=1.0e-4, max=1.0e4)
    scales = scales / torch.sqrt(scales.max() * scales.min()).clamp_min(epsilon)
    return scales


def _asymmetric_group_fake_quant(
    values: torch.Tensor,
    *,
    bits: int,
    group_size: int,
    clip_ratio: float,
) -> torch.Tensor:
    if values.ndim != 2:
        raise ValueError("group fake quantization expects a rank-two logical weight")
    if type(bits) is not int or bits not in {4, 3, 2}:
        raise ValueError("affine group fake quantization expects 4, 3, or 2 bits")
    maximum_code = (1 << bits) - 1
    result = torch.empty_like(values)
    epsilon = torch.finfo(torch.float32).eps
    for start in range(0, values.shape[1], group_size):
        stop = min(start + group_size, values.shape[1])
        group = values[:, start:stop]
        minimum = group.amin(dim=1, keepdim=True)
        maximum = group.amax(dim=1, keepdim=True)
        midpoint = (minimum + maximum) * 0.5
        half_range = (maximum - minimum) * (0.5 * float(clip_ratio))
        lower = midpoint - half_range
        upper = midpoint + half_range
        dynamic = (upper - lower).abs() > epsilon

        # A clamped affine zero-point can represent the requested interval
        # only when zero is inside its range. Without this widening, a
        # non-constant all-positive/all-negative group collapses at one end of
        # the codebook.
        zero_value = torch.zeros_like(lower)
        lower = torch.minimum(lower, zero_value)
        upper = torch.maximum(upper, zero_value)
        scale = ((upper - lower) / maximum_code).clamp_min(epsilon)
        zero = torch.round(-lower / scale).clamp(0, maximum_code)
        codes = torch.round(group / scale + zero).clamp(0, maximum_code)
        dequantized = (codes - zero) * scale

        # A constant group has zero range but is exactly representable by one
        # affine code. Handling it explicitly prevents an epsilon-scale collapse.
        constant_dequantized = midpoint.expand_as(group)
        result[:, start:stop] = torch.where(dynamic, dequantized, constant_dequantized)
    return result


def _group_fake_quant(
    values: torch.Tensor,
    *,
    bits: BitWidth,
    group_size: int,
    clip_ratio: float,
) -> torch.Tensor:
    bits = canonical_bit_width(bits)
    return _asymmetric_group_fake_quant(
        values,
        bits=bits,
        group_size=group_size,
        clip_ratio=clip_ratio,
    )


def _logical_weight(module: nn.Module) -> torch.Tensor:
    if isinstance(module, nn.Linear):
        return module.weight
    raise TypeError(f"unsupported AWQ target module: {type(module).__name__}")


def _write_logical_weight(module: nn.Module, logical: torch.Tensor) -> None:
    if isinstance(module, nn.Linear):
        module.weight.copy_(logical)
    else:
        raise TypeError(f"unsupported AWQ target module: {type(module).__name__}")


def _autocast(device: torch.device, precision: str):
    if precision == "bf16":
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    if precision == "fp32":
        return torch.autocast(device_type=device.type, enabled=False)
    raise ValueError(f"unsupported precision: {precision}")
