from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn

from .bit_width import (
    BitWidth,
    canonical_bit_width,
    quantization_levels,
)


class QuantizationError(RuntimeError):
    """The second-order GPTQ pass could not produce finite fake-dequant weights."""


@dataclass(frozen=True)
class ModuleQuantizationStats:
    name: str
    module_type: str
    bits: BitWidth
    quantization_levels: int
    group_size: int
    rows: int
    columns: int
    hessian_samples: int
    damping: float
    weighted_error: float
    weight_mse: float


class _HessianCollector:
    def __init__(self, columns: int) -> None:
        self.columns = int(columns)
        self.hessian: torch.Tensor | None = None
        self.samples = 0

    def __call__(self, _module: nn.Module, inputs: tuple[Any, ...]) -> None:
        if not inputs or not torch.is_tensor(inputs[0]):
            raise QuantizationError("quantized module received no tensor input")

        values = inputs[0].detach()

        if values.shape[-1] != self.columns:
            raise QuantizationError(
                f"module input width changed: expected={self.columns}, got={values.shape[-1]}"
            )

        matrix = values.reshape(-1, self.columns)

        if self.hessian is None:
            self.hessian = torch.zeros(
                (self.columns, self.columns),
                dtype=torch.float32,
                device=matrix.device,
            )

        for chunk in matrix.split(1024, dim=0):
            x = chunk.to(dtype=torch.float32)
            self.hessian.addmm_(x.transpose(0, 1), x)
            self.samples += int(x.shape[0])

    def finish(self) -> tuple[torch.Tensor, int]:
        if self.hessian is None or self.samples < 1:
            raise QuantizationError("calibration produced no module inputs")

        # self.hessian is created while calibration is running under
        # torch.inference_mode(). Such a tensor cannot be modified in-place
        # after leaving InferenceMode.
        #
        # clone() creates a normal tensor, after which the GPTQ Hessian
        # normalization can safely be performed.
        with torch.inference_mode(False):
            hessian = self.hessian.detach().clone() * (2.0 / float(self.samples))

        if not torch.isfinite(hessian).all().item():
            raise QuantizationError("calibration Hessian contains non-finite values")

        return hessian, self.samples


def gptq_quantize_model(
    model: nn.Module,
    calibration_blocks: torch.Tensor,
    *,
    bits: BitWidth,
    group_size: int,
    damp_percent: float,
    act_order: bool,
    symmetric: bool,
    column_block_size: int,
    micro_batch_size: int,
    precision: str,
    device: torch.device,
) -> list[dict[str, Any]]:
    """Apply true blockwise GPTQ to every transformer-block projection.

    Each block first observes C4 activations and accumulates ``X.T @ X`` for
    every Linear projection. The columns are then quantized sequentially
    with inverse-Hessian error propagation, including cross-column-block
    compensation. The stored weights are fake-dequantized floating tensors;
    this function intentionally does not claim to create a packed inference
    format.
    """

    try:
        bits = canonical_bit_width(bits)
    except (TypeError, ValueError) as error:
        raise QuantizationError("GPTQ supports only 4, 3, or 2 bits") from error

    if group_size != 128:
        raise QuantizationError("this experiment protocol requires group_size=128")

    if calibration_blocks.ndim != 2 or calibration_blocks.shape[0] < 1:
        raise QuantizationError("calibration_blocks must be a non-empty rank-2 tensor")

    if calibration_blocks.shape[0] % micro_batch_size:
        raise QuantizationError(
            "calibration block count must divide micro_batch_size exactly"
        )

    blocks = _transformer_blocks(model)

    records: list[dict[str, Any]] = []

    flags = {module: bool(module.training) for module in model.modules()}

    model.eval()

    try:
        for block_index, (block_prefix, block) in enumerate(blocks):
            modules = _quantizable_modules(block)

            if not modules:
                raise QuantizationError(
                    f"transformer block {block_index} has no projections"
                )

            collectors: dict[str, _HessianCollector] = {}
            handles = []

            for relative_name, module in modules:
                _, columns = _weight_matrix(module).shape

                collector = _HessianCollector(columns)

                collectors[relative_name] = collector

                handles.append(module.register_forward_pre_hook(collector))

            try:
                with torch.inference_mode():
                    for start in range(
                        0,
                        calibration_blocks.shape[0],
                        micro_batch_size,
                    ):
                        input_ids = calibration_blocks[
                            start : start + micro_batch_size
                        ].to(
                            device=device,
                            dtype=torch.long,
                            non_blocking=True,
                        )

                        with _calibration_autocast(
                            precision,
                            device,
                        ):
                            model(
                                input_ids=input_ids,
                                use_cache=False,
                            )

            finally:
                for handle in handles:
                    handle.remove()

            for relative_name, module in modules:
                hessian, samples = collectors[relative_name].finish()

                original = _weight_matrix(module).detach()

                quantized, details = gptq_quantize_weight(
                    original,
                    hessian,
                    bits=bits,
                    group_size=group_size,
                    damp_percent=damp_percent,
                    act_order=act_order,
                    symmetric=symmetric,
                    column_block_size=column_block_size,
                )

                _replace_weight_matrix(
                    module,
                    quantized,
                )

                stats = ModuleQuantizationStats(
                    name=f"{block_prefix}.{relative_name}",
                    module_type=type(module).__name__,
                    bits=bits,
                    quantization_levels=quantization_levels(bits),
                    group_size=group_size,
                    rows=int(original.shape[0]),
                    columns=int(original.shape[1]),
                    hessian_samples=samples,
                    damping=float(details["damping"]),
                    weighted_error=float(details["weighted_error"]),
                    weight_mse=float(details["weight_mse"]),
                )

                records.append(asdict(stats))

                del hessian, quantized

            del collectors, handles

            if device.type == "cuda":
                torch.cuda.empty_cache()

    finally:
        for module, training in flags.items():
            module.training = training

    if not records:
        raise QuantizationError("GPTQ quantized no transformer projections")

    return records


def gptq_quantize_weight(
    weight: torch.Tensor,
    hessian: torch.Tensor,
    *,
    bits: BitWidth,
    group_size: int,
    damp_percent: float,
    act_order: bool,
    symmetric: bool,
    column_block_size: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Quantize one ``[out_features, in_features]`` weight with GPTQ.

    Unlike round-to-nearest, every quantization residual is propagated into
    the unprocessed columns using the Cholesky factor of the inverse activation
    Hessian. This is the defining second-order GPTQ compensation step.

    """

    if weight.ndim != 2 or hessian.ndim != 2:
        raise QuantizationError("weight and Hessian must both be rank two")

    columns = weight.shape[1]

    if hessian.shape != (columns, columns):
        raise QuantizationError("Hessian shape must equal the weight input dimension")

    try:
        bits = canonical_bit_width(bits)
    except (TypeError, ValueError) as error:
        raise QuantizationError("invalid GPTQ bit/group/block configuration") from error

    if group_size < 1 or column_block_size < 1:
        raise QuantizationError("invalid GPTQ bit/group/block configuration")

    if not 0.0 < damp_percent < 1.0:
        raise QuantizationError("damp_percent must lie in (0, 1)")

    original_device = weight.device
    original_dtype = weight.dtype

    work = weight.detach().to(
        dtype=torch.float32,
        copy=True,
    )

    h = hessian.detach().to(
        device=original_device,
        dtype=torch.float32,
        copy=True,
    )

    if not torch.isfinite(work).all().item() or not torch.isfinite(h).all().item():
        raise QuantizationError("weight/Hessian contains non-finite values")

    diagonal = torch.diag(h)

    dead = diagonal <= torch.finfo(h.dtype).eps

    if dead.any():
        indices = torch.arange(
            columns,
            device=h.device,
        )

        h[indices[dead], indices[dead]] = 1.0

        work[:, dead] = 0.0

    permutation: torch.Tensor | None = None

    if act_order:
        permutation = torch.argsort(
            torch.diag(h),
            descending=True,
        )

        work = work[:, permutation]

        h = h.index_select(0, permutation).index_select(1, permutation)

    mean_diagonal = float(torch.diag(h).mean().item())

    damping = max(
        mean_diagonal * float(damp_percent),
        torch.finfo(h.dtype).eps,
    )

    h_inverse_factor, actual_damping = _inverse_hessian_factor(
        h,
        damping,
    )

    inverse_diagonal = torch.diag(h_inverse_factor)

    if (
        not torch.isfinite(inverse_diagonal).all().item()
        or not (inverse_diagonal > 0).all().item()
    ):
        raise QuantizationError("inverse-Hessian Cholesky diagonal is not positive")

    quantized = torch.zeros_like(work)

    total_weighted_error = torch.zeros(
        (),
        device=work.device,
        dtype=torch.float64,
    )

    scale: torch.Tensor | None = None
    zero: torch.Tensor | None = None

    for block_start in range(
        0,
        columns,
        column_block_size,
    ):
        block_end = min(
            block_start + column_block_size,
            columns,
        )

        block_weights = work[
            :,
            block_start:block_end,
        ].clone()

        block_quantized = torch.zeros_like(block_weights)

        block_errors = torch.zeros_like(block_weights)

        inverse_block = h_inverse_factor[
            block_start:block_end,
            block_start:block_end,
        ]

        for local_column in range(block_end - block_start):
            column = block_start + local_column

            if column % group_size == 0 or scale is None or zero is None:
                group_end = min(
                    column + group_size,
                    columns,
                )

                scale, zero = _group_parameters(
                    work[:, column:group_end],
                    bits=bits,
                    symmetric=symmetric,
                )

            current = block_weights[
                :,
                local_column,
            ].clone()

            diagonal_factor = inverse_block[
                local_column,
                local_column,
            ]

            dequantized = _fake_dequantize(
                current,
                scale,
                zero,
                bits,
            )

            block_quantized[
                :,
                local_column,
            ] = dequantized

            residual = (current - dequantized) / diagonal_factor

            block_weights[
                :,
                local_column:,
            ] -= residual.unsqueeze(1) * inverse_block[
                local_column,
                local_column:,
            ].unsqueeze(0)

            block_errors[
                :,
                local_column,
            ] = residual

            total_weighted_error += (
                (current.double() - dequantized.double()).pow(2)
                / (2.0 * diagonal_factor.double().pow(2))
            ).sum()

        quantized[
            :,
            block_start:block_end,
        ] = block_quantized

        if block_end < columns:
            work[:, block_end:] -= (
                block_errors
                @ h_inverse_factor[
                    block_start:block_end,
                    block_end:,
                ]
            )

    if permutation is not None:
        inverse_permutation = torch.argsort(permutation)

        quantized = quantized[
            :,
            inverse_permutation,
        ]

    if not torch.isfinite(quantized).all().item():
        raise QuantizationError("GPTQ produced non-finite fake-dequant weights")

    mse = float((weight.detach().float() - quantized).pow(2).mean().item())

    weighted_error = float(total_weighted_error.item())

    if (
        not math.isfinite(mse)
        or not math.isfinite(weighted_error)
        or not math.isfinite(actual_damping)
    ):
        raise QuantizationError("GPTQ error statistics are non-finite")

    result = quantized.to(
        device=original_device,
        dtype=original_dtype,
    )

    return result, {
        "damping": float(actual_damping),
        "weighted_error": weighted_error,
        "weight_mse": mse,
    }


def _inverse_hessian_factor(
    hessian: torch.Tensor,
    initial_damping: float,
) -> tuple[torch.Tensor, float]:
    indices = torch.arange(
        hessian.shape[0],
        device=hessian.device,
    )

    for attempt in range(6):
        damping = initial_damping * (10.0**attempt)

        damped = hessian.clone()

        damped[
            indices,
            indices,
        ] += damping

        factor, info = torch.linalg.cholesky_ex(damped)

        if int(info.max().item()) == 0:
            inverse = torch.cholesky_inverse(factor)

            upper, upper_info = torch.linalg.cholesky_ex(
                inverse,
                upper=True,
            )

            if int(upper_info.max().item()) == 0 and torch.isfinite(upper).all().item():
                return upper, float(damping)

    raise QuantizationError(
        "activation Hessian remains non-positive-definite after damping"
    )


def _group_parameters(
    values: torch.Tensor,
    *,
    bits: BitWidth,
    symmetric: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    try:
        integer_bits = canonical_bit_width(bits)
    except (TypeError, ValueError) as error:
        raise QuantizationError("GPTQ supports only 4, 3, or 2 bits") from error
    maximum_code = float(2**integer_bits - 1)

    minimum = values.amin(
        dim=1,
        keepdim=True,
    )

    maximum = values.amax(
        dim=1,
        keepdim=True,
    )

    zero_value = torch.zeros_like(minimum)

    minimum = torch.minimum(
        minimum,
        zero_value,
    )

    maximum = torch.maximum(
        maximum,
        zero_value,
    )

    if symmetric:
        absolute = torch.maximum(
            minimum.abs(),
            maximum.abs(),
        )

        minimum = -absolute
        maximum = absolute

    scale = (maximum - minimum) / maximum_code

    scale = torch.where(
        scale > 0,
        scale,
        torch.ones_like(scale),
    )

    if symmetric:
        zero = torch.full_like(
            scale,
            float((2**integer_bits + 1) // 2),
        )

    else:
        zero = torch.round(-minimum / scale).clamp_(
            0,
            maximum_code,
        )

    return scale, zero


def _fake_dequantize(
    values: torch.Tensor,
    scale: torch.Tensor,
    zero: torch.Tensor,
    bits: BitWidth,
) -> torch.Tensor:
    try:
        integer_bits = canonical_bit_width(bits)
    except (TypeError, ValueError) as error:
        raise QuantizationError("GPTQ supports only 4, 3, or 2 bits") from error

    maximum_code = float(2**integer_bits - 1)

    codes = torch.round(values.unsqueeze(1) / scale + zero).clamp_(
        0,
        maximum_code,
    )

    return ((codes - zero) * scale).squeeze(1)


def _transformer_blocks(
    model: nn.Module,
) -> list[tuple[str, nn.Module]]:
    architecture = getattr(
        getattr(
            model,
            "config",
            None,
        ),
        "model_type",
        None,
    )

    if architecture == "opt":
        blocks = getattr(
            getattr(
                model,
                "model",
                None,
            ),
            "decoder",
            None,
        )

        blocks = getattr(
            blocks,
            "layers",
            None,
        )

        prefix = "model.decoder.layers"

    else:
        raise QuantizationError(f"unsupported GPTQ model architecture: {architecture}")

    if (
        not isinstance(
            blocks,
            (nn.ModuleList, list, tuple),
        )
        or not blocks
    ):
        raise QuantizationError("could not locate transformer blocks")

    return [(f"{prefix}.{index}", block) for index, block in enumerate(blocks)]


def _calibration_autocast(
    precision: str,
    device: torch.device,
):
    if precision == "bf16_mixed" and device.type == "cuda":
        return torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        )

    if precision == "fp32" or device.type == "cpu":
        return nullcontext()

    raise QuantizationError(f"unsupported calibration precision: {precision}")


def _quantizable_modules(
    block: nn.Module,
) -> list[tuple[str, nn.Module]]:
    return [
        (name, module)
        for name, module in block.named_modules()
        if name
        and isinstance(
            module,
            nn.Linear,
        )
    ]


def _weight_matrix(
    module: nn.Module,
) -> torch.Tensor:
    weight = getattr(
        module,
        "weight",
        None,
    )

    if (
        not isinstance(
            weight,
            torch.Tensor,
        )
        or weight.ndim != 2
    ):
        raise QuantizationError(
            f"unsupported quantized module weight: {type(module).__name__}"
        )

    return weight


def _replace_weight_matrix(
    module: nn.Module,
    matrix: torch.Tensor,
) -> None:
    target = getattr(
        module,
        "weight",
        None,
    )

    if not isinstance(
        target,
        torch.Tensor,
    ):
        raise QuantizationError("quantized module has no weight tensor")

    value = matrix

    if value.shape != target.shape:
        raise QuantizationError("fake-dequant weight shape changed")

    with torch.no_grad():
        target.copy_(
            value.to(
                device=target.device,
                dtype=target.dtype,
            )
        )
