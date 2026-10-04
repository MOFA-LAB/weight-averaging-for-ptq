from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from src.distributed import DistributedContext

ESTIMATOR_NAME = "hutchinson_rademacher_autograd_hvp"
LAMBDA_MAX_ESTIMATOR_NAME = "symmetric_lanczos_autograd_hvp"
OBJECTIVE_NAME = "fixed_validation_prefix_cross_entropy"
CI95_NORMAL_MULTIPLIER = 1.959963984540054


def _parameter_state_label(model: nn.Module) -> str:
    """Return the live model-weight identity measured by this estimator."""

    return "W"


def _positive_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return int(value)


def _validate_config(config: dict[str, Any]) -> tuple[int, int, int, int, str]:
    if not isinstance(config, dict):
        raise TypeError("Hessian configuration must be a mapping")
    expected = {"batches", "micro_batch_size", "probes", "seed", "precision"}
    unknown = sorted(set(config) - expected)
    missing = sorted(expected - set(config))
    if unknown or missing:
        raise ValueError(
            f"invalid Hessian configuration; missing={missing}, unknown={unknown}"
        )
    batches = _positive_integer(config["batches"], "hessian.batches")
    micro_batch_size = _positive_integer(
        config["micro_batch_size"], "hessian.micro_batch_size"
    )
    probes = _positive_integer(config["probes"], "hessian.probes")
    seed = config["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("hessian.seed must be a non-negative integer")
    precision = str(config["precision"])
    if precision not in {"fp32", "bf16_mixed"}:
        raise ValueError("hessian.precision must be fp32 or bf16_mixed")
    return batches, micro_batch_size, probes, int(seed), precision


def _nonnegative_finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a non-negative finite number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{label} must be a non-negative finite number")
    return result


def validate_curvature_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return a strict normalized curvature-estimation configuration.

    This is the shared schema used by both cheap trajectory measurements and
    higher-accuracy final endpoint measurements.  Keeping the schema identical
    makes the complete numerical protocol suitable for an artifact identity.
    """

    if not isinstance(config, dict):
        raise TypeError("curvature configuration must be a mapping")
    expected = {
        "batches",
        "micro_batch_size",
        "trace_probes",
        "lambda_max_iterations",
        "lambda_max_tolerance",
        "seed",
        "precision",
    }
    unknown = sorted(set(config) - expected)
    missing = sorted(expected - set(config))
    if unknown or missing:
        raise ValueError(
            f"invalid curvature configuration; missing={missing}, unknown={unknown}"
        )
    seed = config["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("curvature.seed must be a non-negative integer")
    precision = str(config["precision"])
    if precision not in {"fp32", "bf16_mixed"}:
        raise ValueError("curvature.precision must be fp32 or bf16_mixed")
    return {
        "batches": _positive_integer(config["batches"], "curvature.batches"),
        "micro_batch_size": _positive_integer(
            config["micro_batch_size"], "curvature.micro_batch_size"
        ),
        "trace_probes": _positive_integer(
            config["trace_probes"], "curvature.trace_probes"
        ),
        "lambda_max_iterations": _positive_integer(
            config["lambda_max_iterations"], "curvature.lambda_max_iterations"
        ),
        "lambda_max_tolerance": _nonnegative_finite_float(
            config["lambda_max_tolerance"], "curvature.lambda_max_tolerance"
        ),
        "seed": int(seed),
        "precision": precision,
    }


@contextmanager
def _math_attention_context(device: torch.device) -> Iterator[None]:
    """Select an attention backend with CUDA double-backward support."""

    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except ImportError:  # pragma: no cover - compatibility for older PyTorch
        if device.type == "cuda" and hasattr(torch.backends.cuda, "sdp_kernel"):
            with torch.backends.cuda.sdp_kernel(
                enable_flash=False,
                enable_math=True,
                enable_mem_efficient=False,
            ):
                yield
        else:
            with nullcontext():
                yield
        return
    # CPU flash attention also lacks the double backward needed by Hessian HVPs.
    with sdpa_kernel([SDPBackend.MATH]):
        yield


@contextmanager
def _precision_context(precision: str, device: torch.device) -> Iterator[None]:
    if precision == "fp32":
        if device.type != "cuda":
            with nullcontext():
                yield
            return
        previous_matmul_tf32 = bool(torch.backends.cuda.matmul.allow_tf32)
        previous_cudnn_tf32 = bool(torch.backends.cudnn.allow_tf32)
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            yield
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous_matmul_tf32
            torch.backends.cudnn.allow_tf32 = previous_cudnn_tf32
        return
    if precision == "bf16_mixed":
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            yield
        return
    raise ValueError(f"unsupported Hessian precision: {precision}")


def _rademacher_like(
    parameter: nn.Parameter,
    generator: torch.Generator,
) -> torch.Tensor:
    # A dedicated generator leaves the global CPU/CUDA RNG streams untouched.
    return (
        torch.empty_like(parameter, memory_format=torch.preserve_format)
        .bernoulli_(0.5, generator=generator)
        .mul_(2.0)
        .sub_(1.0)
    )


def _single_probe(
    model: nn.Module,
    input_ids: torch.Tensor,
    parameters: tuple[nn.Parameter, ...],
    *,
    seed: int,
    precision: str,
    device: torch.device,
) -> tuple[float, int]:
    """Evaluate one common-random-number Hutchinson probe on one minibatch."""

    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    probes = tuple(_rademacher_like(parameter, generator) for parameter in parameters)
    targets = input_ids[:, 1:]
    token_count = int(targets.numel())
    if token_count < 1:
        raise ValueError("Hessian input sequences must contain at least two tokens")

    with _math_attention_context(device), _precision_context(precision, device):
        logits = model(input_ids=input_ids, use_cache=False).logits[:, :-1, :]
        # This is the unmasked full validation CE. There are deliberately no
        # token-frequency, quintile, or residual-analysis code paths here.
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            reduction="mean",
        )
        first_gradients = torch.autograd.grad(
            loss,
            parameters,
            create_graph=True,
            allow_unused=True,
        )
        directional_derivative: torch.Tensor | None = None
        for gradient, probe in zip(first_gradients, probes):
            if gradient is None or not gradient.requires_grad:
                continue
            term = torch.dot(gradient.reshape(-1), probe.reshape(-1))
            directional_derivative = (
                term
                if directional_derivative is None
                else directional_derivative + term
            )

        estimate_tensor = torch.zeros((), dtype=torch.float64, device=device)
        if directional_derivative is not None:
            products = torch.autograd.grad(
                directional_derivative,
                parameters,
                allow_unused=True,
            )
            for product, probe in zip(products, probes):
                if product is not None:
                    estimate_tensor.add_(
                        torch.dot(product.reshape(-1), probe.reshape(-1))
                        .detach()
                        .double()
                    )
    estimate = float(estimate_tensor)
    del loss, first_gradients, directional_derivative, probes
    return estimate, token_count


def estimate_hessian_trace(
    model: nn.Module,
    corpus: Any,
    config: dict[str, Any],
    device: torch.device,
    distributed: DistributedContext,
) -> dict[str, Any]:
    """Estimate the live-weight full-validation Hessian trace.

    Validation minibatch ``b`` is owned by rank ``b % world_size``. Every rank
    constructs the same Rademacher vector for a given probe index, so the
    rank-local Hessian-vector products combine into the same estimator as a
    single process evaluating the complete fixed validation prefix. One
    all-reduce combines all per-probe token-weighted sums and counts.

    The function never swaps model parameters and therefore measures W only.
    It uses ``torch.autograd.grad`` rather than ``backward``, preserving
    parameter ``.grad`` fields, optimizer state, and global RNG state.
    """

    batches, micro_batch_size, probe_count, seed, precision = _validate_config(config)
    world_size = int(distributed.world_size)
    rank = int(distributed.rank)
    if world_size < 1 or not 0 <= rank < world_size:
        raise ValueError(
            f"invalid distributed context rank={rank}, world_size={world_size}"
        )
    validation_blocks = int(corpus.validation_blocks)
    if validation_blocks < 1:
        raise ValueError("validation corpus has no complete sequence blocks")
    if micro_batch_size > validation_blocks:
        raise ValueError(
            "curvature.micro_batch_size exceeds the validation corpus; "
            "refusing to wrap and repeat validation blocks"
        )
    maximum_batches = validation_blocks // micro_batch_size
    used_batches = min(batches, maximum_batches)
    parameters = tuple(
        parameter for parameter in model.parameters() if parameter.requires_grad
    )
    if not parameters:
        raise ValueError("Hessian trace requires at least one trainable parameter")
    parameter_count = sum(parameter.numel() for parameter in parameters)

    local_weighted = torch.zeros(
        probe_count,
        dtype=torch.float64,
        device=device,
    )
    local_token_count = 0
    local_batch_count = 0
    training_flags = {module: bool(module.training) for module in model.modules()}
    local_error: str | None = None
    try:
        model.eval()
        for batch_index in range(rank, used_batches, world_size):
            input_ids = corpus.batch(
                "validation",
                start_block=batch_index * micro_batch_size,
                batch_size=micro_batch_size,
                device=device,
            )
            batch_tokens: int | None = None
            for probe_index in range(probe_count):
                estimate, token_count = _single_probe(
                    model,
                    input_ids,
                    parameters,
                    seed=seed + probe_index,
                    precision=precision,
                    device=device,
                )
                if batch_tokens is None:
                    batch_tokens = token_count
                elif token_count != batch_tokens:
                    raise RuntimeError("validation token count changed across probes")
                local_weighted[probe_index] += estimate * token_count
            local_token_count += int(batch_tokens or 0)
            local_batch_count += 1
            del input_ids
    except torch.cuda.OutOfMemoryError:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        local_error = (
            "CUDA out of memory; reduce curvature.micro_batch_size, "
            "curvature.batches, or curvature.trace_probes"
        )
    except Exception as exc:  # noqa: BLE001 - synchronize rank-local failures
        local_error = f"{type(exc).__name__}: {exc}"
    finally:
        # Preserve intentionally mixed train/eval states exactly.
        for module, was_training in training_flags.items():
            module.training = was_training
    _synchronize_computation_error(distributed, local_error, "Hessian trace")

    reduced = torch.cat(
        (
            local_weighted,
            torch.tensor(
                [float(local_token_count), float(local_batch_count)],
                dtype=torch.float64,
                device=device,
            ),
        )
    )
    distributed.all_reduce_sum(reduced)
    global_token_count = round(float(reduced[-2]))
    global_batch_count = round(float(reduced[-1]))
    if global_token_count < 1:
        raise RuntimeError("distributed Hessian estimate has no validation tokens")
    if global_batch_count != used_batches:
        raise RuntimeError(
            "distributed Hessian batch sharding is incomplete: "
            f"found {global_batch_count}/{used_batches}"
        )

    probe_estimates_tensor = reduced[:probe_count] / float(global_token_count)
    probe_estimates = [float(value) for value in probe_estimates_tensor.cpu()]
    trace = math.fsum(probe_estimates) / probe_count
    if probe_count > 1:
        variance = math.fsum((value - trace) ** 2 for value in probe_estimates) / (
            probe_count - 1
        )
        trace_std = math.sqrt(max(variance, 0.0))
    else:
        trace_std = 0.0
    trace_standard_error = trace_std / math.sqrt(probe_count)
    margin = CI95_NORMAL_MULTIPLIER * trace_standard_error

    return {
        "trace": float(trace),
        "trace_std": float(trace_std),
        "trace_standard_error": float(trace_standard_error),
        "trace_ci95_lower": float(trace - margin),
        "trace_ci95_upper": float(trace + margin),
        "probe_estimates": probe_estimates,
        "metadata": {
            "estimator": ESTIMATOR_NAME,
            "objective": OBJECTIVE_NAME,
            "parameter_state": _parameter_state_label(model),
            "parameter_count": int(parameter_count),
            "validation_batches": int(global_batch_count),
            "validation_micro_batch_size": int(micro_batch_size),
            "validation_token_count": int(global_token_count),
            "probes": int(probe_count),
            "probe_distribution": "Rademacher{-1,+1}",
            "probe_seed": int(seed),
            "common_probe_across_batches_and_ranks": True,
            "common_probes_across_endpoints": True,
            "precision": precision,
            "world_size": world_size,
            "batch_sharding": "round_robin_by_global_batch_index",
            "confidence_interval": "normal_approximation_95_percent",
            "standard_deviation": "unbiased_across_hutchinson_probes",
        },
    }


def _synchronize_computation_error(
    distributed: DistributedContext,
    error: str | None,
    operation: str,
) -> None:
    """Raise rank-local failures coherently before entering a tensor collective."""

    gather = getattr(distributed, "gather_objects", None)
    if gather is None:
        if error is not None:
            raise RuntimeError(f"{operation} failed: {error}")
        return
    errors = gather(error)
    failures = [
        f"rank {rank}: {message}"
        for rank, message in enumerate(errors)
        if message is not None
    ]
    if failures:
        raise RuntimeError(f"{operation} failed; " + " | ".join(failures))


def _vector_dot(
    left: tuple[torch.Tensor, ...],
    right: tuple[torch.Tensor, ...],
    device: torch.device,
) -> torch.Tensor:
    result = torch.zeros((), dtype=torch.float64, device=device)
    for lhs, rhs in zip(left, right):
        result.add_(torch.dot(lhs.reshape(-1), rhs.reshape(-1)).double())
    return result


def _vector_norm(vector: tuple[torch.Tensor, ...], device: torch.device) -> float:
    squared_norm = float(_vector_dot(vector, vector, device))
    return math.sqrt(max(squared_norm, 0.0))


def _scaled_vector(
    vector: tuple[torch.Tensor, ...],
    scale: float,
) -> tuple[torch.Tensor, ...]:
    return tuple(value.mul(float(scale)) for value in vector)


def _single_hessian_vector_product(
    model: nn.Module,
    input_ids: torch.Tensor,
    parameters: tuple[nn.Parameter, ...],
    vector: tuple[torch.Tensor, ...],
    *,
    precision: str,
    device: torch.device,
) -> tuple[tuple[torch.Tensor, ...], int]:
    """Compute the minibatch mean-CE Hessian-vector product without ``.grad`` writes."""

    targets = input_ids[:, 1:]
    token_count = int(targets.numel())
    if token_count < 1:
        raise ValueError("Hessian input sequences must contain at least two tokens")
    with _math_attention_context(device), _precision_context(precision, device):
        logits = model(input_ids=input_ids, use_cache=False).logits[:, :-1, :]
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            reduction="mean",
        )
        first_gradients = torch.autograd.grad(
            loss,
            parameters,
            create_graph=True,
            allow_unused=True,
        )
        directional_derivative: torch.Tensor | None = None
        for gradient, direction in zip(first_gradients, vector):
            if gradient is None or not gradient.requires_grad:
                continue
            term = torch.dot(gradient.reshape(-1), direction.reshape(-1))
            directional_derivative = (
                term
                if directional_derivative is None
                else directional_derivative + term
            )
        products: tuple[torch.Tensor | None, ...]
        if directional_derivative is None:
            products = tuple(None for _ in parameters)
        else:
            products = torch.autograd.grad(
                directional_derivative,
                parameters,
                allow_unused=True,
            )
        result = tuple(
            torch.zeros_like(parameter, memory_format=torch.preserve_format)
            if product is None
            else product.detach()
            for parameter, product in zip(parameters, products)
        )
    del loss, first_gradients, directional_derivative, products
    return result, token_count


def _full_validation_hessian_vector_product(
    model: nn.Module,
    corpus: Any,
    parameters: tuple[nn.Parameter, ...],
    vector: tuple[torch.Tensor, ...],
    *,
    batches: int,
    micro_batch_size: int,
    precision: str,
    device: torch.device,
    distributed: DistributedContext,
) -> tuple[tuple[torch.Tensor, ...], int, int]:
    """Apply the Hessian of one fixed validation-prefix objective to ``vector``."""

    rank = int(distributed.rank)
    world_size = int(distributed.world_size)
    local_products: tuple[torch.Tensor, ...] | None = None
    allocation_error: str | None = None
    try:
        local_products = tuple(
            torch.zeros_like(parameter, memory_format=torch.preserve_format)
            for parameter in parameters
        )
    except torch.cuda.OutOfMemoryError:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        allocation_error = (
            "CUDA out of memory while allocating Hessian-vector-product workspace; "
            "reduce curvature.micro_batch_size or the selected endpoint count"
        )
    except Exception as exc:  # noqa: BLE001 - synchronize rank-local allocation failures
        allocation_error = f"{type(exc).__name__}: {exc}"
    _synchronize_computation_error(
        distributed,
        allocation_error,
        "Hessian-vector-product workspace allocation",
    )
    if local_products is None:
        raise RuntimeError(
            "Hessian-vector-product workspace allocation returned no vectors"
        )
    local_token_count = 0
    local_batch_count = 0
    local_error: str | None = None
    try:
        for batch_index in range(rank, batches, world_size):
            input_ids = corpus.batch(
                "validation",
                start_block=batch_index * micro_batch_size,
                batch_size=micro_batch_size,
                device=device,
            )
            products, token_count = _single_hessian_vector_product(
                model,
                input_ids,
                parameters,
                vector,
                precision=precision,
                device=device,
            )
            for accumulator, product in zip(local_products, products):
                accumulator.add_(product, alpha=float(token_count))
            local_token_count += token_count
            local_batch_count += 1
            del input_ids, products
    except torch.cuda.OutOfMemoryError:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        local_error = (
            "CUDA out of memory; reduce curvature.micro_batch_size, "
            "curvature.batches, or lambda_max_iterations"
        )
    except Exception as exc:  # noqa: BLE001 - synchronize rank-local failures
        local_error = f"{type(exc).__name__}: {exc}"
    _synchronize_computation_error(distributed, local_error, "Hessian-vector product")

    counts = torch.tensor(
        [float(local_token_count), float(local_batch_count)],
        dtype=torch.float64,
        device=device,
    )
    distributed.all_reduce_sum(counts)
    for product in local_products:
        distributed.all_reduce_sum(product)
    global_token_count = round(float(counts[0]))
    global_batch_count = round(float(counts[1]))
    if global_token_count < 1:
        raise RuntimeError(
            "distributed Hessian-vector product has no validation tokens"
        )
    if global_batch_count != batches:
        raise RuntimeError(
            "distributed Hessian-vector-product sharding is incomplete: "
            f"found {global_batch_count}/{batches} batches"
        )
    inverse_tokens = 1.0 / float(global_token_count)
    for product in local_products:
        product.mul_(inverse_tokens)
    return local_products, global_token_count, global_batch_count


def estimate_hessian_lambda_max(
    model: nn.Module,
    corpus: Any,
    config: dict[str, Any],
    device: torch.device,
    distributed: DistributedContext,
) -> dict[str, Any]:
    """Estimate the largest algebraic eigenvalue with deterministic Lanczos.

    The operator is the Hessian of the token-weighted mean cross-entropy over
    the fixed leading validation minibatches.  The three-term symmetric
    Lanczos recurrence stores only three parameter-sized vectors, which keeps
    the estimator usable for the larger model setting.  The returned residual
    is the standard Ritz residual estimate ``beta * |last_component|``.
    """

    normalized = validate_curvature_config(config)
    batches = int(normalized["batches"])
    micro_batch_size = int(normalized["micro_batch_size"])
    iterations = int(normalized["lambda_max_iterations"])
    tolerance = float(normalized["lambda_max_tolerance"])
    seed = int(normalized["seed"])
    precision = str(normalized["precision"])
    world_size = int(distributed.world_size)
    rank = int(distributed.rank)
    if world_size < 1 or not 0 <= rank < world_size:
        raise ValueError(
            f"invalid distributed context rank={rank}, world_size={world_size}"
        )
    validation_blocks = int(corpus.validation_blocks)
    if validation_blocks < 1:
        raise ValueError("validation corpus has no complete sequence blocks")
    if micro_batch_size > validation_blocks:
        raise ValueError(
            "curvature.micro_batch_size exceeds the validation corpus; "
            "refusing to wrap and repeat validation blocks"
        )
    maximum_batches = validation_blocks // micro_batch_size
    used_batches = min(batches, maximum_batches)
    parameters = tuple(
        parameter for parameter in model.parameters() if parameter.requires_grad
    )
    if not parameters:
        raise ValueError("largest Hessian eigenvalue requires trainable parameters")
    parameter_count = sum(parameter.numel() for parameter in parameters)

    current: tuple[torch.Tensor, ...] | None = None
    initialization_error: str | None = None
    try:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        initial = tuple(
            _rademacher_like(parameter, generator) for parameter in parameters
        )
        initial_norm = _vector_norm(initial, device)
        if initial_norm == 0.0:
            raise RuntimeError("Lanczos initialization produced a zero vector")
        current = _scaled_vector(initial, 1.0 / initial_norm)
        del initial
    except torch.cuda.OutOfMemoryError:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        initialization_error = (
            "CUDA out of memory while allocating Lanczos vectors; reduce curvature settings "
            "or the selected endpoint count"
        )
    except Exception as exc:  # noqa: BLE001 - synchronize rank-local initialization failures
        initialization_error = f"{type(exc).__name__}: {exc}"
    _synchronize_computation_error(
        distributed, initialization_error, "Lanczos initialization"
    )
    if current is None:
        raise RuntimeError("Lanczos initialization returned no vector")
    previous: tuple[torch.Tensor, ...] | None = None
    previous_beta = 0.0
    alphas: list[float] = []
    betas: list[float] = []
    ritz_values: list[float] = []
    estimate = 0.0
    residual = math.inf
    converged = False
    global_token_count = 0
    global_batch_count = 0
    training_flags = {module: bool(module.training) for module in model.modules()}
    try:
        model.eval()
        for iteration in range(iterations):
            product, global_token_count, global_batch_count = (
                _full_validation_hessian_vector_product(
                    model,
                    corpus,
                    parameters,
                    current,
                    batches=used_batches,
                    micro_batch_size=micro_batch_size,
                    precision=precision,
                    device=device,
                    distributed=distributed,
                )
            )
            iteration_error: str | None = None
            try:
                if previous is not None:
                    for value, old_direction in zip(product, previous):
                        value.add_(old_direction, alpha=-previous_beta)
                alpha = float(_vector_dot(current, product, device))
                for value, direction in zip(product, current):
                    value.add_(direction, alpha=-alpha)
                # One inexpensive local reorthogonalization against q_(k-1)
                # reduces finite-precision drift without retaining the full basis.
                if previous is not None:
                    correction = float(_vector_dot(previous, product, device))
                    for value, old_direction in zip(product, previous):
                        value.add_(old_direction, alpha=-correction)
                beta = _vector_norm(product, device)
                alphas.append(alpha)
                betas.append(beta)

                tridiagonal = torch.diag(torch.tensor(alphas, dtype=torch.float64))
                if len(alphas) > 1:
                    off_diagonal = torch.tensor(betas[:-1], dtype=torch.float64)
                    tridiagonal += torch.diag(off_diagonal, diagonal=1)
                    tridiagonal += torch.diag(off_diagonal, diagonal=-1)
                eigenvalues, eigenvectors = torch.linalg.eigh(tridiagonal)
                estimate = float(eigenvalues[-1])
                residual = beta * abs(float(eigenvectors[-1, -1]))
                ritz_values.append(estimate)
                threshold = tolerance * max(1.0, abs(estimate))
                converged = (
                    beta <= torch.finfo(parameters[0].dtype).eps
                    or residual <= threshold
                )
                if not converged:
                    previous, current = current, _scaled_vector(product, 1.0 / beta)
                    previous_beta = beta
                del product
            except torch.cuda.OutOfMemoryError:
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                iteration_error = (
                    "CUDA out of memory during a Lanczos update; reduce curvature settings "
                    "or the selected endpoint count"
                )
            except Exception as exc:  # noqa: BLE001 - synchronize each Lanczos round
                iteration_error = f"{type(exc).__name__}: {exc}"
            _synchronize_computation_error(
                distributed,
                iteration_error,
                f"Lanczos iteration {iteration + 1}",
            )
            if converged:
                break
    finally:
        for module, was_training in training_flags.items():
            module.training = was_training

    return {
        "lambda_max": float(estimate),
        "lambda_max_residual": float(residual),
        "lambda_max_converged": bool(converged),
        "lambda_max_iterations": len(alphas),
        "lambda_max_ritz_values": ritz_values,
        "metadata": {
            "estimator": LAMBDA_MAX_ESTIMATOR_NAME,
            "objective": OBJECTIVE_NAME,
            "parameter_state": _parameter_state_label(model),
            "parameter_count": int(parameter_count),
            "validation_batches": int(global_batch_count),
            "validation_micro_batch_size": int(micro_batch_size),
            "validation_token_count": int(global_token_count),
            "maximum_iterations": int(iterations),
            "relative_residual_tolerance": float(tolerance),
            "initial_distribution": "Rademacher{-1,+1}",
            "initial_seed": int(seed),
            "common_initial_vector_across_endpoints_and_ranks": True,
            "precision": precision,
            "world_size": world_size,
            "batch_sharding": "round_robin_by_global_batch_index",
            "eigenvalue_selection": "largest_algebraic_ritz_value",
            "reported_quantity": "largest_algebraic_eigenvalue_not_spectral_radius",
            "recurrence": "three_term_with_previous_vector_reorthogonalization",
            "iterations_run": len(alphas),
            "converged": bool(converged),
            "final_ritz_residual": float(residual),
            "ritz_residual_definition": "beta_next_times_abs_last_ritz_vector_component",
            "convergence_test": "residual <= tolerance * max(1, abs(lambda_max))",
        },
    }


def estimate_curvature(
    model: nn.Module,
    corpus: Any,
    config: dict[str, Any],
    device: torch.device,
    distributed: DistributedContext,
) -> dict[str, Any]:
    """Measure trace and largest eigenvalue under one strict shared protocol."""

    normalized = validate_curvature_config(config)
    trace = estimate_hessian_trace(
        model,
        corpus,
        {
            "batches": normalized["batches"],
            "micro_batch_size": normalized["micro_batch_size"],
            "probes": normalized["trace_probes"],
            "seed": normalized["seed"],
            "precision": normalized["precision"],
        },
        device,
        distributed,
    )
    lambda_max = estimate_hessian_lambda_max(
        model,
        corpus,
        normalized,
        device,
        distributed,
    )
    return {
        "hessian_trace": trace["trace"],
        "hessian_trace_std": trace["trace_std"],
        "hessian_trace_standard_error": trace["trace_standard_error"],
        "hessian_trace_ci95_lower": trace["trace_ci95_lower"],
        "hessian_trace_ci95_upper": trace["trace_ci95_upper"],
        "hessian_trace_probe_estimates": trace["probe_estimates"],
        "hessian_lambda_max": lambda_max["lambda_max"],
        "hessian_lambda_max_residual": lambda_max["lambda_max_residual"],
        "hessian_lambda_max_converged": lambda_max["lambda_max_converged"],
        "hessian_lambda_max_iterations": lambda_max["lambda_max_iterations"],
        "hessian_lambda_max_ritz_values": lambda_max["lambda_max_ritz_values"],
        "hessian_metadata": {
            "curvature_config": normalized,
            "trace": trace["metadata"],
            "lambda_max": lambda_max["metadata"],
        },
    }
