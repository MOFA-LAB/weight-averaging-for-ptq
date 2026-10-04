from __future__ import annotations

import copy
import csv
import gc
import math
import os
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .config import load_config, resume_config
from .data import TokenBlocks, prepare_data
from .source import discover_sources, load_endpoint_state
from .utils import atomic_json, exclusive_lock, read_json


def run(
    config_path: str | Path, *, method: str, prepare_data_only: bool = False
) -> dict[str, Any]:
    config = load_config(config_path, method)
    sources = discover_sources(config, data_only=prepare_data_only)
    calibration_manifest, refined_manifest = prepare_data(config, sources["validation"])
    if prepare_data_only:
        print("[data] C4 and RefinedWeb caches are ready", flush=True)
        return {"calibration": calibration_manifest, "refinedweb": refined_manifest}
    output = Path(config["project"]["output_dir"]).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(output / ".run.lock"):
        protocol = {
            "method": method,
            "config": resume_config(config),
            "sources": sources,
            "calibration": calibration_manifest,
            "refinedweb": refined_manifest,
        }
        record = output / "run_config.json"
        if record.is_file():
            if not config["project"]["resume"]:
                raise ValueError(
                    "output already exists; enable project.resume or choose another output_dir"
                )
            if read_json(record) != protocol:
                raise ValueError(
                    "PTQ settings or source roster changed; choose a fresh project.output_dir"
                )
        else:
            if any(path.name != ".run.lock" for path in output.iterdir()):
                raise ValueError(
                    "output directory is not empty and has no run_config.json"
                )
            atomic_json(record, protocol)
        atomic_json(output / "resolved_config.json", config)
        device = _initialize_runtime(config)
        precision = config["evaluation"].get(
            "precision", config["runtime"].get("precision", "bf16")
        )
        if device.type == "cpu":
            precision = "fp32"
        binding = sources["validation"]
        length, blocks = binding["sequence_length"], binding["num_blocks"]
        corpora = {
            "fineweb_edu_validation": TokenBlocks(binding["path"], length, blocks),
            "refinedweb_heldout": TokenBlocks(
                Path(
                    config["evaluation"]["refinedweb_heldout"]["cache_dir"]
                ).expanduser()
                / refined_manifest["file"],
                length,
                blocks,
            ),
        }
        calibration = TokenBlocks(
            Path(config["calibration"]["cache_dir"]).expanduser()
            / calibration_manifest["file"],
            length,
            config["calibration"]["num_sequences"],
        )
        from src.model import build_model

        model_config = copy.deepcopy(sources["model"])
        model_config["gradient_checkpointing"] = False
        results = []
        for endpoint in sources["endpoints"]:
            print(f"[{method.upper()}] {endpoint['endpoint_key']}", flush=True)
            directory = output / "results" / endpoint["run_id"] / endpoint["name"]
            directory.mkdir(parents=True, exist_ok=True)
            baseline = _read_result(directory / "fp.json", endpoint, None)
            trials = {
                bits: _read_result(directory / f"w{bits}.json", endpoint, bits)
                for bits in config["quantization"]["bits"]
            }
            if baseline is not None and all(trials.values()):
                results.extend([baseline, *trials.values()])
                print("[resume] endpoint complete", flush=True)
                continue
            model = build_model(model_config, seed=int(config["project"]["seed"]))
            state = load_endpoint_state(endpoint)
            model.load_state_dict(state, strict=True)
            model.to(device=device, dtype=torch.float32).eval()
            if baseline is None:
                baseline = {
                    "method": method,
                    "endpoint": endpoint,
                    "bits": None,
                    "metrics": _evaluate(model, corpora, config, device, precision),
                }
                atomic_json(directory / "fp.json", baseline)
            results.append(baseline)
            moments = None
            if method == "awq" and not all(trials.values()):
                from PTQ.AWQ.awq_ptq.quantize import collect_activation_second_moments

                moments = collect_activation_second_moments(
                    model,
                    calibration,
                    device=device,
                    micro_batch_size=config["calibration"]["micro_batch_size"],
                    excluded_modules=config["quantization"]["exclude_modules"],
                    precision="fp32" if precision == "fp32" else "bf16",
                    reconstruction_tokens=config["quantization"][
                        "reconstruction_tokens"
                    ],
                )
            for bits in config["quantization"]["bits"]:
                if trials[bits] is not None:
                    results.append(trials[bits])
                    continue
                # Each width starts from exactly the floating endpoint, never from a prior trial.
                model.load_state_dict(state, strict=True)
                print(f"[{method.upper()}] W{bits}", flush=True)
                quantization = _quantize(
                    model, calibration, moments, config, method, bits, device, precision
                )
                metrics = _evaluate(model, corpora, config, device, precision)
                result = {
                    "method": method,
                    "endpoint": endpoint,
                    "bits": bits,
                    "metrics": metrics,
                    "quantization": quantization,
                    "delta_from_baseline": {
                        split: {
                            key: values[key] - baseline["metrics"][split][key]
                            for key in ("loss", "perplexity")
                        }
                        for split, values in metrics.items()
                    },
                }
                quant = config["quantization"]
                if quant.get(
                    "save_quantized_model", quant.get("save_dequantized_state", False)
                ):
                    artifact = directory / f"w{bits}_fake_dequant.pt"
                    torch.save(
                        {
                            "model": {
                                key: value.detach().cpu()
                                for key, value in model.state_dict().items()
                            },
                            "metadata": {
                                "method": method,
                                "bits": bits,
                                "packed_inference_format": False,
                                "source_endpoint": endpoint,
                            },
                        },
                        artifact,
                    )
                    result["artifact"] = str(artifact.relative_to(output))
                atomic_json(directory / f"w{bits}.json", result)
                results.append(result)
            del model, state, moments
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        report = {"complete": True, "method": method, "results": results}
        atomic_json(output / "summary.json", report)
        _write_tables(output, results, config["quantization"]["bits"])
        atomic_json(
            output / "COMPLETED", {"status": "completed", "trials": len(results)}
        )
        print((output / "ppl_comparison.md").read_text(encoding="utf-8"), flush=True)
        return report


def _initialize_runtime(config: dict[str, Any]) -> torch.device:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    seed = int(config["project"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    runtime = config["runtime"]
    torch.use_deterministic_algorithms(runtime.get("deterministic_algorithms", True))
    if runtime["device"] == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("runtime.device=cuda, but CUDA is unavailable")
    gpu = int(runtime.get("gpu_id", 0))
    if gpu >= torch.cuda.device_count():
        raise ValueError(f"runtime.gpu_id {gpu} is not visible")
    torch.cuda.set_device(gpu)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = bool(runtime.get("tf32", False))
    torch.backends.cudnn.allow_tf32 = bool(runtime.get("tf32", False))
    return torch.device("cuda", gpu)


@torch.no_grad()
def _evaluate(model, corpora, config, device, precision) -> dict[str, Any]:
    result = {}
    model.eval()
    for split, blocks in corpora.items():
        total_nll, total_targets = 0.0, 0
        micro = config["evaluation"]["micro_batch_size"]
        for start in range(0, blocks.num_blocks, micro):
            count = min(micro, blocks.num_blocks - start)
            inputs = blocks.batch(start, count, device)
            autocast = (
                torch.autocast("cuda", dtype=torch.bfloat16)
                if precision != "fp32" and device.type == "cuda"
                else nullcontext()
            )
            with autocast:
                logits = model(input_ids=inputs, use_cache=False).logits[:, :-1, :]
            labels = inputs[:, 1:]
            nll = F.cross_entropy(
                logits.float().reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                reduction="sum",
            )
            if not torch.isfinite(nll).item():
                raise RuntimeError(f"non-finite evaluation loss on {split}")
            total_nll += float(nll)
            total_targets += labels.numel()
        loss = total_nll / total_targets
        result[split] = {
            "loss": loss,
            "perplexity": math.exp(min(loss, 700)),
            "perplexity_capped": loss > 700,
            "target_tokens": total_targets,
            "num_blocks": blocks.num_blocks,
            "sequence_length": blocks.sequence_length,
        }
    return result


def _quantize(model, calibration, moments, config, method, bits, device, precision):
    quant = config["quantization"]
    if method == "gptq":
        from PTQ.GPTQ.gptq_ptq.quantizer import gptq_quantize_model

        modules = gptq_quantize_model(
            model,
            calibration.tokens,
            bits=bits,
            group_size=quant["group_size"],
            damp_percent=quant["damp_percent"],
            act_order=quant["act_order"],
            symmetric=quant["symmetric"],
            column_block_size=quant["column_block_size"],
            micro_batch_size=config["calibration"]["micro_batch_size"],
            precision="fp32" if precision == "fp32" else "bf16_mixed",
            device=device,
        )
        return {
            "algorithm": quant["algorithm"],
            "bits": bits,
            "group_size": quant["group_size"],
            "packed": False,
            "modules": modules,
        }
    from PTQ.AWQ.awq_ptq.quantize import apply_awq_fake_dequant

    return apply_awq_fake_dequant(model, moments, bits=bits, config=quant)


def _read_result(path: Path, endpoint: dict, bits: int | None):
    if not path.is_file():
        return None
    try:
        record = read_json(path)
        if record.get("endpoint") != endpoint or record.get("bits") != bits:
            return None
        for split in ("fineweb_edu_validation", "refinedweb_heldout"):
            for key in ("loss", "perplexity"):
                if not math.isfinite(record["metrics"][split][key]):
                    return None
        if (
            record.get("artifact")
            and not (path.parents[3] / record["artifact"]).is_file()
        ):
            return None
        return record
    except (KeyError, TypeError, ValueError):
        return None


def _write_tables(output: Path, results: list[dict], widths: list[int]) -> None:
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["endpoint", "category", "bits", "dataset", "loss", "perplexity"]
        )
        for item in results:
            for split, metrics in item["metrics"].items():
                writer.writerow(
                    [
                        item["endpoint"]["endpoint_key"],
                        item["endpoint"]["category"],
                        item["bits"] if item["bits"] is not None else "FP",
                        split,
                        metrics["loss"],
                        metrics["perplexity"],
                    ]
                )
    lines = ["# Perplexity comparison", ""]
    for split in ("fineweb_edu_validation", "refinedweb_heldout"):
        lines.extend(
            [
                f"## {split}",
                "",
                "| Endpoint | FP | " + " | ".join(f"W{bit}" for bit in widths) + " |",
                "| --- | ---: | " + " | ".join("---:" for _ in widths) + " |",
            ]
        )
        grouped = {}
        for item in results:
            grouped.setdefault(item["endpoint"]["endpoint_key"], {})[item["bits"]] = (
                item["metrics"][split]["perplexity"]
            )
        for name, values in grouped.items():
            lines.append(
                f"| {name} | "
                + " | ".join(f"{values[bit]:.4f}" for bit in [None, *widths])
                + " |"
            )
        lines.append("")
    (output / "ppl_comparison.md").write_text("\n".join(lines), encoding="utf-8")
