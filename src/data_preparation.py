"""Download and tokenize FineWeb-Edu for the OPT experiments."""

from __future__ import annotations

import fcntl
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

DATASET_NAME = "HuggingFaceFW/fineweb-edu"
DATASET_CONFIG = "sample-100BT"
DATASET_REVISION = "fc9850dff5e2d0f8f776efe41b24a1c49556cfc5"
TOKENIZER_NAME = "facebook/opt-125m"
TOKENIZER_REVISION = "d76d9f3da23d0f5eb5528dc2c9a61269e911fe49"
SEED = 20260722
SHUFFLE_BUFFER = 10_000
VALIDATION_FRACTION = 0.02


class DataPreparationError(RuntimeError):
    """The data cache is incomplete or incompatible with the requested run."""


def _path(value: str, config: dict) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(config.get("_project_root", Path.cwd())) / path
    return path.resolve()


def _read_json(path: Path) -> dict:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise DataPreparationError(f"cannot read {path}: {exc}") from exc
    if not isinstance(result, dict):
        raise DataPreparationError(f"expected a JSON object in {path}")
    return result


def _write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _identity(config: dict) -> dict:
    preparation = config["data"]["preparation"]
    return {
        "dataset_name": DATASET_NAME,
        "dataset_config": DATASET_CONFIG,
        "dataset_revision": preparation.get("dataset_revision", DATASET_REVISION),
        "split": "train",
        "text_field": "text",
        "id_field": "id",
        "shuffle_buffer": SHUFFLE_BUFFER,
        "seed": SEED,
        "train_tokens": int(preparation.get("minimum_train_tokens", 20_000_000_000)),
        "validation_tokens": int(preparation.get("validation_tokens", 2_000_000)),
        "validation_fraction": VALIDATION_FRACTION,
        "append_eos": True,
        "tokenizer_name": TOKENIZER_NAME,
        "tokenizer_revision": TOKENIZER_REVISION,
        "tokenizer_vocab_size": int(config["data"]["tokenizer_vocab_size"]),
        "storage_dtype": "int32",
    }


def verify_token_cache(config: dict[str, Any]) -> dict[str, Any]:
    """Check cache sizes/settings without reading entire binary files."""
    data = config["data"]
    root = _path(data["root"], config)
    manifest = _read_json(root / data["manifest"])
    identity = manifest.get("identity", {})
    if identity.get("storage_dtype") != "int32" or data["storage_dtype"] != "int32":
        raise DataPreparationError("token caches must use int32")
    if identity.get("tokenizer_vocab_size") != int(data["tokenizer_vocab_size"]):
        raise DataPreparationError(
            "tokenizer vocabulary differs from the configured model"
        )
    counts = {}
    for split in ("train", "validation"):
        path = root / data[f"{split}_file"]
        if not path.is_file():
            raise DataPreparationError(f"missing token cache: {path}")
        size = path.stat().st_size
        if size <= 0 or size % 4 or size != int(manifest[f"{split}_file_bytes"]):
            raise DataPreparationError(f"incomplete int32 token cache: {path}")
        counts[split] = size // 4
    budget = int(config.get("training", {}).get("train_token_budget", 0))
    if counts["train"] < budget:
        raise DataPreparationError(
            f"cache has {counts['train']:,} training tokens; need {budget:,}"
        )
    if int(manifest.get("actual_train_tokens", counts["train"])) != counts["train"]:
        raise DataPreparationError(
            "manifest token count does not match the training file"
        )
    preparation = data.get("preparation", {})
    if preparation.get("mode") == "fineweb_edu_local_snapshot":
        for key, expected in _identity(config).items():
            if identity.get(key) != expected:
                raise DataPreparationError(
                    f"cache setting {key} differs; use another data.root"
                )
        if (
            counts["train"] != identity["train_tokens"]
            or counts["validation"] != identity["validation_tokens"]
        ):
            raise DataPreparationError(
                "cache sizes differ from the requested token counts"
            )
    print(
        f"[prepare-data] ready: {counts['train']:,} train / {counts['validation']:,} validation tokens",
        flush=True,
    )
    return manifest


def _download_snapshot(raw_dir: Path, revision: str) -> None:
    from huggingface_hub import snapshot_download

    if raw_dir.parts[-2:] != ("sample", "100BT"):
        raise DataPreparationError("data.preparation.raw_dir must end in sample/100BT")
    print(
        f"[prepare-data] downloading/reusing {DATASET_NAME} {DATASET_CONFIG}",
        flush=True,
    )
    snapshot_download(
        repo_id=DATASET_NAME,
        repo_type="dataset",
        revision=revision,
        allow_patterns=["sample/100BT/*.parquet"],
        local_dir=raw_dir.parent.parent,
        max_workers=8,
    )


def _load_tokenizer(config: dict):
    from transformers import AutoTokenizer

    assets = config["data"]["preparation"].get(
        "tokenizer_assets_dir", "scripts/opt_125m_tokenizer"
    )
    assets_dir = _path(assets, config)
    if not assets_dir.is_dir():
        raise DataPreparationError(f"missing bundled OPT tokenizer: {assets_dir}")
    tokenizer = AutoTokenizer.from_pretrained(
        assets_dir, local_files_only=True, use_fast=True
    )
    if (
        len(tokenizer) != int(config["data"]["tokenizer_vocab_size"])
        or tokenizer.eos_token_id is None
    ):
        raise DataPreparationError("tokenizer vocabulary or EOS token is incompatible")
    return tokenizer


def _build_stream(shards: list[Path]):
    from datasets import load_dataset

    return load_dataset(
        "parquet",
        data_files={"train": [str(path) for path in shards]},
        split="train",
        streaming=True,
    ).shuffle(seed=SEED, buffer_size=SHUFFLE_BUFFER)


def _is_validation(key: str) -> bool:
    # Stable document split from the original experiments, not an integrity check.
    bucket = hashlib.blake2b(
        key.encode("utf-8", errors="replace"), digest_size=8
    ).digest()
    return int.from_bytes(bucket, "big") / float(2**64) < VALIDATION_FRACTION


def _tokenize(
    config: dict, root: Path, shards: list[Path], tokenizer, batch_size: int = 256
) -> None:
    data = config["data"]
    identity = _identity(config)
    targets = {
        split: int(identity[f"{split}_tokens"]) for split in ("train", "validation")
    }
    if min(targets.values()) < 1 or batch_size < 1:
        raise DataPreparationError(
            "token counts and tokenization batch size must be positive"
        )
    source_files = [
        {"name": path.name, "bytes": path.stat().st_size} for path in shards
    ]
    finals = {split: root / data[f"{split}_file"] for split in targets}
    temporaries = {
        split: path.with_suffix(path.suffix + ".tmp") for split, path in finals.items()
    }
    progress_path = root / "tokenization_progress.json"
    offsets = dict.fromkeys(targets, 0)
    documents = 0
    complete = False
    resuming = progress_path.is_file()
    if resuming:
        progress = _read_json(progress_path)
        if (
            progress.get("identity") != identity
            or progress.get("source_files") != source_files
        ):
            raise DataPreparationError(
                "unfinished tokenization uses different settings or input files"
            )
        offsets = progress["offsets"]
        documents = int(progress["documents"])
        complete = bool(progress.get("complete", False))
        for split, target in targets.items():
            if not 0 <= int(offsets[split]) <= target:
                raise DataPreparationError("invalid tokenization progress offsets")
    elif any(path.exists() for path in [*finals.values(), *temporaries.values()]):
        raise DataPreparationError(
            "token files exist without a manifest/progress record; choose another data.root"
        )

    def save_progress(done: bool = False) -> None:
        _write_json(
            progress_path,
            {
                "identity": identity,
                "source_files": source_files,
                "offsets": offsets,
                "documents": documents,
                "complete": done,
            },
        )

    if not complete:
        maps = {}
        for split, target in targets.items():
            path = temporaries[split]
            if resuming and (not path.is_file() or path.stat().st_size != 4 * target):
                raise DataPreparationError(
                    f"missing or incomplete tokenization work file: {path}"
                )
            maps[split] = np.memmap(
                path, dtype=np.int32, mode="r+" if resuming else "w+", shape=(target,)
            )
        save_progress()
        stream = iter(_build_stream(shards))
        if documents:
            print(
                f"[prepare-data] replaying {documents:,} rows to restore the shuffle buffer",
                flush=True,
            )
        for _ in range(documents):
            try:
                next(stream)
            except StopIteration as exc:
                raise DataPreparationError(
                    "input ended before the saved resume position"
                ) from exc
        batches = 0
        try:
            while any(offsets[split] < targets[split] for split in targets):
                examples = []
                for _ in range(batch_size):
                    try:
                        example = next(stream)
                    except StopIteration:
                        break
                    if example.get("text") is None:
                        raise DataPreparationError("FineWeb row has no text")
                    examples.append(example)
                if not examples:
                    raise DataPreparationError(
                        "FineWeb stream ended before the requested token counts"
                    )
                encoded = tokenizer(
                    [str(example["text"]) for example in examples],
                    add_special_tokens=False,
                    return_attention_mask=False,
                    return_token_type_ids=False,
                )["input_ids"]
                for example, token_ids in zip(examples, encoded):
                    if not token_ids:
                        continue
                    tokens = [*token_ids, int(tokenizer.eos_token_id)]
                    key = str(example.get("id") or example["text"])
                    if (
                        _is_validation(key)
                        and offsets["validation"] < targets["validation"]
                    ):
                        split = "validation"
                    elif offsets["train"] < targets["train"]:
                        split = "train"
                    else:
                        split = "validation"
                    count = min(len(tokens), targets[split] - offsets[split])
                    start = offsets[split]
                    maps[split][start : start + count] = tokens[:count]
                    offsets[split] += count
                documents += len(examples)
                batches += 1
                if batches % 100 == 0:
                    for mapping in maps.values():
                        mapping.flush()
                    save_progress()
                    print(
                        f"[prepare-data] train={offsets['train']:,}/{targets['train']:,}; validation={offsets['validation']:,}/{targets['validation']:,}",
                        flush=True,
                    )
            for mapping in maps.values():
                mapping.flush()
            save_progress(done=True)
        finally:
            for mapping in maps.values():
                mapping.flush()
                mapping._mmap.close()
    for split, target in targets.items():
        candidate = (
            temporaries[split] if temporaries[split].is_file() else finals[split]
        )
        if not candidate.is_file() or candidate.stat().st_size != target * 4:
            raise DataPreparationError(f"incomplete tokenized data: {candidate}")
        if candidate != finals[split]:
            candidate.replace(finals[split])
    tokenizer.save_pretrained(root / "tokenizer")
    _write_json(
        root / data["manifest"],
        {
            "identity": identity,
            "train_file_bytes": finals["train"].stat().st_size,
            "validation_file_bytes": finals["validation"].stat().st_size,
            "actual_train_tokens": targets["train"],
            "source": {"kind": "local_parquet_snapshot", "files": source_files},
        },
    )
    progress_path.unlink(missing_ok=True)


def prepare_data(config: dict[str, Any]) -> dict[str, Any]:
    """Prepare once or reuse a complete cache, without starting CUDA or training."""
    data = config["data"]
    root = _path(data["root"], config)
    root.mkdir(parents=True, exist_ok=True)
    preparation = data["preparation"]
    mode = preparation["mode"]
    with (root / ".prepare_data.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if mode == "verify_existing" or (root / data["manifest"]).is_file():
            return verify_token_cache(config)
        if mode != "fineweb_edu_local_snapshot":
            raise DataPreparationError(f"unknown data preparation mode: {mode}")
        tokenizer = _load_tokenizer(config)
        raw_dir = _path(preparation["raw_dir"], config)
        _download_snapshot(
            raw_dir, preparation.get("dataset_revision", DATASET_REVISION)
        )
        shards = sorted(raw_dir.glob("*.parquet"))
        if not shards:
            raise DataPreparationError(f"no Parquet files found in {raw_dir}")
        _tokenize(config, root, shards, tokenizer)
        return verify_token_cache(config)
