from __future__ import annotations

import os
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .utils import atomic_json, exclusive_lock, read_json


class TokenBlocks:
    def __init__(self, path: str | Path, sequence_length: int, num_blocks: int):
        self.path = Path(path)
        self.sequence_length = int(sequence_length)
        self.num_blocks = int(num_blocks)
        self.usable_input_tokens = self.sequence_length * self.num_blocks
        self.target_tokens = (self.sequence_length - 1) * self.num_blocks
        if self.path.stat().st_size < self.usable_input_tokens * 4:
            raise ValueError(f"not enough tokens in {self.path}")
        values = np.memmap(
            self.path, dtype="<i4", mode="c", shape=(self.usable_input_tokens,)
        )
        self.tokens = torch.from_numpy(values).view(
            self.num_blocks, self.sequence_length
        )

    def batch(self, start: int, count: int, device: torch.device) -> torch.Tensor:
        return self.tokens[start : start + count].to(device=device, dtype=torch.long)


def prepare_data(config: dict[str, Any], binding: dict[str, Any]) -> tuple[dict, dict]:
    """Both methods call this function and therefore share exactly the same caches."""
    calibration = config["calibration"]
    refined = config["evaluation"]["refinedweb_heldout"]
    tokenizer_record = dict(binding["tokenizer"])
    if not tokenizer_record.get("tokenizer_name"):
        tokenizer_record["assets_dir"] = binding["tokenizer_assets_dir"]
    common = {
        "tokenizer": tokenizer_record,
        "tokenization": "no_special_tokens_document_eos_concat",
        "sequence_length": binding["sequence_length"],
        "storage_dtype": "int32",
    }
    calibration_protocol = {
        **common,
        **{
            key: calibration[key]
            for key in (
                "dataset_name",
                "dataset_config",
                "split",
                "revision",
                "text_field",
                "seed",
                "shuffle_buffer_size",
            )
        },
        "num_blocks": calibration["num_sequences"],
        "selection": "streaming_shuffle_prefix",
    }
    refined_protocol = {
        **common,
        **{
            key: refined[key]
            for key in ("dataset_name", "revision", "split", "text_field", "shard_path")
        },
        "num_blocks": binding["num_blocks"],
        "selection": "single_shard_physical_row_order_prefix",
    }
    prepared = []
    tokenizer = None
    for role, protocol, options, filename in (
        ("C4", calibration_protocol, calibration, "calibration_tokens.int32.bin"),
        (
            "RefinedWeb",
            refined_protocol,
            refined,
            "refinedweb_heldout_tokens.int32.bin",
        ),
    ):
        root = Path(options["cache_dir"]).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        with exclusive_lock(root / ".prepare.lock"):
            path, manifest_path = root / filename, root / "manifest.json"
            count = protocol["num_blocks"] * protocol["sequence_length"]
            if manifest_path.is_file():
                manifest = read_json(manifest_path)
                if manifest.get("protocol") != protocol:
                    raise ValueError(
                        f"{role} cache settings changed; choose a fresh cache_dir: {root}"
                    )
                if not path.is_file() or path.stat().st_size != count * 4:
                    raise ValueError(
                        f"{role} token cache is missing or truncated: {path}"
                    )
                if manifest.get("complete") is not True:
                    raise ValueError(f"{role} token cache is incomplete: {root}")
                prepared.append(manifest)
                continue
            if path.exists():
                raise ValueError(f"token file exists without manifest.json: {path}")
            if tokenizer is None:
                tokenizer = _load_tokenizer(binding)
            texts = (
                _c4_texts(options, root)
                if role == "C4"
                else _refinedweb_texts(options, root)
            )
            print(f"[data] preparing {role}: {count:,} tokens", flush=True)
            _write_tokens(texts, tokenizer, path, count)
            manifest = {
                "complete": True,
                "protocol": protocol,
                "file": filename,
                "file_bytes": count * 4,
                "num_tokens": count,
            }
            atomic_json(manifest_path, manifest)
            prepared.append(manifest)
    return prepared[0], prepared[1]


def _load_tokenizer(binding: dict[str, Any]):
    from transformers import AutoTokenizer

    record = binding["tokenizer"]
    assets = binding["tokenizer_assets_dir"]
    if assets:
        tokenizer = AutoTokenizer.from_pretrained(
            assets, use_fast=True, local_files_only=True
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            record["tokenizer_name"],
            revision=record.get("tokenizer_revision"),
            use_fast=True,
        )
    if (
        record.get("tokenizer_vocab_size")
        and len(tokenizer) != record["tokenizer_vocab_size"]
    ):
        raise ValueError("PTQ tokenizer vocabulary differs from the training tokenizer")
    for key in ("bos_token_id", "eos_token_id", "pad_token_id"):
        if key in record and getattr(tokenizer, key) != record[key]:
            raise ValueError(f"PTQ tokenizer {key} differs from the training tokenizer")
    if tokenizer.eos_token_id is None:
        raise ValueError("PTQ tokenizer needs an EOS token")
    return tokenizer


def _c4_texts(options: dict[str, Any], root: Path) -> Iterable[str]:
    from datasets import load_dataset

    stream = load_dataset(
        options["dataset_name"],
        options["dataset_config"],
        split=options["split"],
        revision=options["revision"],
        streaming=True,
        cache_dir=str(root / "huggingface"),
    ).shuffle(seed=options["seed"], buffer_size=options["shuffle_buffer_size"])
    for example in stream:
        yield example[options["text_field"]]


def _refinedweb_texts(options: dict[str, Any], root: Path) -> Iterable[str]:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=options["dataset_name"],
        repo_type="dataset",
        filename=options["shard_path"],
        revision=options["revision"],
        cache_dir=root / "huggingface",
    )
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(
        batch_size=1024, columns=[options["text_field"]], use_threads=False
    ):
        yield from batch.column(0).to_pylist()


def _write_tokens(texts: Iterable[str], tokenizer: Any, path: Path, count: int) -> None:
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    os.close(descriptor)
    try:
        output = np.memmap(name, dtype="<i4", mode="w+", shape=(count,))
        offset = 0
        for text in texts:
            if not isinstance(text, str):
                raise ValueError("dataset text field contains a non-string value")
            encoded = tokenizer(
                text,
                add_special_tokens=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )["input_ids"]
            if not encoded:
                continue
            tokens = [*encoded, tokenizer.eos_token_id]
            take = min(len(tokens), count - offset)
            output[offset : offset + take] = tokens[:take]
            offset += take
            if offset == count:
                break
        if offset != count:
            raise ValueError(f"dataset ended after {offset} tokens; expected {count}")
        output.flush()
        del output
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)
