from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass
class TokenCorpus:
    train: torch.Tensor
    validation: torch.Tensor
    sequence_length: int
    manifest: dict[str, Any]

    def __post_init__(self) -> None:
        if self.sequence_length < 2:
            raise ValueError("sequence_length must be at least two")
        train_count = self.train.numel() // self.sequence_length * self.sequence_length
        validation_count = (
            self.validation.numel() // self.sequence_length * self.sequence_length
        )
        if train_count < self.sequence_length:
            raise ValueError("training token cache contains no complete sequence")
        if validation_count < self.sequence_length:
            raise ValueError("validation token cache contains no complete sequence")
        self.train = self.train[:train_count].view(-1, self.sequence_length)
        self.validation = self.validation[:validation_count].view(
            -1, self.sequence_length
        )

    @property
    def train_blocks(self) -> int:
        return int(self.train.shape[0])

    @property
    def validation_blocks(self) -> int:
        return int(self.validation.shape[0])

    def batch(
        self,
        split: str,
        start_block: int,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if split not in {"train", "validation"}:
            raise ValueError(f"unsupported split: {split}")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        blocks = self.train if split == "train" else self.validation
        count = int(blocks.shape[0])
        indices = (torch.arange(batch_size) + int(start_block)) % count
        return blocks.index_select(0, indices).to(
            device=device,
            dtype=torch.long,
            non_blocking=True,
        )


def load_corpus(config: dict[str, Any], sequence_length: int) -> TokenCorpus:
    root = Path(str(config["root"])).expanduser().resolve()
    manifest_path = root / str(config["manifest"])
    train_path = root / str(config["train_file"])
    validation_path = root / str(config["validation_file"])
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"missing tokenization manifest: {manifest_path}; upload the complete cache directory"
        ) from exc
    if not isinstance(manifest, dict):
        raise TypeError("tokenization manifest must be a JSON object")
    expected_train_bytes = int(manifest["train_file_bytes"])
    expected_validation_bytes = int(manifest["validation_file_bytes"])
    _verify_file(train_path, expected_train_bytes)
    _verify_file(validation_path, expected_validation_bytes)
    metadata = manifest.get("identity", {})
    if metadata.get("storage_dtype", "int32") != "int32":
        raise ValueError("token cache must use int32 storage")
    if int(metadata.get("tokenizer_vocab_size", config["tokenizer_vocab_size"])) != int(
        config["tokenizer_vocab_size"]
    ):
        raise ValueError("tokenizer vocabulary differs between config and cache")
    train = torch.from_numpy(np.memmap(train_path, dtype=np.int32, mode="c"))
    validation = torch.from_numpy(np.memmap(validation_path, dtype=np.int32, mode="c"))
    return TokenCorpus(train, validation, int(sequence_length), manifest)


def _verify_file(
    path: Path,
    expected_bytes: int,
) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing binary token cache: {path}")
    actual = path.stat().st_size
    if actual != expected_bytes:
        raise RuntimeError(
            f"incomplete token cache {path}: expected {expected_bytes} bytes, got {actual}"
        )
    if actual == 0 or actual % np.dtype(np.int32).itemsize:
        raise ValueError(f"token cache is empty or not an int32 array: {path}")
