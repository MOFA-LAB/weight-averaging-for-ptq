from __future__ import annotations

from typing import Any, TypeAlias

BitWidth: TypeAlias = int

SUPPORTED_BIT_WIDTHS: tuple[BitWidth, ...] = (4, 3, 2)


def canonical_bit_width(value: Any) -> BitWidth:
    """Return the one protocol spelling for a supported nominal bit width."""

    if type(value) is int and value in {4, 3, 2}:
        return value
    raise ValueError("bit width must be one of 4, 3, or 2")


def canonical_bit_roster(values: list[Any]) -> list[BitWidth]:
    normalized = [canonical_bit_width(value) for value in values]
    requested = set(normalized)
    return [value for value in SUPPORTED_BIT_WIDTHS if value in requested]


def bit_label(value: Any) -> str:
    return str(canonical_bit_width(value))


def bit_file_tag(value: Any) -> str:
    return bit_label(value)


def quantization_levels(value: Any) -> int:
    return 1 << canonical_bit_width(value)
