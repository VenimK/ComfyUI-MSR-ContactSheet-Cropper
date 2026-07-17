"""Typed data structures for MSR contact sheet nodes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict


class ComfyInputSpec(TypedDict, total=False):
    """Shape of a single ComfyUI INPUT_TYPES entry value."""

    default: bool | int | str
    min: int
    max: int
    step: int
    multiline: bool
    forceInput: bool
    tooltip: str


@dataclass(frozen=True)
class PanelCoords:
    """Pixel coordinates for one panel."""

    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top


@dataclass(frozen=True)
class GridSpec:
    """Resolved grid specification for a contact sheet."""

    cols: int
    rows: int
    width: int
    height: int
