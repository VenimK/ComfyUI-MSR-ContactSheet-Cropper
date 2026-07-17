"""ComfyUI custom nodes for cropping and assembling MSR Contact Sheets.

This module registers the MSR node pack with ComfyUI. All implementation
lives in the `msr_nodes` package so the root file stays small and readable.
"""

from __future__ import annotations

import logging

from msr_nodes.assembler import MSRContactSheetAssembler
from msr_nodes.constants import (
    DEFAULT_GRID_SIZE,
    LAYOUT_PRESETS,
    PANEL_OUTPUT_NAMES,
)
from msr_nodes.cropper import MSRContactSheetCropper

logger = logging.getLogger(__name__)

NODE_CLASS_MAPPINGS = {
    "MSRContactSheetCropper": MSRContactSheetCropper,
    "MSRContactSheetAssembler": MSRContactSheetAssembler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MSRContactSheetCropper": "MSR Crop Contact Sheet",
    "MSRContactSheetAssembler": "MSR Assemble Contact Sheet",
}

__all__ = [
    "MSRContactSheetCropper",
    "MSRContactSheetAssembler",
    "DEFAULT_GRID_SIZE",
    "LAYOUT_PRESETS",
    "PANEL_OUTPUT_NAMES",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]

logger.info("Licon-MSR Utils loaded: %s", list(NODE_CLASS_MAPPINGS.keys()))
