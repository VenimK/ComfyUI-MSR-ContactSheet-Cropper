"""MSR Contact Sheet Assembler ComfyUI node."""

from __future__ import annotations

import logging
from typing import Any

import torch
from PIL import Image

from msr_nodes.constants import LAYOUT_PRESETS, PANEL_OUTPUT_NAMES
from msr_nodes.grid import parse_panel_mapping
from msr_nodes.image_utils import pil_to_tensor, tensor_to_pil
from msr_nodes.types import ComfyInputSpec

logger = logging.getLogger(__name__)


class MSRContactSheetAssembler:
    """ComfyUI node that composites panels back into a contact sheet."""

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "background": ("IMAGE",),
                "slot_1": ("IMAGE",),
                "slot_2": ("IMAGE",),
                "slot_3": ("IMAGE",),
                "slot_4": ("IMAGE",),
            },
            "optional": {
                "layout": (
                    ["ideogram_3x3", "midjourney_4x4", "custom_3x3"],
                    ComfyInputSpec({"default": "ideogram_3x3"}),
                ),
                "panel_mapping": (
                    "STRING",
                    ComfyInputSpec({"default": "0,1,2,4,5", "multiline": False}),
                ),
                "cell_size": (
                    "INT",
                    ComfyInputSpec({"default": 333, "min": 1, "max": 4096, "step": 1}),
                ),
                "background_color": (
                    "INT",
                    ComfyInputSpec({"default": 0, "min": 0, "max": 255, "step": 1}),
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("contact_sheet", "status")
    FUNCTION = "assemble"
    CATEGORY = "Licon-MSR / Assemble"

    def assemble(
        self,
        background: torch.Tensor,
        slot_1: torch.Tensor,
        slot_2: torch.Tensor,
        slot_3: torch.Tensor,
        slot_4: torch.Tensor,
        layout: str = "ideogram_3x3",
        panel_mapping: str = "0,1,2,4,5",
        cell_size: int = 333,
        background_color: int = 0,
    ) -> tuple[torch.Tensor, str]:
        """Assemble individual panels into a contact sheet grid."""
        logger.info("MSR Contact Sheet Assembler started (layout=%s)", layout)

        try:
            preset = LAYOUT_PRESETS.get(layout, LAYOUT_PRESETS["ideogram_3x3"])
            grid_cols, grid_rows = preset["grid"]
            mapping = parse_panel_mapping(
                panel_mapping, len(PANEL_OUTPUT_NAMES), grid_cols, grid_rows
            )

            panels = {
                "background": background,
                "slot1": slot_1,
                "slot2": slot_2,
                "slot3": slot_3,
                "slot4": slot_4,
            }

            batch_size = max(p.shape[0] for p in panels.values())

            canvas_w = grid_cols * cell_size
            canvas_h = grid_rows * cell_size
            bg_val = background_color / 255.0

            result = torch.full(
                (batch_size, canvas_h, canvas_w, 3),
                bg_val,
                dtype=torch.float32,
            )

            for name, cell_idx in zip(PANEL_OUTPUT_NAMES, mapping):
                col = cell_idx % grid_cols
                row = cell_idx // grid_cols

                left = col * cell_size
                top = row * cell_size

                panel = panels[name]
                if panel.shape[0] < batch_size:
                    repeats = batch_size // panel.shape[0] + 1
                    panel = panel.repeat(repeats, 1, 1, 1)[:batch_size]

                for b in range(batch_size):
                    pil_panel = tensor_to_pil(panel[b : b + 1])
                    pil_resized = pil_panel.resize((cell_size, cell_size), Image.LANCZOS)
                    resized_tensor = pil_to_tensor(pil_resized)

                    h_fit = min(resized_tensor.shape[1], cell_size)
                    w_fit = min(resized_tensor.shape[2], cell_size)
                    result[b, top : top + h_fit, left : left + w_fit, :] = resized_tensor[
                        0, :h_fit, :w_fit, :
                    ]

            status = (
                f"OK | assembled {grid_cols}x{grid_rows} grid "
                f"({canvas_w}x{canvas_h}) | batch={batch_size}"
            )
            logger.info(status)
            return result, status

        except (ValueError, RuntimeError, OSError) as exc:
            logger.exception("Failed to assemble contact sheet: %s", exc)
            empty = torch.zeros((1, cell_size * 3, cell_size * 3, 3), dtype=torch.float32)
            return empty, f"ERROR: {exc}"
