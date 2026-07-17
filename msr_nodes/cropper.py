"""MSR Contact Sheet Cropper ComfyUI node."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any

import torch
from PIL import Image

from msr_nodes.constants import (
    DEFAULT_GRID_SIZE,
    LAYOUT_PRESETS,
    PANEL_OUTPUT_NAMES,
)
from msr_nodes.grid import parse_panel_mapping, resolve_panel_coords
from msr_nodes.image_utils import (
    crop_tensor,
    draw_debug_overlay,
    make_mask,
    pil_to_tensor,
    tensor_to_pil,
)
from msr_nodes.types import ComfyInputSpec

logger = logging.getLogger(__name__)


class MSRContactSheetCropper:
    """ComfyUI node that crops a contact sheet into panels."""

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "contact_sheet": ("IMAGE",),
            },
            "optional": {
                "layout": (
                    ["ideogram_3x3", "midjourney_4x4", "custom_3x3"],
                    ComfyInputSpec({"default": "ideogram_3x3"}),
                ),
                "detect_grid": ("BOOLEAN", ComfyInputSpec({"default": True})),
                "panel_mapping": (
                    "STRING",
                    ComfyInputSpec({"default": "0,1,2,4,5", "multiline": False}),
                ),
                "grid_size": (
                    "INT",
                    ComfyInputSpec(
                        {
                            "default": DEFAULT_GRID_SIZE,
                            "min": 1,
                            "max": 10000,
                            "step": 1,
                        }
                    ),
                ),
                "filename_prefix": ("STRING", ComfyInputSpec({"default": "msr"})),
                "output_folder": ("STRING", ComfyInputSpec({"default": "msr_crops"})),
                "subfolder_by_layout": ("BOOLEAN", ComfyInputSpec({"default": False})),
                "include_timestamp": ("BOOLEAN", ComfyInputSpec({"default": False})),
                "save_to_disk": ("BOOLEAN", ComfyInputSpec({"default": True})),
                "save_backup_panels": ("BOOLEAN", ComfyInputSpec({"default": False})),
                "save_debug_overlay": ("BOOLEAN", ComfyInputSpec({"default": False})),
            },
        }

    RETURN_TYPES = (
        "IMAGE",
        "IMAGE",
        "IMAGE",
        "IMAGE",
        "IMAGE",
        "MASK",
        "MASK",
        "MASK",
        "MASK",
        "MASK",
        "IMAGE",
        "STRING",
    )
    RETURN_NAMES = (
        "background",
        "slot_1",
        "slot_2",
        "slot_3",
        "slot_4",
        "mask_bg",
        "mask_slot_1",
        "mask_slot_2",
        "mask_slot_3",
        "mask_slot_4",
        "debug_overlay",
        "status",
    )
    FUNCTION = "crop_panels"
    CATEGORY = "Licon-MSR / Crop"

    def crop_panels(
        self,
        contact_sheet: torch.Tensor,
        layout: str = "ideogram_3x3",
        detect_grid: bool = True,
        panel_mapping: str = "0,1,2,4,5",
        grid_size: int = DEFAULT_GRID_SIZE,
        filename_prefix: str = "msr",
        output_folder: str = "msr_crops",
        subfolder_by_layout: bool = False,
        include_timestamp: bool = False,
        save_to_disk: bool = True,
        save_backup_panels: bool = False,
        save_debug_overlay: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        """Crop a contact sheet tensor into panels, masks, overlay, and status."""
        logger.info("MSR Contact Sheet Cropper started (layout=%s)", layout)

        try:
            if contact_sheet.ndim != 4:
                raise ValueError(
                    f"Expected 4D tensor (B, H, W, C), got shape {contact_sheet.shape}"
                )

            batch_size, height, width, _ = contact_sheet.shape
            if width <= 0 or height <= 0:
                raise ValueError(f"Invalid image dimensions: {width}x{height}")

            preset = LAYOUT_PRESETS.get(layout, LAYOUT_PRESETS["ideogram_3x3"])
            grid_cols, grid_rows = preset["grid"]

            mapping = parse_panel_mapping(
                panel_mapping, len(PANEL_OUTPUT_NAMES), grid_cols, grid_rows
            )

            first_pil = tensor_to_pil(contact_sheet[:1])
            panel_coords, backup_coords, grid_detect_status = resolve_panel_coords(
                width,
                height,
                layout,
                detect_grid,
                mapping,
                grid_size,
                first_pil,
            )

            # Aspect ratio warning
            aspect = width / float(height)
            warnings: list[str] = []
            if not 0.9 <= aspect <= 1.1:
                logger.warning(
                    "Input aspect ratio %.2f deviates from square; "
                    "panels may not align correctly",
                    aspect,
                )
                warnings.append("non-square input")

            # Prepare output folder with optional layout subfolder and timestamp
            save_dir = output_folder
            if subfolder_by_layout:
                save_dir = os.path.join(save_dir, layout)
            if include_timestamp and save_to_disk:
                ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                save_dir = os.path.join(save_dir, ts)

            # Process each image in batch
            all_panel_tensors: list[list[torch.Tensor]] = [[] for _ in PANEL_OUTPUT_NAMES]
            all_mask_tensors: list[list[torch.Tensor]] = [[] for _ in PANEL_OUTPUT_NAMES]
            overlay_tensors: list[torch.Tensor] = []

            for batch_idx in range(batch_size):
                img_tensor = contact_sheet[batch_idx : batch_idx + 1]

                # Debug overlay (from first image or every image if batch == 1)
                if batch_idx == 0 or batch_size == 1:
                    pil_for_overlay = tensor_to_pil(img_tensor)
                    overlay_pil = draw_debug_overlay(
                        pil_for_overlay, panel_coords, grid_cols, grid_rows
                    )
                    overlay_tensors.append(pil_to_tensor(overlay_pil))

                    if save_to_disk and save_debug_overlay:
                        self._save_panel(
                            overlay_pil,
                            save_dir,
                            filename_prefix,
                            "debug_overlay",
                            batch_idx,
                            batch_size,
                        )

                # GPU-side crop each panel
                for panel_idx, name in enumerate(PANEL_OUTPUT_NAMES):
                    coords = panel_coords[name]
                    crop = crop_tensor(
                        img_tensor,
                        coords.top,
                        coords.left,
                        coords.bottom,
                        coords.right,
                    )
                    all_panel_tensors[panel_idx].append(crop)

                    mask = make_mask(
                        coords.bottom - coords.top,
                        coords.right - coords.left,
                        crop.shape[0],
                    )
                    all_mask_tensors[panel_idx].append(mask)

                    if save_to_disk:
                        self._save_panel(
                            tensor_to_pil(crop),
                            save_dir,
                            filename_prefix,
                            name,
                            batch_idx,
                            batch_size,
                        )

                # Optional backup panels
                if save_backup_panels and save_to_disk:
                    for name, coords in backup_coords.items():
                        crop = crop_tensor(
                            img_tensor,
                            coords.top,
                            coords.left,
                            coords.bottom,
                            coords.right,
                        )
                        self._save_panel(
                            tensor_to_pil(crop),
                            save_dir,
                            filename_prefix,
                            name,
                            batch_idx,
                            batch_size,
                        )

            # Stack batch outputs
            panel_outputs = [
                torch.cat(tensors, dim=0) if batch_size > 1 else tensors[0]
                for tensors in all_panel_tensors
            ]
            mask_outputs = [
                torch.cat(tensors, dim=0) if batch_size > 1 else tensors[0]
                for tensors in all_mask_tensors
            ]

            overlay_output = (
                overlay_tensors[0]
                if overlay_tensors
                else torch.zeros((1, height, width, 3), dtype=torch.float32)
            )

            status = self._build_status(
                layout=layout,
                grid_cols=grid_cols,
                grid_rows=grid_rows,
                width=width,
                height=height,
                batch_size=batch_size,
                panel_mapping=panel_mapping,
                grid_detect_status=grid_detect_status,
                save_dir=save_dir if save_to_disk else None,
                warnings=warnings,
            )

            if save_to_disk:
                self._write_manifest(
                    save_dir,
                    layout=layout,
                    grid_cols=grid_cols,
                    grid_rows=grid_rows,
                    width=width,
                    height=height,
                    panel_mapping=mapping,
                    grid_detect_status=grid_detect_status,
                    saved_count=len(PANEL_OUTPUT_NAMES) * batch_size
                    + (len(backup_coords) * batch_size if save_backup_panels else 0),
                )

            logger.info("All panels processed: %s", status)
            return tuple(panel_outputs + mask_outputs + [overlay_output, status])

        except (ValueError, RuntimeError, OSError) as exc:
            logger.exception("Failed to crop contact sheet: %s", exc)
            panel_size = max(grid_size // 3, 1)
            empty_img = torch.zeros((1, panel_size, panel_size, 3), dtype=torch.float32)
            empty_mask = torch.zeros((1, panel_size, panel_size), dtype=torch.float32)
            empty_overlay = torch.zeros(
                (1, max(panel_size, 64), max(panel_size, 64), 3), dtype=torch.float32
            )
            error_status = f"ERROR: {exc}"
            return (
                empty_img,
                empty_img,
                empty_img,
                empty_img,
                empty_img,
                empty_mask,
                empty_mask,
                empty_mask,
                empty_mask,
                empty_mask,
                empty_overlay,
                error_status,
            )

    @staticmethod
    def _save_panel(
        pil_img: Image.Image,
        output_folder: str,
        filename_prefix: str,
        name: str,
        batch_idx: int,
        batch_size: int,
    ) -> str:
        """Save one panel to disk and return the path."""
        os.makedirs(output_folder, exist_ok=True)
        suffix = f"_{batch_idx:04d}" if batch_size > 1 else ""
        filename = f"{filename_prefix}_{name}{suffix}.png"
        save_path = os.path.join(output_folder, filename)
        pil_img.save(save_path)
        return save_path

    @staticmethod
    def _build_status(
        layout: str,
        grid_cols: int,
        grid_rows: int,
        width: int,
        height: int,
        batch_size: int,
        panel_mapping: str,
        grid_detect_status: str,
        save_dir: str | None,
        warnings: list[str],
    ) -> str:
        """Build a concise, human-readable status string."""
        parts = [
            f"OK | layout={layout}",
            f"grid={grid_cols}x{grid_rows}",
            f"image={width}x{height}",
            f"batch={batch_size}",
            f"mapping={panel_mapping}",
            f"detect={grid_detect_status}",
        ]
        if save_dir:
            parts.append(f"saved={save_dir}")
        if warnings:
            parts.append(f"warnings={', '.join(warnings)}")
        return " | ".join(parts)

    @staticmethod
    def _write_manifest(
        output_folder: str,
        layout: str,
        grid_cols: int,
        grid_rows: int,
        width: int,
        height: int,
        panel_mapping: list[int],
        grid_detect_status: str,
        saved_count: int,
    ) -> None:
        """Write a JSON manifest describing the crop run."""
        os.makedirs(output_folder, exist_ok=True)
        manifest = {
            "created_at": datetime.now().isoformat(),
            "layout": layout,
            "grid": {"cols": grid_cols, "rows": grid_rows},
            "image": {"width": width, "height": height},
            "panel_mapping": panel_mapping,
            "grid_detect_status": grid_detect_status,
            "saved_files": saved_count,
        }
        path = os.path.join(output_folder, "manifest.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
