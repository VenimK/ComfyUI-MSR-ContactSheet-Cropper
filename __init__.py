"""ComfyUI custom nodes for cropping and assembling MSR Contact Sheets.

Provides two nodes:
    - MSRContactSheetCropper: Crops a contact sheet into individual panels.
    - MSRContactSheetAssembler: Composites panels back into a contact sheet.

Features:
    - Auto-detect grid lines from image content
    - Layout presets (Ideogram 3x3, Midjourney 4x4, Custom 3x3)
    - Batch processing support
    - Visual debug overlay output
    - Panel-to-slot remapping
    - Smart file organization with timestamps
    - Mask outputs for inpainting workflows
    - GPU-side tensor cropping
    - Live status reporting
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)

# --- Constants ---------------------------------------------------------------

DEFAULT_GRID_SIZE = 1000

# Layout presets: panel_name -> (col_start, row_start, col_end, row_end) in cell units
LAYOUT_PRESETS: dict[str, dict[str, Any]] = {
    "ideogram_3x3": {
        "grid": (3, 3),
        "panels": {
            "background": (0, 0, 1, 1),
            "slot1": (1, 0, 2, 1),
            "slot2": (2, 0, 3, 1),
            "slot3": (1, 1, 2, 2),
            "slot4": (2, 1, 3, 2),
        },
        "backup": {
            "backup_A": (0, 1, 1, 2),
            "backup_B": (0, 2, 1, 3),
            "backup_bg": (1, 2, 2, 3),
            "backup_wide": (2, 2, 3, 3),
        },
    },
    "midjourney_4x4": {
        "grid": (4, 4),
        "panels": {
            "background": (0, 0, 1, 1),
            "slot1": (1, 0, 2, 1),
            "slot2": (2, 0, 3, 1),
            "slot3": (3, 0, 4, 1),
            "slot4": (0, 1, 1, 2),
        },
        "backup": {},
    },
    "custom_3x3": {
        "grid": (3, 3),
        "panels": {
            "background": (0, 0, 1, 1),
            "slot1": (1, 0, 2, 1),
            "slot2": (2, 0, 3, 1),
            "slot3": (0, 1, 1, 2),
            "slot4": (1, 1, 2, 2),
        },
        "backup": {
            "backup_A": (2, 1, 3, 2),
            "backup_B": (0, 2, 1, 3),
            "backup_bg": (1, 2, 2, 3),
            "backup_wide": (2, 2, 3, 3),
        },
    },
}

PANEL_OUTPUT_NAMES = ("background", "slot1", "slot2", "slot3", "slot4")
OVERLAY_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255),
    (255, 255, 0), (255, 0, 255), (0, 255, 255),
    (255, 128, 0), (128, 0, 255), (255, 255, 255),
]


# --- Helper functions --------------------------------------------------------


def _detect_grid_lines(
    img_np: np.ndarray, expected_cells: int = 3,
) -> tuple[list[int], list[int]] | None:
    """Detect internal grid lines from image content.

    Uses row/column variance profiles to find cell boundaries.

    Args:
        img_np: Image as numpy array (H, W, C) in uint8.
        expected_cells: Expected number of cells per axis (e.g. 3 for 3x3).

    Returns:
        Tuple of (horizontal_line_positions, vertical_line_positions) in pixels,
        or None if detection fails.
    """
    gray = img_np.mean(axis=2) if img_np.ndim == 3 else img_np

    row_diffs = np.abs(np.diff(gray, axis=0)).mean(axis=1)
    col_diffs = np.abs(np.diff(gray, axis=1)).mean(axis=0)

    h_lines = _find_grid_peaks(row_diffs, expected_cells - 1, gray.shape[0])
    v_lines = _find_grid_peaks(col_diffs, expected_cells - 1, gray.shape[1])

    if h_lines is None or v_lines is None:
        return None

    return h_lines, v_lines


def _find_grid_peaks(
    profile: np.ndarray, n_expected: int, total_size: int,
) -> list[int] | None:
    """Find the N strongest evenly-spaced peaks in a 1D profile."""
    if len(profile) == 0 or n_expected <= 0:
        return []

    threshold = float(profile.mean()) + float(profile.std()) * 1.5
    peaks = np.where(profile > threshold)[0]

    if len(peaks) == 0:
        return None

    gap = max(total_size // 50, 3)
    groups: list[list[int]] = [[int(peaks[0])]]
    for p in peaks[1:]:
        if p - groups[-1][-1] <= gap:
            groups[-1].append(int(p))
        else:
            groups.append([int(p)])

    centers = [int(np.mean(g)) for g in groups]

    if len(centers) > n_expected:
        strengths = [float(profile[c]) for c in centers]
        top_idx = np.argsort(strengths)[-n_expected:]
        centers = sorted([centers[i] for i in top_idx])

    if len(centers) < n_expected:
        logger.debug(
            "Grid detection: found %d/%d expected lines", len(centers), n_expected
        )
        return None

    return centers


def _cell_to_pixel(
    cell: tuple[int, int, int, int],
    width: int, height: int,
    grid_cols: int, grid_rows: int,
) -> tuple[int, int, int, int]:
    """Convert cell coordinates to pixel coordinates."""
    left = int(cell[0] * width / grid_cols)
    top = int(cell[1] * height / grid_rows)
    right = int(cell[2] * width / grid_cols)
    bottom = int(cell[3] * height / grid_rows)
    return left, top, right, bottom


def _crop_tensor(
    tensor: torch.Tensor, top: int, left: int, bottom: int, right: int,
) -> torch.Tensor:
    """Crop a tensor on GPU/CPU: (B, H, W, C) -> (B, H', W', C)."""
    return tensor[:, top:bottom, left:right, :].contiguous()


def _tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert a single image tensor (1, H, W, C) to PIL Image."""
    img_np = (tensor[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(img_np).convert("RGB")


def _pil_to_tensor(pil_img: Image.Image) -> torch.Tensor:
    """Convert a PIL Image to a tensor (1, H, W, C) in [0, 1]."""
    arr = np.array(pil_img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def _make_mask(height: int, width: int, batch: int = 1) -> torch.Tensor:
    """Create a white mask tensor (B, H, W) filled with 1.0."""
    return torch.ones((batch, height, width), dtype=torch.float32)


def _draw_debug_overlay(
    pil_img: Image.Image,
    panel_coords: dict[str, tuple[int, int, int, int]],
) -> Image.Image:
    """Draw labeled rectangles on the image for visual debugging."""
    overlay = pil_img.copy()
    draw = ImageDraw.Draw(overlay)

    for i, (name, (left, top, right, bottom)) in enumerate(panel_coords.items()):
        color = OVERLAY_COLORS[i % len(OVERLAY_COLORS)]
        draw.rectangle([left, top, right - 1, bottom - 1], outline=color, width=3)
        bbox = draw.textbbox((left + 4, top + 4), name)
        draw.rectangle(
            [bbox[0] - 2, bbox[1] - 2, bbox[2] + 2, bbox[3] + 2], fill=color
        )
        draw.text((left + 4, top + 4), name, fill=(0, 0, 0))

    return overlay


def _parse_panel_mapping(mapping_str: str, n_slots: int = 5) -> list[int]:
    """Parse a panel mapping string like '0,1,2,5,6' into cell indices.

    Each index refers to a cell in row-major order (0=top-left, 1=top-center, ...).
    """
    try:
        indices = [int(x.strip()) for x in mapping_str.split(",")]
    except ValueError:
        logger.warning("Invalid panel mapping '%s', using default", mapping_str)
        return [0, 1, 2, 4, 5]

    if len(indices) != n_slots:
        logger.warning(
            "Panel mapping has %d indices, expected %d; using default",
            len(indices), n_slots,
        )
        return [0, 1, 2, 4, 5]

    return indices


def _apply_panel_mapping(
    mapping: list[int], grid_cols: int, grid_rows: int,
) -> dict[str, tuple[int, int, int, int]]:
    """Build cell coordinates for each output panel based on mapping."""
    result: dict[str, tuple[int, int, int, int]] = {}
    for name, cell_idx in zip(PANEL_OUTPUT_NAMES, mapping):
        col = cell_idx % grid_cols
        row = cell_idx // grid_cols
        result[name] = (col, row, col + 1, row + 1)
    return result


class MSRContactSheetCropper:
    """ComfyUI node that crops a contact sheet into Licon-MSR panels.

    Supports auto-detection, layout presets, batch processing, mask outputs,
    debug overlay, panel remapping, and GPU-side cropping.
    """

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "contact_sheet": ("IMAGE",),
            },
            "optional": {
                "layout": (
                    ["ideogram_3x3", "midjourney_4x4", "custom_3x3"],
                    {"default": "ideogram_3x3"},
                ),
                "detect_grid": ("BOOLEAN", {"default": False}),
                "grid_size": ("INT", {"default": DEFAULT_GRID_SIZE, "min": 1, "max": 10000, "step": 1}),
                "panel_mapping": ("STRING", {"default": "0,1,2,4,5", "multiline": False}),
                "filename_prefix": ("STRING", {"default": "msr"}),
                "save_to_disk": ("BOOLEAN", {"default": True}),
                "save_backup_panels": ("BOOLEAN", {"default": False}),
                "include_timestamp": ("BOOLEAN", {"default": False}),
                "output_folder": ("STRING", {"default": "msr_crops"}),
            },
        }

    RETURN_TYPES = (
        "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE",
        "MASK", "MASK", "MASK", "MASK", "MASK",
        "IMAGE", "STRING",
    )
    RETURN_NAMES = (
        "background", "slot1", "slot2", "slot3", "slot4",
        "mask_bg", "mask_slot1", "mask_slot2", "mask_slot3", "mask_slot4",
        "debug_overlay", "status",
    )
    FUNCTION = "crop_panels"
    CATEGORY = "Licon-MSR / Utils"

    # --- Public API ----------------------------------------------------------

    def crop_panels(
        self,
        contact_sheet: torch.Tensor,
        layout: str = "ideogram_3x3",
        detect_grid: bool = False,
        grid_size: int = DEFAULT_GRID_SIZE,
        panel_mapping: str = "0,1,2,4,5",
        filename_prefix: str = "msr",
        save_to_disk: bool = True,
        save_backup_panels: bool = False,
        include_timestamp: bool = False,
        output_folder: str = "msr_crops",
    ) -> tuple[torch.Tensor, ...]:
        """Crop the contact sheet into panels with full feature support.

        Args:
            contact_sheet: Input image tensor (B, H, W, C) in [0, 1].
            layout: Layout preset name.
            detect_grid: Auto-detect grid lines from image content.
            grid_size: Reference grid dimension (used when detect_grid is False).
            panel_mapping: Comma-separated cell indices for panel assignment.
            filename_prefix: Prefix for saved filenames.
            save_to_disk: Whether to write crops to disk.
            save_backup_panels: Whether to also process backup panels.
            include_timestamp: Add timestamp subfolder to output path.
            output_folder: Output directory for saved files.

        Returns:
            Tuple of (5 IMAGE, 5 MASK, debug overlay IMAGE, status STRING).
        """
        logger.info(
            "MSR Contact Sheet Cropper started (layout=%s, batch=%d)",
            layout, contact_sheet.shape[0],
        )
        logger.debug("Input tensor shape: %s", contact_sheet.shape)

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

            # Auto-detect grid lines if requested
            detected_lines = None
            if detect_grid:
                first_pil = _tensor_to_pil(contact_sheet[:1])
                img_np = np.array(first_pil)
                detected_lines = _detect_grid_lines(img_np, grid_cols)
                if detected_lines is not None:
                    h_lines, v_lines = detected_lines
                    logger.info(
                        "Grid detected: H lines %s, V lines %s", h_lines, v_lines
                    )
                else:
                    logger.warning(
                        "Grid detection failed, falling back to grid_size=%d",
                        grid_size,
                    )

            # Parse panel mapping
            mapping = _parse_panel_mapping(panel_mapping, len(PANEL_OUTPUT_NAMES))

            # Build panel pixel coordinates
            if detect_grid and detected_lines is not None:
                panel_coords = self._coords_from_detected(
                    width, height, h_lines, v_lines, mapping, grid_cols, grid_rows
                )
                backup_coords = self._coords_from_detected_backup(
                    width, height, h_lines, v_lines, preset, grid_cols, grid_rows
                )
            else:
                cell_coords = _apply_panel_mapping(mapping, grid_cols, grid_rows)
                panel_coords = {
                    name: _cell_to_pixel(cell, width, height, grid_cols, grid_rows)
                    for name, cell in cell_coords.items()
                }
                backup_coords = {
                    name: _cell_to_pixel(cell, width, height, grid_cols, grid_rows)
                    for name, cell in preset.get("backup", {}).items()
                }

            # Aspect ratio warning
            aspect = width / float(height)
            if not 0.9 <= aspect <= 1.1:
                logger.warning(
                    "Input aspect ratio %.2f deviates from square; "
                    "panels may not align correctly",
                    aspect,
                )

            # Prepare output folder with optional timestamp
            save_dir = output_folder
            if include_timestamp and save_to_disk:
                ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                save_dir = os.path.join(output_folder, ts)

            # Process each image in batch
            all_panel_tensors: list[list[torch.Tensor]] = [
                [] for _ in PANEL_OUTPUT_NAMES
            ]
            all_mask_tensors: list[list[torch.Tensor]] = [
                [] for _ in PANEL_OUTPUT_NAMES
            ]
            overlay_tensors: list[torch.Tensor] = []

            for batch_idx in range(batch_size):
                img_tensor = contact_sheet[batch_idx : batch_idx + 1]

                # Debug overlay (from first image or every image if batch == 1)
                if batch_idx == 0 or batch_size == 1:
                    pil_for_overlay = _tensor_to_pil(img_tensor)
                    overlay_pil = _draw_debug_overlay(pil_for_overlay, panel_coords)
                    overlay_tensors.append(_pil_to_tensor(overlay_pil))

                # GPU-side crop each panel
                for panel_idx, name in enumerate(PANEL_OUTPUT_NAMES):
                    left, top, right, bottom = panel_coords[name]
                    crop = _crop_tensor(img_tensor, top, left, bottom, right)
                    all_panel_tensors[panel_idx].append(crop)

                    mask = _make_mask(bottom - top, right - left, 1)
                    all_mask_tensors[panel_idx].append(mask)

                    if save_to_disk:
                        pil_crop = _tensor_to_pil(crop)
                        self._save_panel(
                            pil_crop, save_dir, filename_prefix,
                            name, batch_idx, batch_size,
                        )

                # Save backup panels
                if save_backup_panels:
                    for name, (left, top, right, bottom) in backup_coords.items():
                        crop = _crop_tensor(img_tensor, top, left, bottom, right)
                        if save_to_disk:
                            pil_crop = _tensor_to_pil(crop)
                            self._save_panel(
                                pil_crop, save_dir, filename_prefix,
                                name, batch_idx, batch_size,
                            )

            # Stack batch results
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

            # Build status string
            status_parts = [
                f"Layout: {layout}",
                f"Grid: {grid_cols}x{grid_rows}",
                f"Image: {width}x{height}",
                f"Batch: {batch_size}",
                f"Mapping: {panel_mapping}",
            ]
            if detect_grid:
                status_parts.append(
                    f"Auto-detect: {'success' if detected_lines else 'failed (fallback)'}"
                )
            if save_to_disk:
                status_parts.append(f"Saved to: {save_dir}")
            status = " | ".join(status_parts)

            logger.info("All panels processed: %s", status)
            return tuple(panel_outputs + mask_outputs + [overlay_output, status])

        except (ValueError, RuntimeError, OSError) as exc:
            logger.exception("Failed to crop contact sheet: %s", exc)
            panel_size = max(grid_size // 3, 1)
            empty_img = torch.zeros((1, panel_size, panel_size, 3), dtype=torch.float32)
            empty_mask = torch.zeros((1, panel_size, panel_size), dtype=torch.float32)
            empty_overlay = torch.zeros(
                (1, panel_size * 3, panel_size * 3, 3), dtype=torch.float32
            )
            error_status = f"ERROR: {exc}"
            return (
                empty_img, empty_img, empty_img, empty_img, empty_img,
                empty_mask, empty_mask, empty_mask, empty_mask, empty_mask,
                empty_overlay, error_status,
            )

    # --- Private helpers -----------------------------------------------------

    @staticmethod
    def _coords_from_detected(
        width: int, height: int,
        h_lines: list[int], v_lines: list[int],
        mapping: list[int],
        grid_cols: int, grid_rows: int,
    ) -> dict[str, tuple[int, int, int, int]]:
        """Build pixel coordinates from detected grid lines + panel mapping."""
        all_h = [0] + sorted(h_lines) + [height]
        all_v = [0] + sorted(v_lines) + [width]

        result: dict[str, tuple[int, int, int, int]] = {}
        for name, cell_idx in zip(PANEL_OUTPUT_NAMES, mapping):
            col = cell_idx % grid_cols
            row = cell_idx // grid_cols
            result[name] = (all_v[col], all_h[row], all_v[col + 1], all_h[row + 1])

        return result

    @staticmethod
    def _coords_from_detected_backup(
        width: int, height: int,
        h_lines: list[int], v_lines: list[int],
        preset: dict[str, Any],
        grid_cols: int, grid_rows: int,
    ) -> dict[str, tuple[int, int, int, int]]:
        """Build backup panel pixel coordinates from detected grid lines."""
        all_h = [0] + sorted(h_lines) + [height]
        all_v = [0] + sorted(v_lines) + [width]

        result: dict[str, tuple[int, int, int, int]] = {}
        for name, cell in preset.get("backup", {}).items():
            col_start, row_start, col_end, row_end = cell
            result[name] = (
                all_v[col_start], all_h[row_start],
                all_v[col_end], all_h[row_end],
            )

        return result

    @staticmethod
    def _save_panel(
        pil_img: Image.Image,
        output_folder: str,
        filename_prefix: str,
        name: str,
        batch_idx: int = 0,
        batch_size: int = 1,
    ) -> None:
        """Save a PIL image to disk."""
        os.makedirs(output_folder, exist_ok=True)
        if batch_size > 1:
            filename = f"{filename_prefix}_{name}_{batch_idx:04d}.png"
        else:
            filename = f"{filename_prefix}_{name}.png"
        save_path = os.path.join(output_folder, filename)
        pil_img.save(save_path)
        logger.debug("Saved: %s", save_path)


class MSRContactSheetAssembler:
    """ComfyUI node that composites panels back into a contact sheet.

    Reverse operation of MSRContactSheetCropper: takes individual panel images
    and arranges them into a grid based on the selected layout.
    """

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "background": ("IMAGE",),
                "slot1": ("IMAGE",),
                "slot2": ("IMAGE",),
                "slot3": ("IMAGE",),
                "slot4": ("IMAGE",),
            },
            "optional": {
                "layout": (
                    ["ideogram_3x3", "midjourney_4x4", "custom_3x3"],
                    {"default": "ideogram_3x3"},
                ),
                "panel_mapping": ("STRING", {"default": "0,1,2,4,5", "multiline": False}),
                "cell_size": ("INT", {"default": 333, "min": 1, "max": 4096, "step": 1}),
                "background_color": ("INT", {"default": 0, "min": 0, "max": 255, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("contact_sheet", "status")
    FUNCTION = "assemble"
    CATEGORY = "Licon-MSR / Utils"

    def assemble(
        self,
        background: torch.Tensor,
        slot1: torch.Tensor,
        slot2: torch.Tensor,
        slot3: torch.Tensor,
        slot4: torch.Tensor,
        layout: str = "ideogram_3x3",
        panel_mapping: str = "0,1,2,4,5",
        cell_size: int = 333,
        background_color: int = 0,
    ) -> tuple[torch.Tensor, str]:
        """Assemble individual panels into a contact sheet grid.

        Args:
            background through slot4: Panel image tensors (B, H, W, C).
            layout: Layout preset name.
            panel_mapping: Cell indices for panel placement.
            cell_size: Target size for each cell in the grid.
            background_color: Fill color for empty cells (0-255).

        Returns:
            Tuple of (contact sheet IMAGE tensor, status STRING).
        """
        logger.info("MSR Contact Sheet Assembler started (layout=%s)", layout)

        try:
            preset = LAYOUT_PRESETS.get(layout, LAYOUT_PRESETS["ideogram_3x3"])
            grid_cols, grid_rows = preset["grid"]
            mapping = _parse_panel_mapping(panel_mapping, len(PANEL_OUTPUT_NAMES))

            panels = {
                "background": background,
                "slot1": slot1,
                "slot2": slot2,
                "slot3": slot3,
                "slot4": slot4,
            }

            batch_size = max(p.shape[0] for p in panels.values())

            canvas_w = grid_cols * cell_size
            canvas_h = grid_rows * cell_size
            bg_val = background_color / 255.0

            result = torch.full(
                (batch_size, canvas_h, canvas_w, 3),
                bg_val, dtype=torch.float32,
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
                    pil_panel = _tensor_to_pil(panel[b : b + 1])
                    pil_resized = pil_panel.resize(
                        (cell_size, cell_size), Image.LANCZOS
                    )
                    resized_tensor = _pil_to_tensor(pil_resized)

                    h_fit = min(resized_tensor.shape[1], cell_size)
                    w_fit = min(resized_tensor.shape[2], cell_size)
                    result[
                        b, top : top + h_fit, left : left + w_fit, :
                    ] = resized_tensor[0, :h_fit, :w_fit, :]

            status = (
                f"Assembled {grid_cols}x{grid_rows} grid "
                f"({canvas_w}x{canvas_h}) | Batch: {batch_size}"
            )
            logger.info(status)
            return result, status

        except (ValueError, RuntimeError, OSError) as exc:
            logger.exception("Failed to assemble contact sheet: %s", exc)
            empty = torch.zeros(
                (1, cell_size * 3, cell_size * 3, 3), dtype=torch.float32
            )
            return empty, f"ERROR: {exc}"


# --- Node registration -------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "MSRContactSheetCropper": MSRContactSheetCropper,
    "MSRContactSheetAssembler": MSRContactSheetAssembler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MSRContactSheetCropper": "MSR Contact Sheet Cropper",
    "MSRContactSheetAssembler": "MSR Contact Sheet Assembler",
}

logger.info(
    "Licon-MSR Utils: MSRContactSheetCropper + MSRContactSheetAssembler loaded successfully"
)