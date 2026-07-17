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

import json
import logging
import os
from datetime import datetime
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

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
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (255, 128, 0),
    (128, 0, 255),
    (255, 255, 255),
]


# --- Helper functions --------------------------------------------------------


def _detect_grid_lines(
    img_np: np.ndarray,
    expected_cells: int = 3,
) -> tuple[list[int], list[int]] | None:
    """Detect internal grid lines from image content using RGB differences.

    Uses row/column color-difference profiles to find cell boundaries.
    This is more robust than grayscale-only profiles for contact sheets
    where neighboring panels may have similar brightness.

    Args:
        img_np: Image as numpy array (H, W, C) in uint8.
        expected_cells: Expected number of cells per axis (e.g. 3 for 3x3).

    Returns:
        Tuple of (horizontal_line_positions, vertical_line_positions) in pixels,
        or None if detection fails.
    """
    if img_np.ndim == 2:
        img_np = np.stack([img_np, img_np, img_np], axis=2)

    # Mean absolute RGB difference across rows/columns
    row_diffs = np.abs(np.diff(img_np, axis=0)).mean(axis=(1, 2))
    col_diffs = np.abs(np.diff(img_np, axis=1)).mean(axis=(1, 2))

    h_lines = _find_grid_peaks(row_diffs, expected_cells - 1, img_np.shape[0])
    v_lines = _find_grid_peaks(col_diffs, expected_cells - 1, img_np.shape[1])

    if h_lines is None or v_lines is None:
        return None

    return h_lines, v_lines


def _find_grid_peaks(
    profile: np.ndarray,
    n_expected: int,
    total_size: int,
) -> list[int] | None:
    """Find the N expected evenly-spaced grid lines in a difference profile.

    Uses a simple comb-matching approach: the best set of internal grid
    lines is the one that maximizes the total profile response at positions
    that are roughly evenly spaced across the image.
    """
    if len(profile) == 0 or n_expected <= 0:
        return []

    if n_expected == 1:
        # Single line: take the strongest peak near the middle half
        start = total_size // 4
        end = 3 * total_size // 4
        return [int(start + np.argmax(profile[start:end]))]

    # Expected spacing between adjacent grid lines
    spacing = total_size / (n_expected + 1)

    # Smooth profile slightly to reduce noise
    window = max(total_size // 200, 3)
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window) / window
    smoothed = np.convolve(profile, kernel, mode="same")

    best_score = -np.inf
    best_offset = 0.0
    # Search offsets within one spacing step
    search_steps = max(int(spacing), 10)
    for step in range(search_steps):
        offset = step * spacing / search_steps
        positions = [int(round(offset + k * spacing)) for k in range(1, n_expected + 1)]
        positions = [max(0, min(p, len(smoothed) - 1)) for p in positions]
        score = sum(smoothed[p] for p in positions)
        if score > best_score:
            best_score = score
            best_offset = offset

    lines = sorted(int(round(best_offset + k * spacing)) for k in range(1, n_expected + 1))

    # Refine each line by looking for a local maximum in a small neighborhood
    refined: list[int] = []
    half_window = max(total_size // 100, 2)
    for line in lines:
        start = max(0, line - half_window)
        end = min(len(smoothed), line + half_window + 1)
        local_peak = start + int(np.argmax(smoothed[start:end]))
        refined.append(local_peak)

    # Validate that we got enough distinct lines
    if len(set(refined)) < n_expected:
        logger.debug("Grid detection: only %d distinct lines found", len(set(refined)))
        return None

    return refined


def _cell_to_pixel(
    cell: tuple[int, int, int, int],
    width: int,
    height: int,
    grid_cols: int,
    grid_rows: int,
) -> tuple[int, int, int, int]:
    """Convert cell coordinates to pixel coordinates."""
    left = int(cell[0] * width / grid_cols)
    top = int(cell[1] * height / grid_rows)
    right = int(cell[2] * width / grid_cols)
    bottom = int(cell[3] * height / grid_rows)
    return left, top, right, bottom


def _crop_tensor(
    tensor: torch.Tensor,
    top: int,
    left: int,
    bottom: int,
    right: int,
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
    grid_cols: int = 3,
    grid_rows: int = 3,
) -> Image.Image:
    """Draw a polished debug overlay with grid lines and labeled panels."""
    width, height = pil_img.size
    overlay = pil_img.copy()
    draw = ImageDraw.Draw(overlay, "RGBA")

    # Full grid lines
    grid_color = (255, 255, 255, 80)
    for col in range(1, grid_cols):
        x = int(col * width / grid_cols)
        draw.line([(x, 0), (x, height)], fill=grid_color, width=2)
    for row in range(1, grid_rows):
        y = int(row * height / grid_rows)
        draw.line([(0, y), (width, y)], fill=grid_color, width=2)

    # Panel rectangles with semi-transparent fill
    for i, (name, (left, top, right, bottom)) in enumerate(panel_coords.items()):
        color = OVERLAY_COLORS[i % len(OVERLAY_COLORS)]
        fill_color = color + (40,)
        draw.rectangle([left, top, right - 1, bottom - 1], fill=fill_color)
        draw.rectangle([left, top, right - 1, bottom - 1], outline=color, width=4)

        label = name.replace("_", " ")
        font = _get_overlay_font()
        text_pos = (left + 6, top + 6)
        bbox = draw.textbbox(text_pos, label, font=font) if font else draw.textbbox(text_pos, label)
        draw.rectangle(
            [bbox[0] - 3, bbox[1] - 3, bbox[2] + 3, bbox[3] + 3],
            fill=color + (220,),
        )
        draw.text(text_pos, label, fill=(0, 0, 0), font=font)

    return overlay.convert("RGB")


def _get_overlay_font(size: int = 16) -> ImageFont.FreeTypeFont | None:
    """Return a readable TrueType font if available, otherwise None."""
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "arial.ttf",
        "DejaVuSans-Bold.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return None


def _parse_panel_mapping(
    mapping_str: str, n_slots: int = 5, grid_cols: int = 3, grid_rows: int = 3
) -> list[int]:
    """Parse and clamp a panel mapping string to valid grid cell indices.

    Each index refers to a cell in row-major order (0=top-left, 1=top-center, ...).
    Out-of-range indices are clamped to the nearest valid cell and logged.
    """
    max_cell = grid_cols * grid_rows - 1
    try:
        indices = [int(x.strip()) for x in mapping_str.split(",")]
    except ValueError:
        logger.warning("Invalid panel mapping '%s', using default", mapping_str)
        return [0, 1, 2, 4, 5]

    if len(indices) != n_slots:
        logger.warning(
            "Panel mapping has %d indices, expected %d; using default",
            len(indices),
            n_slots,
        )
        return [0, 1, 2, 4, 5]

    clamped: list[int] = []
    for idx in indices:
        if idx < 0 or idx > max_cell:
            logger.warning("Panel mapping index %d out of range [0, %d]; clamping", idx, max_cell)
            idx = max(0, min(idx, max_cell))
        clamped.append(idx)

    return clamped


def _apply_panel_mapping(
    mapping: list[int],
    grid_cols: int,
    grid_rows: int,
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
                "detect_grid": ("BOOLEAN", {"default": True}),
                "panel_mapping": ("STRING", {"default": "0,1,2,4,5", "multiline": False}),
                "grid_size": (
                    "INT",
                    {"default": DEFAULT_GRID_SIZE, "min": 1, "max": 10000, "step": 1},
                ),
                "filename_prefix": ("STRING", {"default": "msr"}),
                "output_folder": ("STRING", {"default": "msr_crops"}),
                "subfolder_by_layout": ("BOOLEAN", {"default": False}),
                "include_timestamp": ("BOOLEAN", {"default": False}),
                "save_to_disk": ("BOOLEAN", {"default": True}),
                "save_backup_panels": ("BOOLEAN", {"default": False}),
                "save_debug_overlay": ("BOOLEAN", {"default": False}),
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
    CATEGORY = "Licon-MSR / Utils"

    # --- Public API ----------------------------------------------------------

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
        """Crop the contact sheet into panels with full feature support.

        Args:
            contact_sheet: Input image tensor (B, H, W, C) in [0, 1].
            layout: Layout preset name.
            detect_grid: Auto-detect grid lines from image content.
            panel_mapping: Comma-separated cell indices for panel assignment.
            grid_size: Reference grid dimension (used when detect_grid is False).
            filename_prefix: Prefix for saved filenames.
            output_folder: Output directory for saved files.
            subfolder_by_layout: Create a subfolder per layout preset.
            include_timestamp: Add timestamp subfolder to output path.
            save_to_disk: Whether to write crops to disk.
            save_backup_panels: Whether to also process backup panels.
            save_debug_overlay: Whether to save the debug overlay image.

        Returns:
            Tuple of (5 IMAGE, 5 MASK, debug overlay IMAGE, status STRING).
        """
        logger.info(
            "MSR Contact Sheet Cropper started (layout=%s, batch=%d)",
            layout,
            contact_sheet.shape[0],
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

            # Parse and clamp panel mapping
            mapping = _parse_panel_mapping(
                panel_mapping, len(PANEL_OUTPUT_NAMES), grid_cols, grid_rows
            )

            # Auto-detect grid lines if requested
            detected_lines = None
            grid_detect_status = "off"
            if detect_grid:
                first_pil = _tensor_to_pil(contact_sheet[:1])
                img_np = np.array(first_pil)
                detected_lines = _detect_grid_lines(img_np, grid_cols)
                if detected_lines is not None:
                    h_lines, v_lines = detected_lines
                    logger.info("Grid detected: H lines %s, V lines %s", h_lines, v_lines)
                else:
                    logger.warning(
                        "Grid detection failed, falling back to grid_size=%d",
                        grid_size,
                    )

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
                    pil_for_overlay = _tensor_to_pil(img_tensor)
                    overlay_pil = _draw_debug_overlay(
                        pil_for_overlay, panel_coords, grid_cols, grid_rows
                    )
                    overlay_tensors.append(_pil_to_tensor(overlay_pil))

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
                    left, top, right, bottom = panel_coords[name]
                    crop = _crop_tensor(img_tensor, top, left, bottom, right)
                    all_panel_tensors[panel_idx].append(crop)

                    mask = _make_mask(bottom - top, right - left, 1)
                    all_mask_tensors[panel_idx].append(mask)

                    if save_to_disk:
                        pil_crop = _tensor_to_pil(crop)
                        self._save_panel(
                            pil_crop,
                            save_dir,
                            filename_prefix,
                            name,
                            batch_idx,
                            batch_size,
                        )

                # Save backup panels
                if save_backup_panels:
                    for name, (left, top, right, bottom) in backup_coords.items():
                        crop = _crop_tensor(img_tensor, top, left, bottom, right)
                        if save_to_disk:
                            pil_crop = _tensor_to_pil(crop)
                            self._save_panel(
                                pil_crop,
                                save_dir,
                                filename_prefix,
                                name,
                                batch_idx,
                                batch_size,
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

            # Build human-readable status string
            if detect_grid:
                grid_detect_status = "success" if detected_lines else "fallback"

            warnings: list[str] = []
            aspect = width / float(height)
            if not 0.9 <= aspect <= 1.1:
                warnings.append("non-square input")

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

            # Write manifest with run metadata
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
            empty_overlay = torch.zeros((1, panel_size * 3, panel_size * 3, 3), dtype=torch.float32)
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

    # --- Private helpers -----------------------------------------------------

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

    @staticmethod
    def _coords_from_detected(
        width: int,
        height: int,
        h_lines: list[int],
        v_lines: list[int],
        mapping: list[int],
        grid_cols: int,
        grid_rows: int,
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
        width: int,
        height: int,
        h_lines: list[int],
        v_lines: list[int],
        preset: dict[str, Any],
        grid_cols: int,
        grid_rows: int,
    ) -> dict[str, tuple[int, int, int, int]]:
        """Build backup panel pixel coordinates from detected grid lines."""
        all_h = [0] + sorted(h_lines) + [height]
        all_v = [0] + sorted(v_lines) + [width]

        result: dict[str, tuple[int, int, int, int]] = {}
        for name, cell in preset.get("backup", {}).items():
            col_start, row_start, col_end, row_end = cell
            result[name] = (
                all_v[col_start],
                all_h[row_start],
                all_v[col_end],
                all_h[row_end],
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
            mapping = _parse_panel_mapping(
                panel_mapping, len(PANEL_OUTPUT_NAMES), grid_cols, grid_rows
            )

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
                    pil_panel = _tensor_to_pil(panel[b : b + 1])
                    pil_resized = pil_panel.resize((cell_size, cell_size), Image.LANCZOS)
                    resized_tensor = _pil_to_tensor(pil_resized)

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


# --- Node registration -------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "MSRContactSheetCropper": MSRContactSheetCropper,
    "MSRContactSheetAssembler": MSRContactSheetAssembler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MSRContactSheetCropper": "MSR Crop Contact Sheet",
    "MSRContactSheetAssembler": "MSR Assemble Contact Sheet",
}

logger.info(
    "Licon-MSR Utils: MSRContactSheetCropper + MSRContactSheetAssembler loaded successfully"
)
