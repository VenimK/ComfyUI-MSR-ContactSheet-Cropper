"""Grid detection and panel mapping utilities."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from msr_nodes.constants import LAYOUT_PRESETS
from msr_nodes.types import PanelCoords

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def parse_panel_mapping(
    mapping_str: str,
    n_slots: int = 5,
    grid_cols: int = 3,
    grid_rows: int = 3,
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
            logger.warning(
                "Panel mapping index %d out of range [0, %d]; clamping",
                idx,
                max_cell,
            )
            idx = max(0, min(idx, max_cell))
        clamped.append(idx)

    return clamped


def detect_grid_lines(
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
        start = total_size // 4
        end = 3 * total_size // 4
        return [int(start + np.argmax(profile[start:end]))]

    spacing = total_size / (n_expected + 1)

    window = max(total_size // 200, 3)
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window) / window
    smoothed = np.convolve(profile, kernel, mode="same")

    best_score = -np.inf
    best_offset = 0.0
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

    half_window = max(total_size // 100, 2)
    refined: list[int] = []
    for line in lines:
        start = max(0, line - half_window)
        end = min(len(smoothed), line + half_window + 1)
        local_peak = start + int(np.argmax(smoothed[start:end]))
        refined.append(local_peak)

    if len(set(refined)) < n_expected:
        logger.debug(
            "Grid detection: only %d distinct lines found",
            len(set(refined)),
        )
        return None

    return refined


def coords_from_detected(
    width: int,
    height: int,
    h_lines: list[int],
    v_lines: list[int],
    mapping: list[int],
    grid_cols: int,
    grid_rows: int,
) -> dict[str, PanelCoords]:
    """Build pixel coordinates from detected grid lines + panel mapping."""
    all_h = [0] + sorted(h_lines) + [height]
    all_v = [0] + sorted(v_lines) + [width]

    from msr_nodes.constants import PANEL_OUTPUT_NAMES

    panel_coords: dict[str, PanelCoords] = {}
    for name, cell_idx in zip(PANEL_OUTPUT_NAMES, mapping):
        col = cell_idx % grid_cols
        row = cell_idx // grid_cols
        panel_coords[name] = PanelCoords(
            left=all_v[col],
            top=all_h[row],
            right=all_v[col + 1],
            bottom=all_h[row + 1],
        )
    return panel_coords


def coords_from_detected_backup(
    width: int,
    height: int,
    h_lines: list[int],
    v_lines: list[int],
    preset: dict,
    grid_cols: int,
    grid_rows: int,
) -> dict[str, PanelCoords]:
    """Build backup panel coordinates from detected grid lines."""
    all_h = [0] + sorted(h_lines) + [height]
    all_v = [0] + sorted(v_lines) + [width]

    backup_coords: dict[str, PanelCoords] = {}
    for name, (cs, rs, ce, re) in preset.get("backup", {}).items():
        if 0 <= cs < ce <= grid_cols and 0 <= rs < re <= grid_rows:
            backup_coords[name] = PanelCoords(
                left=all_v[cs],
                top=all_h[rs],
                right=all_v[ce],
                bottom=all_h[re],
            )
    return backup_coords


def coords_from_fallback(
    width: int,
    height: int,
    mapping: list[int],
    grid_cols: int,
    grid_rows: int,
) -> dict[str, PanelCoords]:
    """Build pixel coordinates from a uniform reference grid."""
    from msr_nodes.constants import PANEL_OUTPUT_NAMES

    panel_coords: dict[str, PanelCoords] = {}
    for name, cell_idx in zip(PANEL_OUTPUT_NAMES, mapping):
        col = cell_idx % grid_cols
        row = cell_idx // grid_cols
        panel_coords[name] = PanelCoords(
            left=int(col * width / grid_cols),
            top=int(row * height / grid_rows),
            right=int((col + 1) * width / grid_cols),
            bottom=int((row + 1) * height / grid_rows),
        )
    return panel_coords


def coords_from_fallback_backup(
    width: int,
    height: int,
    preset: dict,
    grid_cols: int,
    grid_rows: int,
) -> dict[str, PanelCoords]:
    """Build backup panel coordinates from a uniform reference grid."""
    backup_coords: dict[str, PanelCoords] = {}
    for name, (cs, rs, ce, re) in preset.get("backup", {}).items():
        backup_coords[name] = PanelCoords(
            left=int(cs * width / grid_cols),
            top=int(rs * height / grid_rows),
            right=int(ce * width / grid_cols),
            bottom=int(re * height / grid_rows),
        )
    return backup_coords


def resolve_panel_coords(
    img_width: int,
    img_height: int,
    layout: str,
    detect_grid: bool,
    mapping: list[int],
    grid_size: int,
    pil_img: object,
) -> tuple[dict[str, PanelCoords], dict[str, PanelCoords], str]:
    """Resolve panel coordinates using detection or fallback grid.

    Returns (panel_coords, backup_coords, detect_status).
    """
    preset = LAYOUT_PRESETS.get(layout, LAYOUT_PRESETS["ideogram_3x3"])
    grid_cols, grid_rows = preset["grid"]

    detected_lines = None
    detect_status = "off"
    panel_coords: dict[str, PanelCoords]
    backup_coords: dict[str, PanelCoords]

    if detect_grid:
        img_np = np.array(pil_img)
        detected_lines = detect_grid_lines(img_np, grid_cols)
        if detected_lines is not None:
            h_lines, v_lines = detected_lines
            logger.info("Grid detected: H lines %s, V lines %s", h_lines, v_lines)
            panel_coords = coords_from_detected(
                img_width, img_height, h_lines, v_lines, mapping, grid_cols, grid_rows
            )
            backup_coords = coords_from_detected_backup(
                img_width, img_height, h_lines, v_lines, preset, grid_cols, grid_rows
            )
            detect_status = "success"
        else:
            logger.info("Grid detection failed, falling back to grid_size=%d", grid_size)
            detect_status = "fallback"

    if not detect_grid or detected_lines is None:
        panel_coords = coords_from_fallback(img_width, img_height, mapping, grid_cols, grid_rows)
        backup_coords = coords_from_fallback_backup(
            img_width, img_height, preset, grid_cols, grid_rows
        )

    return panel_coords, backup_coords, detect_status
