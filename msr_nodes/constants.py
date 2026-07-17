"""Constants and layout presets for MSR contact sheet processing."""

from __future__ import annotations

from typing import Any

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

PANEL_OUTPUT_NAMES = (
    "background",
    "slot1",
    "slot2",
    "slot3",
    "slot4",
)

# High-contrast colors for debug overlay rectangles (RGB)
OVERLAY_COLORS = [
    (255, 99, 71),  # tomato
    (50, 205, 50),  # lime green
    (30, 144, 255),  # dodger blue
    (255, 215, 0),  # gold
    (238, 130, 238),  # violet
    (0, 255, 255),  # cyan
    (255, 165, 0),  # orange
    (128, 0, 255),  # purple
    (255, 255, 255),  # white
]
