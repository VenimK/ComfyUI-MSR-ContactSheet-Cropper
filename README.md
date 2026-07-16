# ComfyUI MSR Contact Sheet Cropper

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![ComfyUI Custom Node](https://img.shields.io/badge/ComfyUI-Custom%20Node-orange.svg)](https://github.com/comfyanonymous/ComfyUI)
[![Tests](https://img.shields.io/badge/tests-35%20passing-brightgreen.svg)](#tests)

A custom node pack for [ComfyUI](https://github.com/comfyanonymous/ComfyUI) that crops and assembles contact sheets for **Licon-MSR** workflows. Supports Ideogram 3x3, Midjourney 4x4, and custom layouts with auto-detection, batch processing, mask outputs, debug overlays, and panel remapping.

## Nodes

### MSR Contact Sheet Cropper

Crops a contact sheet into individual panels with GPU-side tensor operations.

### MSR Contact Sheet Assembler

Reverse operation — composites individual panels back into a contact sheet grid.

## Features

- **Auto-detect grid lines** — Analyzes image content to find cell boundaries automatically
- **Layout presets** — Ideogram 3x3, Midjourney 4x4, Custom 3x3
- **Batch processing** — Handles B > 1 input tensors, outputs stacked batches
- **Visual debug overlay** — Labeled rectangle overlay output for verification
- **Panel-to-slot remapping** — Custom cell indices for non-standard layouts
- **Reverse assemble mode** — Reconstruct contact sheets from individual panels
- **Smart file organization** — Optional timestamp subfolders for saved crops
- **Mask outputs** — White masks for each panel (useful for inpainting workflows)
- **GPU-side cropping** — Tensor slicing keeps data on-device when possible
- **Live status output** — String output with runtime info (layout, grid, batch size, etc.)
- **ComfyUI Manager friendly** — Proper `NODE_CLASS_MAPPINGS` + display names

## Panel Layout (Ideogram 3x3)

```
+---------+---------+---------+
|         |         |         |
|  bg     |  slot1  |  slot2  |   Row 1
| cell 0  | cell 1  | cell 2  |
+---------+---------+---------+
|         |         |         |
| backup_A|  slot3  |  slot4  |   Row 2
| cell 3  | cell 4  | cell 5  |
+---------+---------+---------+
|         |         |         |
| backup_B|backup_bg|backup_w |   Row 3
| cell 6  | cell 7  | cell 8  |
+---------+---------+---------+
```

Default panel mapping: `0,1,2,4,5` (bg=cell0, slot1=cell1, slot2=cell2, slot3=cell4, slot4=cell5)

## Installation

### Method 1: ComfyUI Manager (Recommended)

1. Open **ComfyUI Manager**
2. Click **Install via Git URL**
3. Paste this URL:
    ```
    https://github.com/VenimK/ComfyUI-MSR-ContactSheet-Cropper
    ```
4. Restart ComfyUI

### Method 2: Manual Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/VenimK/ComfyUI-MSR-ContactSheet-Cropper
```

Restart ComfyUI.

## Cropper Node

### Inputs

| Input | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `contact_sheet` | IMAGE | Yes | — | Contact sheet image tensor (B, H, W, C) |
| `layout` | Combo | No | `ideogram_3x3` | Layout preset: `ideogram_3x3`, `midjourney_4x4`, `custom_3x3` |
| `detect_grid` | BOOLEAN | No | `False` | Auto-detect grid lines from image content |
| `grid_size` | INT | No | `1000` | Reference grid dimension (used when detect_grid is False) |
| `panel_mapping` | STRING | No | `"0,1,2,4,5"` | Comma-separated cell indices for panel assignment |
| `filename_prefix` | STRING | No | `"msr"` | Prefix for saved filenames |
| `save_to_disk` | BOOLEAN | No | `True` | Whether to persist crops to disk |
| `save_backup_panels` | BOOLEAN | No | `False` | Also save backup panels to disk |
| `include_timestamp` | BOOLEAN | No | `False` | Add timestamp subfolder to output path |
| `output_folder` | STRING | No | `"msr_crops"` | Directory for saved files |

### Outputs

| Output | Type | Description |
|--------|------|-------------|
| `background` | IMAGE | Panel assigned to cell 0 (top-left by default) |
| `slot1` | IMAGE | Panel assigned to cell 1 |
| `slot2` | IMAGE | Panel assigned to cell 2 |
| `slot3` | IMAGE | Panel assigned to cell 4 |
| `slot4` | IMAGE | Panel assigned to cell 5 |
| `mask_bg` | MASK | White mask for background panel |
| `mask_slot1` | MASK | White mask for slot1 panel |
| `mask_slot2` | MASK | White mask for slot2 panel |
| `mask_slot3` | MASK | White mask for slot3 panel |
| `mask_slot4` | MASK | White mask for slot4 panel |
| `debug_overlay` | IMAGE | Contact sheet with labeled crop rectangles |
| `status` | STRING | Runtime status info (layout, grid, batch, mapping, save path) |

## Assembler Node

### Inputs

| Input | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `background` | IMAGE | Yes | — | Background panel tensor |
| `slot1` | IMAGE | Yes | — | Slot1 panel tensor |
| `slot2` | IMAGE | Yes | — | Slot2 panel tensor |
| `slot3` | IMAGE | Yes | — | Slot3 panel tensor |
| `slot4` | IMAGE | Yes | — | Slot4 panel tensor |
| `layout` | Combo | No | `ideogram_3x3` | Layout preset |
| `panel_mapping` | STRING | No | `"0,1,2,4,5"` | Cell indices for panel placement |
| `cell_size` | INT | No | `333` | Target pixel size for each grid cell |
| `background_color` | INT | No | `0` | Fill color for empty cells (0-255) |

### Outputs

| Output | Type | Description |
|--------|------|-------------|
| `contact_sheet` | IMAGE | Assembled contact sheet tensor |
| `status` | STRING | Assembly status info |

## Usage

### Basic Cropping

1. Generate a 3x3 contact sheet using Ideogram (or similar).
2. Load **MSR Contact Sheet Cropper** in ComfyUI.
3. Connect the contact sheet image to `contact_sheet`.
4. The node outputs 5 IMAGE tensors + 5 MASK tensors + debug overlay + status string.

### Auto-Detection

Enable `detect_grid` to automatically find grid lines from image content. Useful for non-standard contact sheet sizes or when grid lines are visually distinct.

### Panel Remapping

Use `panel_mapping` to reassign which grid cell maps to which output slot. Format: comma-separated cell indices in row-major order.

Example: `"1,0,2,4,5"` swaps background and slot1.

### Batch Processing

Feed a batch tensor (B > 1) to process multiple contact sheets at once. Outputs are stacked along the batch dimension. Saved files are indexed: `msr_background_0000.png`, `msr_background_0001.png`, etc.

### Reverse Assembly

Use **MSR Contact Sheet Assembler** to composite panels back into a grid. Useful for previewing modified panels in contact sheet format.

### Example Workflow

```
[Load Image] → [MSR Contact Sheet Cropper] → background → [Licon-MSR Background Node]
                                         → slot1      → [Licon-MSR Slot1 Node]
                                         → slot2      → [Licon-MSR Slot2 Node]
                                         → slot3      → [Licon-MSR Slot3 Node]
                                         → slot4      → [Licon-MSR Slot4 Node]
                                         → debug_overlay → [Preview Image]
                                         → status        → [Show Text]
```

## Requirements

- Python >= 3.9
- ComfyUI
- torch
- numpy
- pillow

## Development

### Setup

```bash
git clone https://github.com/VenimK/ComfyUI-MSR-ContactSheet-Cropper
cd ComfyUI-MSR-ContactSheet-Cropper
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### Linting

```bash
ruff check .
black --check .
```

### Tests

```bash
pytest
```

35 tests covering output structure, panel content, layout presets, remapping, batch processing, grid detection, disk saving, error handling, and assembler.

## License

[MIT](LICENSE)

