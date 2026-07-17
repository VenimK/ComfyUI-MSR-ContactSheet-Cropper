# ComfyUI MSR Contact Sheet Cropper

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![ComfyUI Custom Node](https://img.shields.io/badge/ComfyUI-Custom%20Node-orange.svg)](https://github.com/comfyanonymous/ComfyUI)
[![Tests](https://img.shields.io/badge/tests-38%20passing-brightgreen.svg)](#tests)
[![ComfyUI Manager](https://img.shields.io/badge/ComfyUI%20Manager-Install%20via%20Git%20URL-blueviolet.svg)](#installation)

> Turn a 3x3 Ideogram or Midjourney contact sheet into the 5 Licon-MSR panels — automatically.

A custom node pack for [ComfyUI](https://github.com/comfyanonymous/ComfyUI) that crops and assembles contact sheets for **Licon-MSR** workflows. Supports Ideogram 3x3, Midjourney 4x4, and custom layouts with auto-detection, batch processing, mask outputs, debug overlays, and panel remapping.

## Nodes

### MSR Contact Sheet Cropper

Crops a contact sheet into individual panels with GPU-side tensor operations.

### MSR Contact Sheet Assembler

Reverse operation — composites individual panels back into a contact sheet grid.

## Why Use This?

- **No manual cropping** — Drop in a contact sheet and get 5 panels instantly.
- **Works with multiple generators** — Ideogram 3x3, Midjourney 4x4, or define your own mapping.
- **Visual feedback** — The `debug_overlay` output shows exactly where each panel was cut.
- **Inpainting-ready** — Each panel comes with a matching white mask.
- **Reversible** — The Assembler node lets you preview modified panels back as a contact sheet.

## Quick Start

1. Generate a contact sheet (e.g. Ideogram 3x3).
2. Add **MSR Crop Contact Sheet** to your ComfyUI workflow.
3. Connect your image to `contact_sheet`.
4. Connect `background`, `slot_1` … `slot_4` to your Licon-MSR nodes.
5. (Optional) View `debug_overlay` to verify the cuts, and `status` to see run info.

```
[Load Image] -> [MSR Crop Contact Sheet] -> background  -> [Licon-MSR bg]
                                         -> slot_1      -> [Licon-MSR slot1]
                                         -> slot_2      -> [Licon-MSR slot2]
                                         -> slot_3      -> [Licon-MSR slot3]
                                         -> slot_4      -> [Licon-MSR slot4]
                                         -> debug_overlay -> [Preview Image]
                                         -> status          -> [Show Text]
```

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

### Panel Mapping Cheatsheet

Use `panel_mapping` to reassign which grid cell goes to which output. Indices are in **row-major order**:

```
+------+------+------+
|  0   |  1   |  2   |  <- row 0
+------+------+------+
|  3   |  4   |  5   |  <- row 1
+------+------+------+
|  6   |  7   |  8   |  <- row 2
+------+------+------+
```

Examples:

| Mapping | Result |
|---------|--------|
| `0,1,2,4,5` | Default Ideogram layout |
| `1,0,2,4,5` | Swap background and slot1 |
| `0,1,2,3,4` | Custom layout: slot3 uses cell 3 (middle-left) |

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
| `detect_grid` | BOOLEAN | No | `True` | Auto-detect grid lines from image content |
| `panel_mapping` | STRING | No | `"0,1,2,4,5"` | Comma-separated cell indices for panel assignment |
| `grid_size` | INT | No | `1000` | Reference grid dimension (used when detect_grid is False) |
| `filename_prefix` | STRING | No | `"msr"` | Prefix for saved filenames |
| `output_directory` | STRING | No | `"msr_crops"` | Directory for saved files |
| `subfolder_by_layout` | BOOLEAN | No | `False` | Create a subfolder per layout preset |
| `include_timestamp` | BOOLEAN | No | `False` | Add timestamp subfolder to output path |
| `save_to_disk` | BOOLEAN | No | `True` | Whether to persist crops to disk |
| `save_backup_panels` | BOOLEAN | No | `False` | Also save backup panels to disk |
| `save_debug_overlay` | BOOLEAN | No | `False` | Save the debug overlay image alongside crops |

### Outputs

| Output | Type | Description |
|--------|------|-------------|
| `background` | IMAGE | Panel assigned to cell 0 (top-left by default) |
| `slot_1` | IMAGE | Panel assigned to cell 1 |
| `slot_2` | IMAGE | Panel assigned to cell 2 |
| `slot_3` | IMAGE | Panel assigned to cell 4 |
| `slot_4` | IMAGE | Panel assigned to cell 5 |
| `mask_bg` | MASK | White mask for background panel |
| `mask_slot_1` | MASK | White mask for slot_1 panel |
| `mask_slot_2` | MASK | White mask for slot_2 panel |
| `mask_slot_3` | MASK | White mask for slot_3 panel |
| `mask_slot_4` | MASK | White mask for slot_4 panel |
| `debug_overlay` | IMAGE | Contact sheet with labeled crop rectangles and grid lines |
| `status` | STRING | Runtime status info (layout, grid, batch, mapping, save path, warnings) |

## Assembler Node

### Inputs

| Input | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `background` | IMAGE | Yes | — | Background panel tensor |
| `slot_1` | IMAGE | Yes | — | Slot 1 panel tensor |
| `slot_2` | IMAGE | Yes | — | Slot 2 panel tensor |
| `slot_3` | IMAGE | Yes | — | Slot 3 panel tensor |
| `slot_4` | IMAGE | Yes | — | Slot 4 panel tensor |
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
2. Load **MSR Crop Contact Sheet** in ComfyUI.
3. Connect the contact sheet image to `contact_sheet`.
4. The node outputs 5 IMAGE tensors + 5 MASK tensors + debug overlay + status string.

The default settings (`detect_grid=True`, `layout=ideogram_3x3`) work for most Ideogram contact sheets out of the box.

### Auto-Detection

`detect_grid=True` (default) automatically finds grid lines from image content. If detection fails, the node silently falls back to the `grid_size` reference grid. Check `status` to see whether detection succeeded or used the fallback.

### Panel Remapping

Use `panel_mapping` to reassign which grid cell maps to which output slot. Format: comma-separated cell indices in row-major order.

Example: `"1,0,2,4,5"` swaps background and slot_1.

### Batch Processing

Feed a batch tensor (B > 1) to process multiple contact sheets at once. Outputs are stacked along the batch dimension. Saved files are indexed: `msr_background_0000.png`, `msr_background_0001.png`, etc.

### File Organization

- `subfolder_by_layout=True` → saves under `msr_crops/ideogram_3x3/`
- `include_timestamp=True` → adds a `YYYY-MM-DD_HH-MM-SS` subfolder
- `save_debug_overlay=True` → writes `msr_debug_overlay.png` alongside the crops
- A `manifest.json` is written to each save folder describing the run (layout, mapping, image size, detection status)

### Reverse Assembly

Use **MSR Assemble Contact Sheet** to composite panels back into a grid. Useful for previewing modified panels in contact sheet format.

### Example Workflow

```
[Load Image] → [MSR Crop Contact Sheet] → background  → [Licon-MSR Background Node]
                                       → slot_1      → [Licon-MSR Slot1 Node]
                                       → slot_2      → [Licon-MSR Slot2 Node]
                                       → slot_3      → [Licon-MSR Slot3 Node]
                                       → slot_4      → [Licon-MSR Slot4 Node]
                                       → debug_overlay → [Preview Image]
                                       → status          → [Show Text]
```

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| Panels are misaligned | `detect_grid` found strong borders/decorations instead of panel edges | Set `detect_grid=False` and tune `grid_size` to match your contact sheet |
| Outputs are all black | Input tensor shape is wrong (not BHWC) or dimensions are 0 | Check `status` output for the error message |
| Slot order is wrong | The default `panel_mapping` doesn't match your generator | Use `debug_overlay` to see the detected cells and adjust `panel_mapping` |
| Saved files are hard to find | Default folder is `msr_crops/` in the ComfyUI working directory | Set `output_directory` to an absolute path or enable `include_timestamp` |

## FAQ

**Q: Does this work with non-square contact sheets?**
A: Yes, but you may see a `non-square input` warning. The grid is still computed; verify with `debug_overlay`.

**Q: Can I process multiple contact sheets at once?**
A: Yes. Pass a batch tensor (B > 1) to `contact_sheet`. Outputs are stacked along the batch dimension.

**Q: What is `grid_size` for?**
A: It's the reference size used when `detect_grid=False`. For a 1000x1000 Ideogram sheet, leave it at `1000`.

**Q: Where are my files saved?**
A: Next to the ComfyUI working directory by default, in `msr_crops/`. A `manifest.json` is written with each run's metadata.

## Requirements

- Python >= 3.9
- ComfyUI
- torch, numpy, pillow

## Project Structure

```
ComfyUI-MSR-ContactSheet-Cropper/
├── __init__.py              # ComfyUI registration shim
├── msr_nodes/               # Core implementation package
│   ├── constants.py         # Layout presets and shared constants
│   ├── types.py             # Internal dataclasses (PanelCoords, GridSpec)
│   ├── image_utils.py       # Tensor/PIL conversion, masks, debug overlay
│   ├── grid.py              # Grid detection and panel mapping
│   ├── cropper.py           # MSRContactSheetCropper node
│   └── assembler.py         # MSRContactSheetAssembler node
├── js/
│   └── msr_nodes.js         # ComfyUI web extension for node colors
├── tests/                   # pytest suite
├── README.md
├── CHANGELOG.md
├── CONTRIBUTING.md
└── CODE_OF_CONDUCT.md
```

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

38 tests covering output structure, panel content, layout presets, remapping, batch processing, grid detection, disk saving, error handling, convenience features, and assembler.

## License

[MIT](LICENSE)

