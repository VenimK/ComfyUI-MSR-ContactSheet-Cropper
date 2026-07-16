# ComfyUI MSR Contact Sheet Cropper

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![ComfyUI Custom Node](https://img.shields.io/badge/ComfyUI-Custom%20Node-orange.svg)](https://github.com/comfyanonymous/ComfyUI)

A custom node for [ComfyUI](https://github.com/comfyanonymous/ComfyUI) that automatically crops 3x3 Ideogram contact sheets into the 5 panels needed for **Licon-MSR** workflows.

## Features

- Crops the 5 essential panels (background + slot1 through slot4)
- Configurable grid size (default 1000 for Ideogram)
- Option to also save 4 backup panels
- Optional disk saving with custom filename prefix and output folder
- Input validation with aspect-ratio warnings
- Clean logging via Python `logging` module

## Panel Layout

The node expects a 3x3 contact sheet (e.g. 1000x1000 from Ideogram). The grid is divided as follows:

```
+---------+---------+---------+
|         |         |         |
|  bg     |  slot1  |  slot2  |   Row 1 (0–333)
| (0,0)   |(333,0)  |(667,0)  |
+---------+---------+---------+
|         |         |         |
| backup_A|  slot3  |  slot4  |   Row 2 (333–667)
| (0,333) |(333,333)|(667,333)|
+---------+---------+---------+
|         |         |         |
| backup_B|backup_bg|backup_w |   Row 3 (667–1000)
| (0,667) |(333,667)|(667,667)|
+---------+---------+---------+
```

**Main outputs (always returned):** background, slot1, slot2, slot3, slot4

**Backup panels (optional, disk only):** backup_A, backup_B, backup_bg, backup_wide

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

## Node Inputs

| Input | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `contact_sheet` | IMAGE | Yes | — | The 3x3 contact sheet image tensor |
| `grid_size` | INT | No | `1000` | Reference grid dimension for coordinate scaling |
| `filename_prefix` | STRING | No | `"msr"` | Prefix for saved filenames |
| `save_to_disk` | BOOLEAN | No | `True` | Whether to persist crops to disk |
| `save_backup_panels` | BOOLEAN | No | `False` | Also save the 4 backup panels |
| `output_folder` | STRING | No | `"msr_crops"` | Directory for saved files |

## Node Outputs

| Output | Type | Description |
|--------|------|-------------|
| `background` | IMAGE | Top-left panel (Panel 1) |
| `slot1` | IMAGE | Top-center panel (Panel 2) |
| `slot2` | IMAGE | Top-right panel (Panel 3) |
| `slot3` | IMAGE | Center panel (Panel 5) |
| `slot4` | IMAGE | Center-right panel (Panel 6) |

## Usage

1. Generate a 3x3 contact sheet using Ideogram (or any tool that produces a similar layout).
2. Load the **MSR Contact Sheet Cropper** node in ComfyUI.
3. Connect the contact sheet image to the `contact_sheet` input.
4. The node outputs 5 cropped IMAGE tensors ready for downstream Licon-MSR nodes.

### Example Workflow

```
[Load Image] → [MSR Contact Sheet Cropper] → background → [Licon-MSR Background Node]
                                         → slot1      → [Licon-MSR Slot1 Node]
                                         → slot2      → [Licon-MSR Slot2 Node]
                                         → slot3      → [Licon-MSR Slot3 Node]
                                         → slot4      → [Licon-MSR Slot4 Node]
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

## License

[MIT](LICENSE)

