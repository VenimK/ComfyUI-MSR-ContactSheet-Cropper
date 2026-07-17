# Contributing

Thanks for helping improve the MSR Contact Sheet Cropper!

## Quick Start

1. Fork and clone the repository.
2. Create a virtual environment with Python 3.9+.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```
4. Run tests:
   ```bash
   pytest
   ```

## Code Style

We use **Ruff** for linting/import sorting and **Black** for formatting.

```bash
ruff check .
black --check .
```

Auto-fix most issues with:

```bash
ruff check . --fix
black .
```

## Pull Request Process

1. Make sure tests pass and the code is formatted.
2. Update `CHANGELOG.md` under the `[Unreleased]` section.
3. Keep changes focused. If you are adding a big feature, open an issue first to discuss the design.
4. Update the README if your change affects user-facing behavior.

## Project Structure

- `__init__.py` — ComfyUI registration shim.
- `msr_nodes/` — Core implementation package.
  - `constants.py` — Layout presets and shared constants.
  - `types.py` — Internal dataclasses and typed helpers.
  - `image_utils.py` — Tensor/PIL conversions, masks, overlays.
  - `grid.py` — Grid detection and panel mapping.
  - `cropper.py` — `MSRContactSheetCropper` node.
  - `assembler.py` — `MSRContactSheetAssembler` node.
- `tests/` — pytest suite.

## Questions?

Open a [GitHub Discussion](https://github.com/VenimK/ComfyUI-MSR-ContactSheet-Cropper/discussions) or [Issue](https://github.com/VenimK/ComfyUI-MSR-ContactSheet-Cropper/issues).
