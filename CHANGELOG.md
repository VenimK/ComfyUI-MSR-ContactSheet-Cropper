# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [1.2.0] - 2026-07-17

### Changed

- Major refactor: split monolithic `__init__.py` into a proper `msr_nodes` package
  (`constants`, `types`, `image_utils`, `grid`, `cropper`, `assembler`).
- Added `PanelCoords` and `GridSpec` dataclasses for cleaner internal APIs.
- Node categories split into `Licon-MSR / Crop` and `Licon-MSR / Assemble`.

### Added

- ComfyUI web extension (`js/msr_nodes.js`) that gives MSR nodes distinctive
  default colors in the workflow graph.
- `CONTRIBUTING.md` and `CODE_OF_CONDUCT.md`.
- PyPI-friendly project URLs (Documentation, Changelog, Issues).

## [1.1.0] - 2026-07-17

### Added

- **MSR Contact Sheet Cropper**
  - Auto-detect grid lines from image content.
  - Layout presets: `ideogram_3x3`, `midjourney_4x4`, `custom_3x3`.
  - Batch processing support (B > 1) with indexed file names.
  - Visual debug overlay output with grid lines and labeled panels.
  - Panel-to-slot remapping via comma-separated cell indices.
  - Mask outputs for each panel (useful for inpainting workflows).
  - GPU-side tensor cropping.
  - Live `status` string output with layout, grid, batch, and save info.
  - Optional timestamp subfolders and per-layout subfolders.
  - `manifest.json` written alongside saved crops with run metadata.
  - `save_debug_overlay` option to persist the overlay image.

- **MSR Contact Sheet Assembler**
  - New reverse node that composites panels back into a contact sheet grid.

- Documentation
  - Complete README rewrite with feature list, I/O tables, layout diagrams, and usage examples.
  - This changelog.

## [1.0.0] - Earlier

- Initial release: basic contact sheet cropping into 5 panels for Licon-MSR workflows.
