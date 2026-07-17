"""Tests for the MSRContactSheetCropper and MSRContactSheetAssembler nodes."""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from __init__ import (
    LAYOUT_PRESETS,
    PANEL_OUTPUT_NAMES,
    MSRContactSheetAssembler,
    MSRContactSheetCropper,
)

# --- Fixtures -----------------------------------------------------------------


def _make_contact_sheet(size: int = 1000) -> torch.Tensor:
    """Create a synthetic contact sheet tensor with distinct panel colors.

    Each cell in the 3x3 grid gets a unique color so we can verify cropping.
    """
    cell = size // 3
    img = np.zeros((size, size, 3), dtype=np.uint8)

    colors = [
        (255, 0, 0),  # (0,0) red – background
        (0, 255, 0),  # (1,0) green – slot1
        (0, 0, 255),  # (2,0) blue – slot2
        (255, 255, 0),  # (0,1) yellow – backup_A
        (255, 0, 255),  # (1,1) magenta – slot3
        (0, 255, 255),  # (2,1) cyan – slot4
        (128, 0, 0),  # (0,2) dark red – backup_B
        (0, 128, 0),  # (1,2) dark green – backup_bg
        (128, 128, 0),  # (2,2) olive – backup_wide
    ]

    for idx, (r, g, b) in enumerate(colors):
        row = idx // 3
        col = idx % 3
        img[row * cell : (row + 1) * cell, col * cell : (col + 1) * cell] = [r, g, b]

    tensor = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0)
    return tensor


@pytest.fixture
def cropper() -> MSRContactSheetCropper:
    return MSRContactSheetCropper()


@pytest.fixture
def assembler() -> MSRContactSheetAssembler:
    return MSRContactSheetAssembler()


@pytest.fixture
def contact_sheet() -> torch.Tensor:
    return _make_contact_sheet(1000)


# --- Cropper: output structure ------------------------------------------------


class TestCropOutputStructure:
    """Tests for output count, types, and shapes."""

    def test_returns_twelve_outputs(self, cropper, contact_sheet):
        result = cropper.crop_panels(contact_sheet, save_to_disk=False)
        assert len(result) == 12

    def test_first_five_are_image_tensors(self, cropper, contact_sheet):
        result = cropper.crop_panels(contact_sheet, save_to_disk=False)
        for i in range(5):
            assert isinstance(result[i], torch.Tensor)
            assert result[i].ndim == 4
            assert result[i].shape[0] == 1
            assert result[i].shape[3] == 3

    def test_next_five_are_mask_tensors(self, cropper, contact_sheet):
        result = cropper.crop_panels(contact_sheet, save_to_disk=False)
        for i in range(5, 10):
            assert isinstance(result[i], torch.Tensor)
            assert result[i].ndim == 3  # (B, H, W)
            assert result[i].shape[0] == 1

    def test_masks_are_white(self, cropper, contact_sheet):
        result = cropper.crop_panels(contact_sheet, save_to_disk=False)
        for i in range(5, 10):
            assert torch.all(result[i] == 1.0)

    def test_overlay_is_image_tensor(self, cropper, contact_sheet):
        result = cropper.crop_panels(contact_sheet, save_to_disk=False)
        overlay = result[10]
        assert isinstance(overlay, torch.Tensor)
        assert overlay.ndim == 4
        assert overlay.shape[3] == 3

    def test_status_is_string(self, cropper, contact_sheet):
        result = cropper.crop_panels(contact_sheet, save_to_disk=False)
        status = result[11]
        assert isinstance(status, str)
        assert "layout=ideogram_3x3" in status
        assert "detect=" in status


# --- Cropper: panel content ---------------------------------------------------


class TestCropPanelContent:
    """Verify that each panel has the correct color content."""

    def test_background_is_red(self, cropper, contact_sheet):
        result = cropper.crop_panels(contact_sheet, save_to_disk=False)
        bg = result[0]
        _, h, w, _ = bg.shape
        pixel = bg[0, h // 2, w // 2, :].cpu().numpy()
        assert pixel[0] > 0.9
        assert pixel[1] < 0.1
        assert pixel[2] < 0.1

    def test_slot1_is_green(self, cropper, contact_sheet):
        result = cropper.crop_panels(contact_sheet, save_to_disk=False)
        slot1 = result[1]
        _, h, w, _ = slot1.shape
        pixel = slot1[0, h // 2, w // 2, :].cpu().numpy()
        assert pixel[0] < 0.1
        assert pixel[1] > 0.9
        assert pixel[2] < 0.1

    def test_slot2_is_blue(self, cropper, contact_sheet):
        result = cropper.crop_panels(contact_sheet, save_to_disk=False)
        slot2 = result[2]
        _, h, w, _ = slot2.shape
        pixel = slot2[0, h // 2, w // 2, :].cpu().numpy()
        assert pixel[0] < 0.1
        assert pixel[1] < 0.1
        assert pixel[2] > 0.9

    def test_slot3_is_magenta(self, cropper, contact_sheet):
        result = cropper.crop_panels(contact_sheet, save_to_disk=False)
        slot3 = result[3]
        _, h, w, _ = slot3.shape
        pixel = slot3[0, h // 2, w // 2, :].cpu().numpy()
        assert pixel[0] > 0.9
        assert pixel[1] < 0.1
        assert pixel[2] > 0.9

    def test_slot4_is_cyan(self, cropper, contact_sheet):
        result = cropper.crop_panels(contact_sheet, save_to_disk=False)
        slot4 = result[4]
        _, h, w, _ = slot4.shape
        pixel = slot4[0, h // 2, w // 2, :].cpu().numpy()
        assert pixel[0] < 0.1
        assert pixel[1] > 0.9
        assert pixel[2] > 0.9


# --- Cropper: layout presets --------------------------------------------------


class TestLayoutPresets:
    """Tests for layout preset selection."""

    def test_ideogram_3x3_default(self, cropper, contact_sheet):
        result = cropper.crop_panels(contact_sheet, layout="ideogram_3x3", save_to_disk=False)
        status = result[11]
        assert "3x3" in status

    def test_midjourney_4x4(self, cropper, contact_sheet):
        result = cropper.crop_panels(contact_sheet, layout="midjourney_4x4", save_to_disk=False)
        status = result[11]
        assert "4x4" in status

    def test_custom_3x3(self, cropper, contact_sheet):
        result = cropper.crop_panels(
            contact_sheet,
            layout="custom_3x3",
            panel_mapping="0,1,2,3,4",
            save_to_disk=False,
        )
        # With mapping 0,1,2,3,4: slot3=cell3=(0,1) which is yellow
        slot3 = result[3]
        _, h, w, _ = slot3.shape
        pixel = slot3[0, h // 2, w // 2, :].cpu().numpy()
        assert pixel[0] > 0.9  # R high (yellow)
        assert pixel[1] > 0.9  # G high (yellow)


# --- Cropper: panel remapping -------------------------------------------------


class TestPanelRemapping:
    """Tests for panel-to-slot remapping."""

    def test_default_mapping(self, cropper, contact_sheet):
        result = cropper.crop_panels(contact_sheet, panel_mapping="0,1,2,4,5", save_to_disk=False)
        # Default: bg=cell0(red), slot1=cell1(green), slot2=cell2(blue)
        bg = result[0]
        _, h, w, _ = bg.shape
        assert bg[0, h // 2, w // 2, 0] > 0.9  # red

    def test_custom_mapping_swaps_panels(self, cropper, contact_sheet):
        # Map background to cell 1 (green), slot1 to cell 0 (red)
        result = cropper.crop_panels(contact_sheet, panel_mapping="1,0,2,4,5", save_to_disk=False)
        bg = result[0]
        _, h, w, _ = bg.shape
        pixel = bg[0, h // 2, w // 2, :].cpu().numpy()
        assert pixel[1] > 0.9  # green (was cell 1)

        slot1 = result[1]
        _, h2, w2, _ = slot1.shape
        pixel1 = slot1[0, h2 // 2, w2 // 2, :].cpu().numpy()
        assert pixel1[0] > 0.9  # red (was cell 0)

    def test_invalid_mapping_falls_back(self, cropper, contact_sheet):
        result = cropper.crop_panels(contact_sheet, panel_mapping="not,numbers", save_to_disk=False)
        # Should fall back to default and still produce 12 outputs
        assert len(result) == 12


# --- Cropper: batch processing ------------------------------------------------


class TestBatchProcessing:
    """Tests for batch (B > 1) support."""

    def test_batch_of_3(self, cropper):
        sheets = torch.cat([_make_contact_sheet(999) for _ in range(3)], dim=0)
        assert sheets.shape[0] == 3
        result = cropper.crop_panels(sheets, save_to_disk=False)
        assert len(result) == 12
        # Each panel should have batch dim = 3
        for i in range(5):
            assert result[i].shape[0] == 3
        # Masks too
        for i in range(5, 10):
            assert result[i].shape[0] == 3

    def test_batch_saves_indexed_files(self, cropper, tmp_path):
        sheets = torch.cat([_make_contact_sheet(999) for _ in range(2)], dim=0)
        cropper.crop_panels(sheets, save_to_disk=True, output_directory=str(tmp_path))
        for name in PANEL_OUTPUT_NAMES:
            assert os.path.exists(tmp_path / f"msr_{name}_0000.png")
            assert os.path.exists(tmp_path / f"msr_{name}_0001.png")


# --- Cropper: grid detection --------------------------------------------------


class TestGridDetection:
    """Tests for auto-detect grid lines."""

    def test_detect_grid_on_synthetic_sheet(self, cropper, contact_sheet):
        result = cropper.crop_panels(contact_sheet, detect_grid=True, save_to_disk=False)
        status = result[11]
        assert "detect=success" in status

    def test_detect_grid_still_crops(self, cropper, contact_sheet):
        result = cropper.crop_panels(contact_sheet, detect_grid=True, save_to_disk=False)
        # Background should still be red
        bg = result[0]
        _, h, w, _ = bg.shape
        pixel = bg[0, h // 2, w // 2, :].cpu().numpy()
        assert pixel[0] > 0.9


# --- Cropper: save to disk ----------------------------------------------------


class TestSaveToDisk:
    """Tests for disk saving functionality."""

    def test_save_main_panels(self, cropper, contact_sheet, tmp_path):
        cropper.crop_panels(
            contact_sheet,
            save_to_disk=True,
            output_directory=str(tmp_path),
            save_backup_panels=False,
        )
        for name in PANEL_OUTPUT_NAMES:
            assert os.path.exists(tmp_path / f"msr_{name}.png")

    def test_save_with_prefix(self, cropper, contact_sheet, tmp_path):
        cropper.crop_panels(
            contact_sheet,
            save_to_disk=True,
            filename_prefix="custom",
            output_directory=str(tmp_path),
        )
        for name in PANEL_OUTPUT_NAMES:
            assert os.path.exists(tmp_path / f"custom_{name}.png")

    def test_save_backup_panels(self, cropper, contact_sheet, tmp_path):
        cropper.crop_panels(
            contact_sheet,
            save_to_disk=True,
            save_backup_panels=True,
            output_directory=str(tmp_path),
        )
        for name in PANEL_OUTPUT_NAMES:
            assert os.path.exists(tmp_path / f"msr_{name}.png")
        # Check at least one backup panel exists
        backup_names = LAYOUT_PRESETS["ideogram_3x3"]["backup"].keys()
        for name in backup_names:
            assert os.path.exists(tmp_path / f"msr_{name}.png")

    def test_no_save_when_disabled(self, cropper, contact_sheet, tmp_path):
        cropper.crop_panels(
            contact_sheet,
            save_to_disk=False,
            output_directory=str(tmp_path),
        )
        assert len(list(tmp_path.iterdir())) == 0

    def test_timestamp_subfolder(self, cropper, contact_sheet, tmp_path):
        cropper.crop_panels(
            contact_sheet,
            save_to_disk=True,
            include_timestamp=True,
            output_directory=str(tmp_path),
        )
        subdirs = [d for d in tmp_path.iterdir() if d.is_dir()]
        assert len(subdirs) == 1
        # Files should be inside the timestamp subdir
        for name in PANEL_OUTPUT_NAMES:
            assert any(f"msr_{name}.png" in str(f) for f in subdirs[0].iterdir())


# --- Cropper: error handling --------------------------------------------------


class TestErrorHandling:
    """Tests for error handling and edge cases."""

    def test_invalid_tensor_shape_3d(self, cropper):
        bad_tensor = torch.zeros((100, 100, 3), dtype=torch.float32)
        result = cropper.crop_panels(bad_tensor, save_to_disk=False)
        assert len(result) == 12
        for i in range(5):
            assert result[i].shape[0] == 1
            assert torch.all(result[i] == 0)

    def test_invalid_tensor_shape_2d(self, cropper):
        bad_tensor = torch.zeros((100, 100), dtype=torch.float32)
        result = cropper.crop_panels(bad_tensor, save_to_disk=False)
        assert len(result) == 12
        for i in range(5):
            assert torch.all(result[i] == 0)

    def test_error_status_string(self, cropper):
        bad_tensor = torch.zeros((100, 100), dtype=torch.float32)
        result = cropper.crop_panels(bad_tensor, save_to_disk=False)
        status = result[11]
        assert "ERROR" in status

    def test_non_square_input_warns(self, cropper, caplog):
        img = np.zeros((1000, 1500, 3), dtype=np.uint8)
        tensor = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0)
        with caplog.at_level("WARNING"):
            result = cropper.crop_panels(tensor, save_to_disk=False)
        assert len(result) == 12
        assert "non-square" in result[11]


# --- Cropper: convenience features -------------------------------------------


class TestConvenienceFeatures:
    """Tests for manifest, debug overlay saving, and subfolder organization."""

    def test_manifest_written(self, cropper, contact_sheet, tmp_path):
        cropper.crop_panels(
            contact_sheet,
            save_to_disk=True,
            output_directory=str(tmp_path),
        )
        manifest_path = tmp_path / "manifest.json"
        assert manifest_path.exists()
        data = json.loads(manifest_path.read_text())
        assert data["layout"] == "ideogram_3x3"
        assert data["panel_mapping"] == [0, 1, 2, 4, 5]
        assert data["grid_detect_status"] in ("success", "fallback", "grid_size_fallback")

    def test_save_debug_overlay(self, cropper, contact_sheet, tmp_path):
        cropper.crop_panels(
            contact_sheet,
            save_to_disk=True,
            save_debug_overlay=True,
            output_directory=str(tmp_path),
        )
        assert (tmp_path / "msr_debug_overlay.png").exists()

    def test_subfolder_by_layout(self, cropper, contact_sheet, tmp_path):
        cropper.crop_panels(
            contact_sheet,
            save_to_disk=True,
            subfolder_by_layout=True,
            output_directory=str(tmp_path),
        )
        layout_dir = tmp_path / "ideogram_3x3"
        assert layout_dir.is_dir()
        for name in PANEL_OUTPUT_NAMES:
            assert (layout_dir / f"msr_{name}.png").exists()


# --- Assembler ----------------------------------------------------------------


class TestAssembler:
    """Tests for the MSRContactSheetAssembler node."""

    def test_assemble_returns_two_outputs(self, assembler, cropper, contact_sheet):
        crops = cropper.crop_panels(contact_sheet, save_to_disk=False)
        panels = crops[:5]
        result = assembler.assemble(*panels)
        assert len(result) == 2

    def test_assemble_output_shape(self, assembler, cropper, contact_sheet):
        crops = cropper.crop_panels(contact_sheet, save_to_disk=False)
        panels = crops[:5]
        sheet, status = assembler.assemble(*panels, cell_size=200)
        assert sheet.ndim == 4
        assert sheet.shape[2] == 600  # 3 * 200
        assert sheet.shape[1] == 600

    def test_assemble_status_string(self, assembler, cropper, contact_sheet):
        crops = cropper.crop_panels(contact_sheet, save_to_disk=False)
        panels = crops[:5]
        _, status = assembler.assemble(*panels)
        assert isinstance(status, str)
        assert "assembled" in status

    def test_assemble_with_custom_mapping(self, assembler, cropper, contact_sheet):
        crops = cropper.crop_panels(contact_sheet, save_to_disk=False)
        panels = crops[:5]
        sheet, status = assembler.assemble(*panels, panel_mapping="1,0,2,4,5")
        assert sheet.shape[0] == 1

    def test_assemble_error_handling(self, assembler):
        bad = torch.zeros((1, 10, 10, 3), dtype=torch.float32)
        result = assembler.assemble(bad, bad, bad, bad, torch.zeros((1, 0, 0, 3)))
        assert len(result) == 2
        assert isinstance(result[1], str)
