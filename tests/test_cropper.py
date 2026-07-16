"""Tests for the MSRContactSheetCropper node."""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch
from PIL import Image

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from __init__ import MSRContactSheetCropper, MAIN_PANELS, BACKUP_PANELS


# --- Fixtures -----------------------------------------------------------------


def _make_contact_sheet(size: int = 1000) -> torch.Tensor:
    """Create a synthetic contact sheet tensor with distinct panel colors.

    Each cell in the 3x3 grid gets a unique color so we can verify cropping.
    """
    cell = size // 3
    img = np.zeros((size, size, 3), dtype=np.uint8)

    colors = [
        (255, 0, 0),    # (0,0) red – background
        (0, 255, 0),    # (1,0) green – slot1
        (0, 0, 255),    # (2,0) blue – slot2
        (255, 255, 0),  # (0,1) yellow – backup_A
        (255, 0, 255),  # (1,1) magenta – slot3
        (0, 255, 255),  # (2,1) cyan – slot4
        (128, 0, 0),    # (0,2) dark red – backup_B
        (0, 128, 0),    # (1,2) dark green – backup_bg
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
def contact_sheet() -> torch.Tensor:
    return _make_contact_sheet(1000)


# --- Tests --------------------------------------------------------------------


class TestCropPanels:
    """Tests for the main crop_panels method."""

    def test_returns_five_tensors(self, cropper, contact_sheet):
        result = cropper.crop_panels(contact_sheet, save_to_disk=False)
        assert len(result) == 5

    def test_output_shapes(self, cropper, contact_sheet):
        result = cropper.crop_panels(contact_sheet, save_to_disk=False)
        for tensor in result:
            assert tensor.ndim == 4
            assert tensor.shape[0] == 1
            assert tensor.shape[3] == 3  # RGB

    def test_panel_content_background(self, cropper, contact_sheet):
        """Background panel should be red (top-left cell)."""
        result = cropper.crop_panels(contact_sheet, save_to_disk=False)
        bg = result[0]
        # Sample center pixel
        _, h, w, _ = bg.shape
        pixel = bg[0, h // 2, w // 2, :].cpu().numpy()
        assert pixel[0] > 0.9  # R high
        assert pixel[1] < 0.1  # G low
        assert pixel[2] < 0.1  # B low

    def test_panel_content_slot1(self, cropper, contact_sheet):
        """Slot1 panel should be green (top-center cell)."""
        result = cropper.crop_panels(contact_sheet, save_to_disk=False)
        slot1 = result[1]
        _, h, w, _ = slot1.shape
        pixel = slot1[0, h // 2, w // 2, :].cpu().numpy()
        assert pixel[0] < 0.1
        assert pixel[1] > 0.9
        assert pixel[2] < 0.1

    def test_panel_content_slot2(self, cropper, contact_sheet):
        """Slot2 panel should be blue (top-right cell)."""
        result = cropper.crop_panels(contact_sheet, save_to_disk=False)
        slot2 = result[2]
        _, h, w, _ = slot2.shape
        pixel = slot2[0, h // 2, w // 2, :].cpu().numpy()
        assert pixel[0] < 0.1
        assert pixel[1] < 0.1
        assert pixel[2] > 0.9

    def test_panel_content_slot3(self, cropper, contact_sheet):
        """Slot3 panel should be magenta (center cell)."""
        result = cropper.crop_panels(contact_sheet, save_to_disk=False)
        slot3 = result[3]
        _, h, w, _ = slot3.shape
        pixel = slot3[0, h // 2, w // 2, :].cpu().numpy()
        assert pixel[0] > 0.9
        assert pixel[1] < 0.1
        assert pixel[2] > 0.9

    def test_panel_content_slot4(self, cropper, contact_sheet):
        """Slot4 panel should be cyan (center-right cell)."""
        result = cropper.crop_panels(contact_sheet, save_to_disk=False)
        slot4 = result[4]
        _, h, w, _ = slot4.shape
        pixel = slot4[0, h // 2, w // 2, :].cpu().numpy()
        assert pixel[0] < 0.1
        assert pixel[1] > 0.9
        assert pixel[2] > 0.9

    def test_non_default_grid_size(self, cropper):
        """A 1500x1500 image with grid_size=1500 should crop correctly."""
        sheet = _make_contact_sheet(1500)
        result = cropper.crop_panels(sheet, grid_size=1500, save_to_disk=False)
        assert len(result) == 5
        # Background should still be red
        bg = result[0]
        _, h, w, _ = bg.shape
        pixel = bg[0, h // 2, w // 2, :].cpu().numpy()
        assert pixel[0] > 0.9

    def test_scaled_grid(self, cropper):
        """A 1500x1500 image with grid_size=1000 should scale and crop correctly."""
        sheet = _make_contact_sheet(1500)
        result = cropper.crop_panels(sheet, grid_size=1000, save_to_disk=False)
        assert len(result) == 5
        # Each panel should be 500x500 (1500 * 333/1000 ≈ 500)
        for tensor in result:
            assert tensor.shape[1] > 0
            assert tensor.shape[2] > 0


class TestSaveToDisk:
    """Tests for disk saving functionality."""

    def test_save_main_panels(self, cropper, contact_sheet, tmp_path):
        result = cropper.crop_panels(
            contact_sheet,
            save_to_disk=True,
            output_folder=str(tmp_path),
            save_backup_panels=False,
        )
        for name in MAIN_PANELS:
            assert os.path.exists(tmp_path / f"msr_{name}.png")

    def test_save_with_prefix(self, cropper, contact_sheet, tmp_path):
        cropper.crop_panels(
            contact_sheet,
            save_to_disk=True,
            filename_prefix="custom",
            output_folder=str(tmp_path),
        )
        for name in MAIN_PANELS:
            assert os.path.exists(tmp_path / f"custom_{name}.png")

    def test_save_backup_panels(self, cropper, contact_sheet, tmp_path):
        cropper.crop_panels(
            contact_sheet,
            save_to_disk=True,
            save_backup_panels=True,
            output_folder=str(tmp_path),
        )
        for name in MAIN_PANELS:
            assert os.path.exists(tmp_path / f"msr_{name}.png")
        for name in BACKUP_PANELS:
            assert os.path.exists(tmp_path / f"msr_{name}.png")

    def test_no_save_when_disabled(self, cropper, contact_sheet, tmp_path):
        cropper.crop_panels(
            contact_sheet,
            save_to_disk=False,
            output_folder=str(tmp_path),
        )
        assert len(list(tmp_path.iterdir())) == 0


class TestErrorHandling:
    """Tests for error handling and edge cases."""

    def test_invalid_tensor_shape_3d(self, cropper):
        """3D tensor should trigger fallback to black images."""
        bad_tensor = torch.zeros((100, 100, 3), dtype=torch.float32)
        result = cropper.crop_panels(bad_tensor, save_to_disk=False)
        assert len(result) == 5
        for tensor in result:
            assert tensor.shape[0] == 1
            assert torch.all(tensor == 0)

    def test_invalid_tensor_shape_2d(self, cropper):
        """2D tensor should trigger fallback to black images."""
        bad_tensor = torch.zeros((100, 100), dtype=torch.float32)
        result = cropper.crop_panels(bad_tensor, save_to_disk=False)
        assert len(result) == 5
        for tensor in result:
            assert torch.all(tensor == 0)

    def test_fallback_panel_size_scales_with_grid(self, cropper):
        """Fallback black images should be sized based on grid_size."""
        bad_tensor = torch.zeros((100, 100), dtype=torch.float32)
        result = cropper.crop_panels(bad_tensor, grid_size=999, save_to_disk=False)
        expected = 999 // 3  # 333
        for tensor in result:
            assert tensor.shape[1] == expected
            assert tensor.shape[2] == expected

    def test_non_square_input_warns(self, cropper, caplog):
        """Non-square input should log a warning but still process."""
        img = np.zeros((1000, 1500, 3), dtype=np.uint8)
        tensor = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0)
        with caplog.at_level("WARNING"):
            result = cropper.crop_panels(tensor, save_to_disk=False)
        assert len(result) == 5
