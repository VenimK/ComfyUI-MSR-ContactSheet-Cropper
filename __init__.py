"""ComfyUI custom node for cropping MSR Contact Sheets into panels for Licon-MSR workflows.

This node takes a 3x3 Ideogram contact sheet image and crops it into the 5 essential
panels (background + 4 slots) used by Licon-MSR pipelines. Optionally, it can also
save backup panels and write all crops to disk.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

# --- Constants ---------------------------------------------------------------

DEFAULT_GRID_SIZE = 1000

# Main 5 panels (always returned as outputs) – coordinates in grid-space
MAIN_PANELS: dict[str, tuple[int, int, int, int]] = {
    "background": (0, 0, 333, 333),
    "slot1": (333, 0, 667, 333),
    "slot2": (667, 0, 1000, 333),
    "slot3": (333, 333, 667, 667),
    "slot4": (667, 333, 1000, 667),
}

# Backup panels (only saved to disk when enabled) – coordinates in grid-space
BACKUP_PANELS: dict[str, tuple[int, int, int, int]] = {
    "backup_A": (0, 333, 333, 667),
    "backup_B": (0, 667, 333, 1000),
    "backup_bg": (333, 667, 667, 1000),
    "backup_wide": (667, 667, 1000, 1000),
}


class MSRContactSheetCropper:
    """ComfyUI node that crops a 3x3 contact sheet into Licon-MSR panels.

    Inputs:
        contact_sheet: IMAGE tensor of shape (B, H, W, C) in [0, 1] range.
        grid_size: The reference grid dimension (default 1000 for Ideogram).
        filename_prefix: Prefix for saved files.
        save_to_disk: Whether to persist crops to disk.
        save_backup_panels: Also save the 4 backup panels.
        output_folder: Directory for saved files.

    Outputs:
        Five IMAGE tensors: background, slot1, slot2, slot3, slot4.
    """

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "contact_sheet": ("IMAGE",),
            },
            "optional": {
                "grid_size": ("INT", {"default": DEFAULT_GRID_SIZE, "min": 1, "max": 10000, "step": 1}),
                "filename_prefix": ("STRING", {"default": "msr"}),
                "save_to_disk": ("BOOLEAN", {"default": True}),
                "save_backup_panels": ("BOOLEAN", {"default": False}),
                "output_folder": ("STRING", {"default": "msr_crops"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("background", "slot1", "slot2", "slot3", "slot4")
    FUNCTION = "crop_panels"
    CATEGORY = "Licon-MSR / Utils"

    # --- Public API ----------------------------------------------------------

    def crop_panels(
        self,
        contact_sheet: torch.Tensor,
        grid_size: int = DEFAULT_GRID_SIZE,
        filename_prefix: str = "msr",
        save_to_disk: bool = True,
        save_backup_panels: bool = False,
        output_folder: str = "msr_crops",
    ) -> tuple[torch.Tensor, ...]:
        """Crop the contact sheet into panels and optionally save to disk.

        Args:
            contact_sheet: Input image tensor (B, H, W, C) in [0, 1].
            grid_size: Reference grid dimension for coordinate scaling.
            filename_prefix: Prefix for saved filenames.
            save_to_disk: Whether to write crops to disk.
            save_backup_panels: Whether to also process backup panels.
            output_folder: Output directory for saved files.

        Returns:
            Tuple of 5 IMAGE tensors (background, slot1, slot2, slot3, slot4).
        """
        logger.info("MSR Contact Sheet Cropper started")
        logger.debug("Input tensor shape: %s", contact_sheet.shape)

        try:
            pil_img = self._tensor_to_pil(contact_sheet)
            width, height = pil_img.size
            logger.info("Contact sheet size: %dx%d", width, height)

            self._validate_dimensions(width, height, grid_size)

            scale_x = width / float(grid_size)
            scale_y = height / float(grid_size)

            cropped_outputs: list[torch.Tensor] = []

            for name, bbox in MAIN_PANELS.items():
                crop_tensor, crop_pil = self._crop_single_panel(
                    pil_img, bbox, scale_x, scale_y
                )
                cropped_outputs.append(crop_tensor)

                if save_to_disk:
                    self._save_panel(crop_pil, output_folder, filename_prefix, name)

            if save_backup_panels:
                logger.info("Processing backup panels")
                for name, bbox in BACKUP_PANELS.items():
                    _, crop_pil = self._crop_single_panel(
                        pil_img, bbox, scale_x, scale_y
                    )
                    if save_to_disk:
                        self._save_panel(crop_pil, output_folder, filename_prefix, name)

            logger.info("All panels processed successfully")
            return tuple(cropped_outputs)

        except (ValueError, RuntimeError, OSError) as exc:
            logger.exception("Failed to crop contact sheet: %s", exc)
            panel_size = max(grid_size // 3, 1)
            empty = torch.zeros((1, panel_size, panel_size, 3), dtype=torch.float32)
            return (empty, empty, empty, empty, empty)

    # --- Private helpers -----------------------------------------------------

    @staticmethod
    def _tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
        """Convert a ComfyUI IMAGE tensor (B, H, W, C) to a PIL Image."""
        if tensor.ndim != 4:
            raise ValueError(
                f"Expected 4D tensor (B, H, W, C), got shape {tensor.shape}"
            )
        img_tensor = tensor[0]
        img_np = (img_tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        return Image.fromarray(img_np).convert("RGB")

    @staticmethod
    def _validate_dimensions(width: int, height: int, grid_size: int) -> None:
        """Validate image dimensions and warn on unexpected aspect ratios."""
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid image dimensions: {width}x{height}")

        if grid_size <= 0:
            raise ValueError(f"grid_size must be positive, got {grid_size}")

        aspect = width / float(height)
        if not 0.9 <= aspect <= 1.1:
            logger.warning(
                "Input aspect ratio %.2f deviates from square; "
                "panels may not align correctly",
                aspect,
            )

    @staticmethod
    def _crop_single_panel(
        pil_img: Image.Image,
        bbox: tuple[int, int, int, int],
        scale_x: float,
        scale_y: float,
    ) -> tuple[torch.Tensor, Image.Image]:
        """Crop a single panel from the image and return (tensor, PIL).

        Args:
            pil_img: Source PIL image.
            bbox: (left, top, right, bottom) in grid coordinates.
            scale_x: Horizontal scale factor.
            scale_y: Vertical scale factor.

        Returns:
            Tuple of (tensor crop, PIL crop).
        """
        left = int(bbox[0] * scale_x)
        top = int(bbox[1] * scale_y)
        right = int(bbox[2] * scale_x)
        bottom = int(bbox[3] * scale_y)

        logger.debug("Cropping panel: (%d, %d) -> (%d, %d)", left, top, right, bottom)
        cropped_pil = pil_img.crop((left, top, right, bottom))

        crop_np = np.array(cropped_pil).astype(np.float32) / 255.0
        crop_tensor = torch.from_numpy(crop_np).unsqueeze(0)
        return crop_tensor, cropped_pil

    @staticmethod
    def _save_panel(
        pil_img: Image.Image,
        output_folder: str,
        filename_prefix: str,
        name: str,
    ) -> None:
        """Save a PIL image to disk under output_folder."""
        os.makedirs(output_folder, exist_ok=True)
        filename = f"{filename_prefix}_{name}.png"
        save_path = os.path.join(output_folder, filename)
        pil_img.save(save_path)
        logger.debug("Saved: %s", save_path)


# --- Node registration -------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "MSRContactSheetCropper": MSRContactSheetCropper
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MSRContactSheetCropper": "MSR Contact Sheet Cropper"
}

logger.info("Licon-MSR Utils: MSRContactSheetCropper loaded successfully")