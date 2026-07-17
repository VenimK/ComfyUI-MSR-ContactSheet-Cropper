"""Image conversion, cropping, mask generation, and debug overlay utilities."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from msr_nodes.constants import OVERLAY_COLORS

if TYPE_CHECKING:
    from msr_nodes.types import PanelCoords

logger = logging.getLogger(__name__)


def tensor_to_pil(img_tensor: torch.Tensor) -> Image.Image:
    """Convert a single (B, H, W, C) or (H, W, C) tensor to RGB PIL Image."""
    if img_tensor.ndim == 4:
        img_tensor = img_tensor[0]
    img_np = (img_tensor.detach().cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(img_np).convert("RGB")


def pil_to_tensor(pil_img: Image.Image) -> torch.Tensor:
    """Convert a PIL Image to (1, H, W, 3) float tensor."""
    img_np = np.array(pil_img).astype(np.float32) / 255.0
    return torch.from_numpy(img_np).unsqueeze(0)


def crop_tensor(
    img_tensor: torch.Tensor,
    top: int,
    left: int,
    bottom: int,
    right: int,
) -> torch.Tensor:
    """Crop a (B, H, W, C) tensor to the given pixel box."""
    h, w = img_tensor.shape[1], img_tensor.shape[2]
    top = max(0, min(top, h))
    bottom = max(top, min(bottom, h))
    left = max(0, min(left, w))
    right = max(left, min(right, w))
    return img_tensor[:, top:bottom, left:right, :]


def make_mask(height: int, width: int, batch: int = 1) -> torch.Tensor:
    """Return a white mask tensor of shape (B, H, W)."""
    return torch.ones((batch, height, width), dtype=torch.float32)


def get_overlay_font(size: int = 16) -> ImageFont.FreeTypeFont | None:
    """Return a readable TrueType font if available, otherwise None."""
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "arial.ttf",
        "DejaVuSans-Bold.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return None


def draw_debug_overlay(
    pil_img: Image.Image,
    panel_coords: dict[str, PanelCoords],
    grid_cols: int = 3,
    grid_rows: int = 3,
) -> Image.Image:
    """Draw a polished debug overlay with grid lines and labeled panels."""
    width, height = pil_img.size
    overlay = pil_img.copy()
    draw = ImageDraw.Draw(overlay, "RGBA")

    # Full grid lines
    grid_color = (255, 255, 255, 80)
    for col in range(1, grid_cols):
        x = int(col * width / grid_cols)
        draw.line([(x, 0), (x, height)], fill=grid_color, width=2)
    for row in range(1, grid_rows):
        y = int(row * height / grid_rows)
        draw.line([(0, y), (width, y)], fill=grid_color, width=2)

    # Panel rectangles with semi-transparent fill
    for i, (name, coords) in enumerate(panel_coords.items()):
        color = OVERLAY_COLORS[i % len(OVERLAY_COLORS)]
        fill_color = color + (40,)
        rect = [coords.left, coords.top, coords.right - 1, coords.bottom - 1]
        draw.rectangle(rect, fill=fill_color)
        draw.rectangle(rect, outline=color, width=4)

        label = name.replace("_", " ")
        font = get_overlay_font()
        text_pos = (coords.left + 6, coords.top + 6)
        bbox = draw.textbbox(text_pos, label, font=font) if font else draw.textbbox(text_pos, label)
        draw.rectangle(
            [bbox[0] - 3, bbox[1] - 3, bbox[2] + 3, bbox[3] + 3],
            fill=color + (220,),
        )
        draw.text(text_pos, label, fill=(0, 0, 0), font=font)

    return overlay.convert("RGB")
