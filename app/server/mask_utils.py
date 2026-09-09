# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""Mask rendering utilities.

Converts SAM3's binary mask outputs (N, H, W) into colored PNG overlay images
that can be directly displayed on the frontend canvas via drawImage().
"""

import base64
import io
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# A palette of visually distinct colors for different objects.
# Triples are in BGR format for OpenCV compatibility; the color a mask
# actually displays on screen is the triple read as RGB (e.g. the first
# entry shows BLUE). The frontend (mask-overlay.js getObjectColor) mirrors
# these displayed colors for the sidebar swatches.
_COLOR_PALETTE = [
    (255, 0, 0),      # blue
    (0, 255, 0),      # green
    (0, 0, 255),      # red
    (255, 255, 0),    # cyan
    (255, 0, 255),    # magenta
    (0, 255, 255),    # yellow
    (128, 0, 0),      # navy
    (0, 128, 0),      # dark green
    (0, 0, 128),      # maroon
    (128, 128, 0),    # teal
    (128, 0, 128),    # purple
    (0, 128, 128),    # olive
    (255, 128, 0),    # sky blue
    (128, 255, 0),    # spring green
    (0, 128, 255),    # orange
    (255, 0, 128),    # violet
    (128, 255, 255),  # light yellow
    (255, 128, 128),  # light blue
    (128, 128, 255),  # light red
    (200, 200, 200),  # gray
]


def get_color_for_obj_id(obj_id: int) -> Tuple[int, int, int]:
    """Return a consistent BGR color for a given object ID."""
    return _COLOR_PALETTE[obj_id % len(_COLOR_PALETTE)]


def render_masks_overlay(
    frame_image: np.ndarray,
    obj_ids: np.ndarray,
    binary_masks: np.ndarray,
    alpha: float = 0.5,
) -> bytes:
    """Composite binary masks onto a frame image and return PNG bytes.

    Args:
        frame_image: Background frame as RGB numpy array (H, W, 3), uint8.
        obj_ids: Array of object IDs (N,).
        binary_masks: Array of binary masks (N, H, W), bool or uint8.
        alpha: Overlay transparency (0.0=invisible, 1.0=opaque).

    Returns:
        PNG image bytes (base64-encoded) with masks overlaid.
    """
    # Work on a copy to avoid mutating the original
    overlay = frame_image.copy()

    if len(obj_ids) == 0 or binary_masks.size == 0:
        # No masks, return the original frame as PNG
        is_success, buffer = cv2.imencode(".png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        if not is_success:
            raise RuntimeError("Failed to encode PNG")
        return base64.b64encode(buffer).decode("utf-8")

    # Create a colored mask layer
    h, w = overlay.shape[:2]
    mask_layer = np.zeros((h, w, 3), dtype=np.uint8)

    for i, obj_id in enumerate(obj_ids):
        if i >= len(binary_masks):
            break
        mask = binary_masks[i]
        if mask.shape != (h, w):
            # Resize mask to match frame resolution
            mask = cv2.resize(
                mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
            ).astype(bool)
        color = get_color_for_obj_id(int(obj_id))
        mask_layer[mask] = color

    # Blend the mask layer with the original frame
    result = cv2.addWeighted(overlay, 1.0, mask_layer, alpha, 0)

    # Draw mask contours for better visibility
    for i, obj_id in enumerate(obj_ids):
        if i >= len(binary_masks):
            break
        mask = binary_masks[i]
        if mask.shape != (h, w):
            mask = cv2.resize(
                mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
            ).astype(bool)
        color = get_color_for_obj_id(int(obj_id))
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(
            cv2.cvtColor(result, cv2.COLOR_RGB2BGR),
            contours, -1, color, 2,
        )

    # Encode as PNG
    is_success, buffer = cv2.imencode(".png", cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    if not is_success:
        raise RuntimeError("Failed to encode PNG")
    return base64.b64encode(buffer).decode("utf-8")


def render_masks_only(
    obj_ids: np.ndarray,
    binary_masks: np.ndarray,
    height: int,
    width: int,
    alpha: float = 0.5,
) -> bytes:
    """Render masks as a transparent PNG overlay (without background frame).

    This produces a smaller image that the frontend can composite on top of
    the video frame, reducing bandwidth.

    Args:
        obj_ids: Array of object IDs (N,).
        binary_masks: Array of binary masks (N, H, W).
        height: Output image height.
        width: Output image width.
        alpha: Overlay transparency.

    Returns:
        PNG image bytes (base64-encoded) with transparent background.
    """
    # Create RGBA image with transparent background
    overlay = np.zeros((height, width, 4), dtype=np.uint8)

    if len(obj_ids) == 0 or binary_masks.size == 0:
        is_success, buffer = cv2.imencode(".png", overlay)
        if not is_success:
            raise RuntimeError("Failed to encode PNG")
        return base64.b64encode(buffer).decode("utf-8")

    for i, obj_id in enumerate(obj_ids):
        if i >= len(binary_masks):
            break
        mask = binary_masks[i]
        if mask.shape != (height, width):
            mask = cv2.resize(
                mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST
            ).astype(bool)
        color = get_color_for_obj_id(int(obj_id))
        # BGR -> BGRA
        overlay[mask] = (*color, int(255 * alpha))

    is_success, buffer = cv2.imencode(".png", overlay)
    if not is_success:
        raise RuntimeError("Failed to encode PNG")
    return base64.b64encode(buffer).decode("utf-8")
