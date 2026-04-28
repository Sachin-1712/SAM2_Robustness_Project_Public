"""
sam_eval_utils.py — Shared Utilities for SAM2 Evaluation Scripts

This module provides reusable helper functions for loading SAM2 mask
outputs, constructing dense label maps, computing binary foreground
metrics, and parsing CLI arguments.

These utilities are used by:
    - compute_consistency.py   (primary clean-vs-corrupted evaluation)
    - compute_iou.py           (secondary GT-based evaluation)
    - analyze_consistency.py   (plotting / summary generation)

Design notes:
    - Dense label maps use the same area-sorted "fill unassigned pixels"
      logic as the qualitative overlays in run_sam2.py, ensuring that
      the quantitative foreground definition matches the visual output.
    - All mask comparison functions operate on numpy arrays and handle
      edge cases (empty masks, zero-area unions) gracefully.
    - Shape mismatches between clean and corrupted masks are resolved
      via nearest-neighbour interpolation, which preserves binary
      mask boundaries without introducing interpolation artefacts.

Author: Sachin (Final Year Project — SAM2 Robustness under Adverse Weather)
"""

import warnings
import numpy as np
import cv2
from pathlib import Path


# ──────────────────────────────────────────────────────────────────────
# Mask Loading
# ──────────────────────────────────────────────────────────────────────

def load_sam_masks(npz_path):
    """
    Safely load SAM2 mask dictionaries from a compressed .npz file.

    Each mask dictionary contains at least:
        - 'segmentation': bool ndarray (H, W)
        - 'area': int
        - 'bbox': list [x, y, w, h]

    Args:
        npz_path: Path to the .npz file (str or Path).

    Returns:
        list of mask dicts on success, or None on failure.

    Notes:
        - Uses allow_pickle=True because SAM2 stores dicts via np.savez.
        - Returns None (not raises) on corrupt/unreadable files so that
          callers can skip gracefully without crashing entire runs.
    """
    npz_path = Path(npz_path)
    if not npz_path.exists():
        warnings.warn(f"Mask file not found: {npz_path}")
        return None

    try:
        data = np.load(str(npz_path), allow_pickle=True)
        masks_raw = data["masks"]
        # Convert from numpy object array to Python list
        masks = list(masks_raw) if isinstance(masks_raw, np.ndarray) else masks_raw
        return masks
    except Exception as e:
        warnings.warn(f"Failed to load mask file {npz_path}: {e}")
        return None


# ──────────────────────────────────────────────────────────────────────
# Dense Label Map Construction
# ──────────────────────────────────────────────────────────────────────

def masks_to_dense_label_map(masks, shape=None):
    """
    Convert a list of SAM2 mask dictionaries into a dense label map.

    Masks are sorted by area (descending) and painted into unassigned
    pixels, producing a full-coverage label map where each pixel is
    assigned to at most one mask. This is the same logic used for the
    qualitative overlays in run_sam2.py.

    Args:
        masks: list of SAM2 mask dicts, each with 'segmentation' and 'area'.
        shape: Optional (H, W) tuple. If provided, masks are resized to
               this shape. If None, shape is inferred from the first mask.

    Returns:
        np.ndarray of shape (H, W), dtype int32.
        Pixel values: 0 = unassigned, 1..N = mask index (1-indexed).
        Returns all-zeros if masks is empty or None.
    """
    if not masks:
        if shape is not None:
            return np.zeros(shape, dtype=np.int32)
        return np.zeros((1, 1), dtype=np.int32)

    # Infer shape from first mask if not provided
    if shape is None:
        shape = masks[0]["segmentation"].shape[:2]

    # Sort by area descending — largest masks go first, smaller masks
    # can overwrite within their boundaries only on unassigned pixels
    sorted_masks = sorted(masks, key=lambda x: x.get("area", 0), reverse=True)

    label_map = np.zeros(shape, dtype=np.int32)
    for i, mask_dict in enumerate(sorted_masks, start=1):
        seg = mask_dict["segmentation"].astype(bool)

        # Resize if mask shape doesn't match target shape
        if seg.shape[:2] != shape:
            seg = cv2.resize(
                seg.astype(np.uint8), (shape[1], shape[0]),
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)

        # Fill only unassigned pixels
        unassigned = (label_map == 0) & seg
        label_map[unassigned] = i

    return label_map


# ──────────────────────────────────────────────────────────────────────
# Binary Foreground Conversion
# ──────────────────────────────────────────────────────────────────────

def dense_to_binary_foreground(label_map):
    """
    Convert a dense label map to a binary foreground mask.

    Args:
        label_map: np.ndarray (H, W) with 0 = background, >0 = foreground.

    Returns:
        np.ndarray (H, W) of dtype bool.
        True = any mask assigned, False = background.
    """
    return label_map > 0


def masks_to_binary_foreground(masks, shape=None):
    """
    Convenience: convert SAM2 masks directly to a binary foreground map.

    Equivalent to dense_to_binary_foreground(masks_to_dense_label_map(...)).

    Args:
        masks: list of SAM2 mask dicts.
        shape: Optional (H, W) target shape.

    Returns:
        np.ndarray (H, W) of dtype bool.
    """
    label_map = masks_to_dense_label_map(masks, shape=shape)
    return dense_to_binary_foreground(label_map)


# ──────────────────────────────────────────────────────────────────────
# Mask Resizing
# ──────────────────────────────────────────────────────────────────────

def resize_mask_nearest(mask, target_shape):
    """
    Resize a binary or integer mask using nearest-neighbour interpolation.

    Nearest-neighbour is critical for masks because bilinear/bicubic
    interpolation would create non-binary intermediate values and blur
    mask boundaries.

    Args:
        mask: np.ndarray (H, W) — bool or int.
        target_shape: (H, W) tuple for the output.

    Returns:
        np.ndarray of the same dtype, resized to target_shape.
    """
    if mask.shape[:2] == target_shape:
        return mask

    original_dtype = mask.dtype
    resized = cv2.resize(
        mask.astype(np.uint8),
        (target_shape[1], target_shape[0]),  # cv2 uses (W, H)
        interpolation=cv2.INTER_NEAREST,
    )
    return resized.astype(original_dtype)


# ──────────────────────────────────────────────────────────────────────
# Binary Metrics
# ──────────────────────────────────────────────────────────────────────

def compute_binary_iou(mask_a, mask_b):
    """
    Compute Intersection-over-Union between two binary masks.

    Args:
        mask_a, mask_b: np.ndarray (H, W) — bool or 0/1 int.
            Must have the same shape.

    Returns:
        float in [0.0, 1.0]. Returns 0.0 if both masks are empty.
    """
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)

    intersection = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()

    if union == 0:
        return 0.0
    return float(intersection) / float(union)


def compute_binary_dice(mask_a, mask_b):
    """
    Compute Dice coefficient (F1 score) between two binary masks.

    Dice = 2 * |A ∩ B| / (|A| + |B|)

    Args:
        mask_a, mask_b: np.ndarray (H, W) — bool or 0/1 int.
            Must have the same shape.

    Returns:
        float in [0.0, 1.0]. Returns 0.0 if both masks are empty.
    """
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)

    intersection = np.logical_and(a, b).sum()
    total = a.sum() + b.sum()

    if total == 0:
        return 0.0
    return float(2 * intersection) / float(total)


# ──────────────────────────────────────────────────────────────────────
# Best-Match Mask IoU
# ──────────────────────────────────────────────────────────────────────

def compute_best_match_mask_iou(clean_masks, corrupt_masks, target_shape=None):
    """
    Compute the mean best-match IoU between individual clean and corrupt masks.

    For each clean mask, find the corrupted mask with the highest IoU
    and record that IoU. Return the mean of these best IoUs across all
    clean masks.

    This metric captures how well individual object segments are
    preserved under corruption, independent of mask ordering.

    Args:
        clean_masks: list of SAM2 mask dicts (clean image).
        corrupt_masks: list of SAM2 mask dicts (corrupted image).
        target_shape: Optional (H, W). If provided, all masks are resized
                      to this shape before comparison.

    Returns:
        float in [0.0, 1.0], or 0.0 if there are no valid clean masks.
    """
    if not clean_masks or not corrupt_masks:
        return 0.0

    def _get_binary(mask_dict, shape):
        seg = mask_dict["segmentation"].astype(bool)
        if shape is not None and seg.shape[:2] != shape:
            seg = resize_mask_nearest(seg, shape)
        return seg

    # Infer target shape from first clean mask if not specified
    if target_shape is None:
        target_shape = clean_masks[0]["segmentation"].shape[:2]

    # Pre-compute corrupted binary masks for efficiency
    corrupt_binary = [_get_binary(m, target_shape) for m in corrupt_masks]

    best_ious = []
    for clean_mask_dict in clean_masks:
        clean_seg = _get_binary(clean_mask_dict, target_shape)

        # Skip zero-area clean masks
        if clean_seg.sum() == 0:
            continue

        best_iou = 0.0
        for corr_seg in corrupt_binary:
            iou = compute_binary_iou(clean_seg, corr_seg)
            if iou > best_iou:
                best_iou = iou
        best_ious.append(best_iou)

    if not best_ious:
        return 0.0
    return float(np.mean(best_ious))


# ──────────────────────────────────────────────────────────────────────
# CLI Parsing Helpers
# ──────────────────────────────────────────────────────────────────────

def parse_conditions(conditions_str):
    """
    Parse a comma-separated string of weather conditions.

    Args:
        conditions_str: e.g. "rain,fog,frost,snow,brightness"

    Returns:
        list of stripped, non-empty strings.
    """
    return [c.strip() for c in conditions_str.split(",") if c.strip()]


def parse_levels(levels_str):
    """
    Parse a comma-separated string of severity levels.

    Args:
        levels_str: e.g. "1,3,5"

    Returns:
        Sorted list of integers.
    """
    levels = [int(x.strip()) for x in levels_str.split(",") if x.strip()]
    return sorted(levels)
