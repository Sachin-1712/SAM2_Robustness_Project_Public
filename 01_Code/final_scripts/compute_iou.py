"""
compute_iou.py — SECONDARY / APPENDIX: Ground-Truth-Based IoU Evaluation

IMPORTANT: This script is retained for supplementary analysis only.
The PRIMARY robustness evaluation uses clean-vs-corrupted foreground
consistency (see compute_consistency.py).

This script compares SAM2 automatic masks against Cityscapes ground-truth
semantic labels. For each GT object, it finds the SAM2 mask with the
highest IoU and reports the mean best-match IoU per image.

This approach requires GT-guided prompting assumptions and is NOT the
main evaluation methodology for the dissertation. It is included as
supplementary material in the appendix.

Usage:
    python scripts/compute_iou.py \\
        --gt_root gt \\
        --mask_root masks \\
        --output_csv results/gt_metrics.csv \\
        --conditions rain,fog,frost,snow,brightness \\
        --levels 1,3,5

Author: Sachin (Final Year Project — SAM2 Robustness under Adverse Weather)
"""

import os
import cv2
import numpy as np
import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm


def compute_binary_iou(mask1, mask2):
    """
    Compute IoU between two binary masks.

    Returns 0.0 if the union is empty.
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0.0
    return intersection / union


def evaluate_image(gt_path, sam_path):
    """
    Compare GT semantic mask against SAM2 automatic masks.

    For each unique GT object ID (excluding background=0 and void=255),
    find the SAM2 mask with the highest IoU. Returns a list of best-match
    IoU values (one per GT object).
    """
    gt_mask = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
    if gt_mask is None:
        print(f"  ⚠  Could not read GT: {gt_path}")
        return []

    try:
        sam_data = np.load(sam_path, allow_pickle=True)
        sam_masks_dicts = sam_data['masks']
    except Exception as e:
        print(f"  ⚠  Could not load SAM masks {sam_path}: {e}")
        return []

    if len(sam_masks_dicts) == 0:
        return []

    # Extract unique GT object IDs, excluding background (0) and void (255)
    obj_ids = np.unique(gt_mask)
    obj_ids = [val for val in obj_ids if val not in (0, 255)]

    ious = []

    for obj_id in obj_ids:
        gt_obj_mask = (gt_mask == obj_id)
        best_iou = 0.0

        for sam_ann in sam_masks_dicts:
            sam_mask = sam_ann['segmentation']
            # Resize SAM mask to GT dimensions if they differ
            if sam_mask.shape != gt_obj_mask.shape:
                sam_mask = cv2.resize(
                    sam_mask.astype(np.uint8),
                    (gt_obj_mask.shape[1], gt_obj_mask.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            iou = compute_binary_iou(gt_obj_mask, sam_mask)
            if iou > best_iou:
                best_iou = iou

        ious.append(best_iou)

    return ious


def main():
    parser = argparse.ArgumentParser(
        description="[SECONDARY/APPENDIX] Compute GT-based IoU metrics for SAM2. "
                    "For the primary evaluation, use compute_consistency.py instead.",
    )
    parser.add_argument("--gt_root", type=str, default="gt",
                        help="Ground truth masks directory.")
    parser.add_argument("--mask_root", type=str, default="masks",
                        help="Root mask output directory.")
    parser.add_argument("--output_csv", type=str, default="results/gt_metrics.csv",
                        help="Output CSV file.")
    parser.add_argument("--conditions", type=str, default="rain,fog,frost,snow,brightness",
                        help="Comma-separated weather conditions.")
    parser.add_argument("--levels", type=str, default="1,3,5",
                        help="Comma-separated severity levels. Default: 1,3,5.")

    args = parser.parse_args()

    # Parse conditions and levels from CLI
    weather_types = [c.strip() for c in args.conditions.split(",") if c.strip()]
    levels = [int(x.strip()) for x in args.levels.split(",") if x.strip()]

    # Build evaluation pairs: clean first, then each (condition, level)
    conditions = [('clean', None)]
    for w in weather_types:
        for l in levels:
            conditions.append((w, f"level_{l}"))

    results = []

    for condition, severity in conditions:
        if condition == 'clean':
            sam_dir = Path(args.mask_root) / "clean"
            cond_str = "clean"
        else:
            sam_dir = Path(args.mask_root) / condition / severity
            cond_str = f"{condition}_{severity}"
            
        if not sam_dir.exists():
            continue # Skip missing directories
            
        print(f"Evaluating {cond_str}...")
        
        sam_files = list(sam_dir.glob("*.npz"))
        
        for sam_file in tqdm(sam_files):
            base = sam_file.stem.replace("_leftImg8bit", "")
            gt_path = Path(args.gt_root) / f"{base}_gtFine_labelIds.png"
            
            if not gt_path.exists():
                continue # Skip if no GT
            
            ious = evaluate_image(gt_path, sam_file)
            
            if ious:
                mean_iou = np.mean(ious)
                results.append({
                    'image': sam_file.stem,
                    'condition': condition,
                    'severity': severity if severity else 'none',
                    'mIoU': mean_iou
                })

    if results:
        df = pd.DataFrame(results)
        df.to_csv(args.output_csv, index=False)
        
        summary = df.groupby(['condition', 'severity'])['mIoU'].mean().reset_index()
        summary_path = Path(args.output_csv).parent / "summary_metrics.csv"
        summary.to_csv(summary_path, index=False)
        print("Summary metrics:")
        print(summary)
    else:
        print("No results computed.")

if __name__ == "__main__":
    main()
