"""
compute_consistency.py — Clean-vs-Corrupted Foreground Consistency Evaluation

PRIMARY evaluation script for the SAM2 robustness study.

This script measures how consistently SAM2 segments the same scene when
the input is degraded by simulated adverse weather. For each image, it
compares the SAM2 output on the clean version against each corrupted
version and reports several overlap metrics.

Methodology:
    1. Load raw SAM2 masks (.npz) for both clean and corrupted images.
    2. Build dense label maps using area-sorted fill (matching the
       qualitative overlay logic in run_sam2.py).
    3. Convert to binary foreground maps (any mask vs. background).
    4. If spatial dimensions differ, resize the corrupted map to the
       clean map dimensions using nearest-neighbour interpolation.
    5. Compute metrics:
       - Foreground IoU          (PRIMARY)
       - Foreground Dice
       - Clean foreground preserved ratio
       - Best-match mask IoU     (per-mask instance metric)
       - Mask count statistics

IMPORTANT: This does NOT compare raw mask label IDs, because SAM2 mask
ordering is arbitrary. Foreground consistency and best-match IoU are
both invariant to mask index assignment.

Outputs:
    - Per-image CSV:  one row per (image, condition, severity)
    - Summary CSV:    mean ± std grouped by (condition, severity)

Usage:
    python scripts/compute_consistency.py \\
        --mask_root runs/run3_diffusion_eval/masks \\
        --output_csv runs/run3_diffusion_eval/results/consistency_metrics_5img.csv \\
        --conditions rain,fog,frost,snow,brightness \\
        --levels 1,3,5

Author: Sachin (Final Year Project — SAM2 Robustness under Adverse Weather)
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from sam_eval_utils import (
    load_sam_masks,
    masks_to_dense_label_map,
    dense_to_binary_foreground,
    resize_mask_nearest,
    compute_binary_iou,
    compute_binary_dice,
    compute_best_match_mask_iou,
    parse_conditions,
    parse_levels,
)


# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────

CSV_COLUMNS = [
    "image",
    "condition",
    "severity",
    "clean_mask_path",
    "corrupt_mask_path",
    "clean_height",
    "clean_width",
    "corrupt_height",
    "corrupt_width",
    "resized_corrupt_to_clean",
    "clean_mask_count",
    "corrupt_mask_count",
    "foreground_iou",
    "foreground_dice",
    "clean_foreground_pixels",
    "corrupt_foreground_pixels",
    "intersection_pixels",
    "union_pixels",
    "clean_foreground_preserved_ratio",
    "best_match_mask_iou_mean",
    "mask_count_difference",
]


# ──────────────────────────────────────────────────────────────────────
# Single Pair Evaluation
# ──────────────────────────────────────────────────────────────────────

def evaluate_pair(clean_npz, corrupt_npz, condition, severity, verbose=False):
    """
    Evaluate foreground consistency between one clean and one corrupted mask file.

    Args:
        clean_npz: Path to the clean mask .npz file.
        corrupt_npz: Path to the corrupted mask .npz file.
        condition: str, e.g. "rain".
        severity: int, e.g. 5.
        verbose: bool, print progress info.

    Returns:
        dict with all CSV_COLUMNS populated, or None if files are unreadable.
    """
    clean_npz = Path(clean_npz)
    corrupt_npz = Path(corrupt_npz)

    # ── Load masks ────────────────────────────────────────────────
    clean_masks = load_sam_masks(clean_npz)
    if clean_masks is None:
        print(f"  ⚠  Skipping {clean_npz.stem}: clean masks unreadable")
        return None

    corrupt_masks = load_sam_masks(corrupt_npz)
    if corrupt_masks is None:
        print(f"  ⚠  Skipping {corrupt_npz.stem}: corrupt masks unreadable "
              f"({condition} L{severity})")
        return None

    # ── Build dense label maps ────────────────────────────────────
    clean_label = masks_to_dense_label_map(clean_masks)
    corrupt_label = masks_to_dense_label_map(corrupt_masks)

    clean_h, clean_w = clean_label.shape
    corrupt_h, corrupt_w = corrupt_label.shape

    # ── Convert to binary foreground ──────────────────────────────
    clean_fg = dense_to_binary_foreground(clean_label)
    corrupt_fg = dense_to_binary_foreground(corrupt_label)

    # ── Handle shape mismatch ─────────────────────────────────────
    resized = False
    if clean_fg.shape != corrupt_fg.shape:
        resized = True
        if verbose:
            print(f"    Resizing corrupt mask from {corrupt_fg.shape} → "
                  f"{clean_fg.shape} (nearest-neighbour)")
        corrupt_fg = resize_mask_nearest(corrupt_fg, clean_fg.shape)

    # ── Compute primary metrics ───────────────────────────────────
    fg_iou = compute_binary_iou(clean_fg, corrupt_fg)
    fg_dice = compute_binary_dice(clean_fg, corrupt_fg)

    clean_fg_pixels = int(clean_fg.sum())
    corrupt_fg_pixels = int(corrupt_fg.sum())
    intersection = int(np.logical_and(clean_fg, corrupt_fg).sum())
    union = int(np.logical_or(clean_fg, corrupt_fg).sum())

    # Coverage ratio: fraction of clean foreground preserved in corrupted
    if clean_fg_pixels > 0:
        preserved_ratio = float(intersection) / float(clean_fg_pixels)
    else:
        preserved_ratio = 1.0  # No clean foreground → trivially preserved

    # ── Best-match mask IoU ───────────────────────────────────────
    target_shape = (clean_h, clean_w)
    best_match_iou = compute_best_match_mask_iou(
        clean_masks, corrupt_masks, target_shape=target_shape
    )

    # ── Mask counts ───────────────────────────────────────────────
    clean_count = len(clean_masks)
    corrupt_count = len(corrupt_masks)
    count_diff = corrupt_count - clean_count

    # ── Build result row ──────────────────────────────────────────
    row = {
        "image": clean_npz.stem,
        "condition": condition,
        "severity": severity,
        "clean_mask_path": str(clean_npz),
        "corrupt_mask_path": str(corrupt_npz),
        "clean_height": clean_h,
        "clean_width": clean_w,
        "corrupt_height": corrupt_h,
        "corrupt_width": corrupt_w,
        "resized_corrupt_to_clean": resized,
        "clean_mask_count": clean_count,
        "corrupt_mask_count": corrupt_count,
        "foreground_iou": round(fg_iou, 6),
        "foreground_dice": round(fg_dice, 6),
        "clean_foreground_pixels": clean_fg_pixels,
        "corrupt_foreground_pixels": corrupt_fg_pixels,
        "intersection_pixels": intersection,
        "union_pixels": union,
        "clean_foreground_preserved_ratio": round(preserved_ratio, 6),
        "best_match_mask_iou_mean": round(best_match_iou, 6),
        "mask_count_difference": count_diff,
    }

    if verbose:
        print(f"    {clean_npz.stem} × {condition}/L{severity}  "
              f"IoU={fg_iou:.4f}  Dice={fg_dice:.4f}  "
              f"masks: {clean_count}→{corrupt_count}")

    return row


# ──────────────────────────────────────────────────────────────────────
# Summary Computation
# ──────────────────────────────────────────────────────────────────────

def compute_summary(rows):
    """
    Group per-image rows by (condition, severity) and compute mean ± std
    for key numeric metrics.

    Args:
        rows: list of dicts (per-image CSV rows).

    Returns:
        list of summary dicts, sorted by (condition, severity).
    """
    from collections import defaultdict

    METRIC_KEYS = [
        "foreground_iou",
        "foreground_dice",
        "clean_foreground_preserved_ratio",
        "best_match_mask_iou_mean",
        "clean_mask_count",
        "corrupt_mask_count",
        "mask_count_difference",
    ]

    groups = defaultdict(list)
    for row in rows:
        key = (row["condition"], int(row["severity"]))
        groups[key].append(row)

    summary_rows = []
    for (condition, severity), group_rows in sorted(groups.items()):
        summary = {
            "condition": condition,
            "severity": severity,
            "n_images": len(group_rows),
        }
        for metric in METRIC_KEYS:
            values = [float(r[metric]) for r in group_rows if r[metric] is not None]
            if values:
                mean_val = float(np.mean(values))
                std_val = float(np.std(values, ddof=0))  # population std for small N
                summary[f"{metric}_mean"] = round(mean_val, 6)
                summary[f"{metric}_std"] = round(std_val, 6)
            else:
                summary[f"{metric}_mean"] = None
                summary[f"{metric}_std"] = None
        summary_rows.append(summary)

    return summary_rows


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute clean-vs-corrupted foreground consistency metrics "
            "for SAM2 robustness evaluation. This is the PRIMARY evaluation "
            "script for the dissertation."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python scripts/compute_consistency.py \\
      --mask_root runs/run3_diffusion_eval/masks \\
      --output_csv runs/run3_diffusion_eval/results/consistency_metrics_5img.csv \\
      --conditions rain,fog,frost,snow,brightness \\
      --levels 1,3,5

  # With verbose output and missing-file tolerance
  python scripts/compute_consistency.py \\
      --mask_root runs/run3_diffusion_eval/masks \\
      --output_csv results/consistency_5img.csv \\
      --allow_missing --verbose
""",
    )

    parser.add_argument(
        "--mask_root", type=str, required=True,
        help="Root directory containing SAM2 mask .npz files. "
             "Expected structure: mask_root/clean/*.npz and "
             "mask_root/<condition>/level_<N>/*.npz.",
    )
    parser.add_argument(
        "--output_csv", type=str, required=True,
        help="Path for the per-image consistency metrics CSV.",
    )
    parser.add_argument(
        "--summary_csv", type=str, default=None,
        help="Path for the grouped summary CSV. If omitted, written "
             "beside output_csv as 'summary_<basename>'.",
    )
    parser.add_argument(
        "--conditions", type=str, default="rain,fog,frost,snow,brightness",
        help="Comma-separated weather conditions to evaluate.",
    )
    parser.add_argument(
        "--levels", type=str, default="1,3,5",
        help="Comma-separated severity levels. Default: 1,3,5.",
    )
    parser.add_argument(
        "--only_images", type=str, default=None,
        help="Comma-separated image filenames or stems. Only these images "
             "will be evaluated.",
    )
    parser.add_argument(
        "--allow_missing", action="store_true",
        help="Skip missing corrupted mask files instead of reporting errors.",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print per-image metric summaries during evaluation.",
    )

    args = parser.parse_args()

    # ── Parse arguments ───────────────────────────────────────────
    mask_root = Path(args.mask_root)
    output_csv = Path(args.output_csv)
    conditions = parse_conditions(args.conditions)
    levels = parse_levels(args.levels)
    only_images = None
    if args.only_images:
        only_images = {
            Path(s.strip()).stem for s in args.only_images.split(",") if s.strip()
        }

    # Derive summary CSV path
    if args.summary_csv:
        summary_csv = Path(args.summary_csv)
    else:
        summary_csv = output_csv.parent / f"summary_{output_csv.name}"

    # ── Validate mask root ────────────────────────────────────────
    clean_dir = mask_root / "clean"
    if not clean_dir.exists():
        print(f"ERROR: Clean mask directory not found: {clean_dir}")
        sys.exit(1)

    # ── Discover clean mask files ─────────────────────────────────
    clean_files = sorted(clean_dir.glob("*.npz"))
    if only_images:
        clean_files = [f for f in clean_files if f.stem in only_images]

    if not clean_files:
        print(f"ERROR: No clean mask files found in {clean_dir}")
        sys.exit(1)

    # ── Startup banner ────────────────────────────────────────────
    print()
    print("=" * 64)
    print("  SAM2 Foreground Consistency Evaluation (PRIMARY)")
    print("=" * 64)
    print(f"  Mask root:     {mask_root}")
    print(f"  Clean images:  {len(clean_files)}")
    print(f"  Conditions:    {conditions}")
    print(f"  Levels:        {levels}")
    print(f"  Output CSV:    {output_csv}")
    print(f"  Summary CSV:   {summary_csv}")
    print(f"  Allow missing: {args.allow_missing}")
    print("=" * 64)
    print()

    # ── Evaluate all pairs ────────────────────────────────────────
    all_rows = []
    n_skipped = 0
    n_missing = 0

    for condition in conditions:
        for level in levels:
            corrupt_dir = mask_root / condition / f"level_{level}"

            if not corrupt_dir.exists():
                if args.allow_missing:
                    print(f"  ⚠  Missing directory: {corrupt_dir} — skipping")
                    n_missing += 1
                    continue
                else:
                    print(f"  ERROR: Missing directory: {corrupt_dir}")
                    print("  Use --allow_missing to skip missing conditions.")
                    sys.exit(1)

            print(f"  {condition.upper()} L{level}:")

            for clean_npz in clean_files:
                corrupt_npz = corrupt_dir / clean_npz.name

                if not corrupt_npz.exists():
                    if args.allow_missing:
                        if args.verbose:
                            print(f"    ⚠  Missing: {corrupt_npz.name}")
                        n_missing += 1
                        continue
                    else:
                        print(f"  ERROR: Expected {corrupt_npz} but not found.")
                        print("  Use --allow_missing to skip missing files.")
                        sys.exit(1)

                row = evaluate_pair(
                    clean_npz, corrupt_npz,
                    condition, level,
                    verbose=args.verbose,
                )
                if row is not None:
                    all_rows.append(row)
                else:
                    n_skipped += 1

    # ── Write per-image CSV ───────────────────────────────────────
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\n  ✓ Per-image CSV written: {output_csv} ({len(all_rows)} rows)")

    # ── Compute and write summary ─────────────────────────────────
    if all_rows:
        summary_rows = compute_summary(all_rows)
        summary_keys = list(summary_rows[0].keys()) if summary_rows else []

        summary_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=summary_keys)
            writer.writeheader()
            writer.writerows(summary_rows)

        print(f"  ✓ Summary CSV written:  {summary_csv} ({len(summary_rows)} rows)")

        # Print summary table to console
        print()
        print("  ┌─────────────────────────────────────────────────────────┐")
        print("  │  CONSISTENCY SUMMARY (mean foreground IoU)             │")
        print("  ├─────────────┬──────────┬───────────────┬───────────────┤")
        print("  │ Condition   │ Severity │ Fg IoU (mean) │ Fg IoU (std)  │")
        print("  ├─────────────┼──────────┼───────────────┼───────────────┤")
        for sr in summary_rows:
            cond = sr["condition"][:11].ljust(11)
            sev = str(sr["severity"]).center(8)
            mean_iou = f"{sr['foreground_iou_mean']:.4f}".center(13)
            std_iou = f"{sr['foreground_iou_std']:.4f}".center(13)
            print(f"  │ {cond} │ {sev} │ {mean_iou} │ {std_iou} │")
        print("  └─────────────┴──────────┴───────────────┴───────────────┘")
    else:
        print("\n  ⚠  No valid pairs evaluated!")

    # ── Final summary ─────────────────────────────────────────────
    print()
    print(f"  Pairs evaluated:   {len(all_rows)}")
    print(f"  Pairs skipped:     {n_skipped} (unreadable files)")
    print(f"  Missing files:     {n_missing}")
    print()
    print("✅ Consistency evaluation complete!")


if __name__ == "__main__":
    main()
