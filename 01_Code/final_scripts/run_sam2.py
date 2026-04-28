"""
run_sam2.py — SAM2 Automatic Mask Generation for Cityscapes Robustness Evaluation

This script runs SAM2's automatic mask generator on clean and weather-corrupted
Cityscapes images. It is designed for a Final Year Project evaluating SAM2
robustness under adverse weather conditions.

CRITICAL METHODOLOGY NOTE:
    For fair clean-vs-corrupted comparison, BOTH clean and corrupted images
    must use the SAME mask generator settings (profile). The only exception
    is an explicit --ablation_mode, which is clearly logged and must never
    be mistaken for the main experiment.

Profiles:
    balanced            — Default. Practical for dissertation main runs.
    high_quality        — Heavier but still reasonable.
    corrupt_permissive  — Ablation/exploratory ONLY. NOT for main results.

Usage examples:
    # Run all default conditions/levels with balanced profile
    python scripts/run_sam2.py --checkpoint checkpoints/sam2.1_hiera_large.pt

    # Run only one image
    python scripts/run_sam2.py --checkpoint checkpoints/sam2.1_hiera_large.pt \\
        --only_images frankfurt_000001_064925_leftImg8bit.png

    # Resume after interruption
    python scripts/run_sam2.py --checkpoint checkpoints/sam2.1_hiera_large.pt --resume

    # Overwrite everything
    python scripts/run_sam2.py --checkpoint checkpoints/sam2.1_hiera_large.pt --overwrite
"""

import os
import cv2
import torch
import numpy as np
import argparse
import hashlib
import json
import csv
import shutil
import time
import traceback
from pathlib import Path
from datetime import datetime, timezone
from tqdm import tqdm

# SAM2 imports
import sam2
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────

CITYSCAPES_SHAPE = (1024, 2048, 3)  # H, W, C
MANIFEST_FILENAME = "run_manifest.json"
IMAGE_LOG_CSV = "image_run_log.csv"
IMAGE_LOG_JSONL = "image_run_log.jsonl"

CSV_FIELDNAMES = [
    "timestamp", "weather_condition", "severity_level", "split_type",
    "input_filename", "input_path", "width", "height", "resolution_ok",
    "profile_used", "checkpoint_used", "config_used",
    "start_time", "end_time", "runtime_seconds",
    "status", "number_of_masks_generated",
    "mask_output_path", "viz_output_path",
]

# ──────────────────────────────────────────────────────────────────────
# Inference Profiles
# ──────────────────────────────────────────────────────────────────────

def get_profile_params(profile_name: str) -> dict:
    """
    Return SAM2 AMG parameters for a named inference profile.

    IMPORTANT: For fair robustness evaluation, the SAME profile must be used
    for both clean and corrupted images in the main experiment. Using different
    settings would confound weather effects with hyperparameter differences.

    Profiles:
        balanced            Practical for dissertation. ~10-15 min/image on
                            consumer GPU. Good mask quality without excessive
                            computation.
        high_quality        Denser point grid and multi-crop. Better coverage
                            but ~2-3x slower than balanced. Use when runtime
                            is not a constraint.
        corrupt_permissive  Very permissive thresholds designed to capture
                            weak masks in degraded images. ONLY for ablation
                            studies — produces more false positives and must
                            NOT be compared directly with balanced/HQ results
                            on clean images.
    """
    profiles = {
        "balanced": {
            "points_per_side": 32,
            "pred_iou_thresh": 0.86,
            "stability_score_thresh": 0.92,
            "crop_n_layers": 0,
            "crop_n_points_downscale_factor": 2,
            "min_mask_region_area": 100.0,
            "box_nms_thresh": 0.70,
        },
        "high_quality": {
            "points_per_side": 64,
            "pred_iou_thresh": 0.80,
            "stability_score_thresh": 0.90,
            "crop_n_layers": 1,
            "crop_n_points_downscale_factor": 2,
            "min_mask_region_area": 50.0,
            "box_nms_thresh": 0.70,
        },
        "corrupt_permissive": {
            "points_per_side": 96,
            "pred_iou_thresh": 0.65,
            "stability_score_thresh": 0.80,
            "crop_n_layers": 2,
            "crop_n_points_downscale_factor": 2,
            "min_mask_region_area": 25.0,
            "box_nms_thresh": 0.70,
        },
    }
    if profile_name not in profiles:
        raise ValueError(
            f"Unknown profile '{profile_name}'. "
            f"Valid profiles: {list(profiles.keys())}"
        )
    return profiles[profile_name]


def build_mask_generator_from_profile(sam2_model, profile_name: str):
    """Build a SAM2AutomaticMaskGenerator using a named profile."""
    params = get_profile_params(profile_name)
    return SAM2AutomaticMaskGenerator(model=sam2_model, **params)


# ──────────────────────────────────────────────────────────────────────
# Experiment Manifest & Signature
# ──────────────────────────────────────────────────────────────────────

def compute_experiment_signature(manifest_data: dict) -> str:
    """
    Compute a SHA-256 hash from the fields that define a run configuration.
    Used to detect whether a resumed run matches the original experiment.
    """
    # Fields that must match for a valid resume
    sig_fields = [
        "config_path", "checkpoint_path", "device",
        "profile", "generator_params",
        "conditions", "levels",
        "strict_resolution", "ablation_mode",
    ]
    sig_dict = {k: manifest_data.get(k) for k in sig_fields}
    sig_str = json.dumps(sig_dict, sort_keys=True, default=str)
    return hashlib.sha256(sig_str.encode("utf-8")).hexdigest()


def load_or_validate_manifest(
    mask_root: Path,
    current_manifest: dict,
    resume: bool,
    allow_mismatch: bool,
) -> dict:
    """
    Load an existing manifest and validate it against the current config,
    or create a new one.

    Returns the manifest dict (possibly loaded from disk).
    Raises SystemExit if resume is requested but manifest doesn't match.
    """
    manifest_path = mask_root / MANIFEST_FILENAME
    current_sig = compute_experiment_signature(current_manifest)
    current_manifest["experiment_signature"] = current_sig

    if manifest_path.exists():
        with open(manifest_path, "r") as f:
            saved = json.load(f)
        saved_sig = saved.get("experiment_signature", "")

        if resume and saved_sig and saved_sig != current_sig:
            if allow_mismatch:
                print(
                    "⚠  WARNING: Manifest mismatch detected but "
                    "--allow_manifest_mismatch is set. Proceeding with "
                    "current settings. Previous results may not be comparable."
                )
            else:
                print("=" * 70)
                print("ERROR: Experiment configuration mismatch!")
                print("=" * 70)
                print(f"  Saved signature:   {saved_sig[:16]}...")
                print(f"  Current signature: {current_sig[:16]}...")
                print()
                print("Existing outputs were produced with different settings.")
                print("Resuming would mix incompatible results.")
                print()
                print("Options:")
                print("  1. Use --overwrite to start fresh")
                print("  2. Use --allow_manifest_mismatch to force resume")
                print("  3. Use a different --mask_root for this config")
                print("=" * 70)
                raise SystemExit(1)

    # Write/update manifest
    mask_root.mkdir(parents=True, exist_ok=True)
    current_manifest["last_updated"] = datetime.now(timezone.utc).isoformat()
    with open(manifest_path, "w") as f:
        json.dump(current_manifest, f, indent=2, default=str)

    return current_manifest


# ──────────────────────────────────────────────────────────────────────
# Per-Image Logging
# ──────────────────────────────────────────────────────────────────────

def _ensure_csv_header(csv_path: Path):
    """Create CSV with header if it doesn't exist yet."""
    if not csv_path.exists():
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
            writer.writeheader()


def append_image_log(mask_root: Path, row: dict):
    """Append a single row to the per-image CSV and JSONL logs."""
    csv_path = mask_root / IMAGE_LOG_CSV
    jsonl_path = mask_root / IMAGE_LOG_JSONL

    _ensure_csv_header(csv_path)

    # CSV
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES, extrasaction="ignore")
        writer.writerow(row)

    # JSONL
    with open(jsonl_path, "a") as f:
        f.write(json.dumps(row, default=str) + "\n")


# ──────────────────────────────────────────────────────────────────────
# Visualization (qualitative only — NOT for IoU computation)
# ──────────────────────────────────────────────────────────────────────

def save_dense_color_overlay(image_bgr, masks, output_path):
    """
    Save a dense visualization of masks overlaid on the input image.

    This is a QUALITATIVE visualization only. Masks are sorted by area
    (descending) and painted into unassigned pixels to produce a full-
    coverage colour overlay. This output is NOT suitable for IoU
    computation — use the raw .npz masks for quantitative evaluation.
    """
    if len(masks) == 0:
        cv2.imwrite(output_path, image_bgr)
        return

    # Sort masks by area descending
    sorted_anns = sorted(masks, key=(lambda x: x['area']), reverse=True)
    h, w = sorted_anns[0]['segmentation'].shape

    # Create dense label map
    label_map = np.zeros((h, w), dtype=np.int32)
    for i, ann in enumerate(sorted_anns, 1):
        m = ann['segmentation']
        # Fill only unassigned pixels
        valid = (label_map == 0) & m
        label_map[valid] = i

    # Assign random colors (consistent seed for reproducibility)
    rng = np.random.RandomState(42)
    num_masks = len(sorted_anns)
    colors = rng.randint(0, 255, size=(num_masks + 1, 3)).astype(np.uint8)
    colors[0] = [0, 0, 0]  # Unassigned background

    color_overlay = colors[label_map]

    # Alpha blend where masks are present
    alpha = 0.5
    mask_valid = label_map > 0

    output_image = image_bgr.copy().astype(np.float32)
    overlay_f = color_overlay.astype(np.float32)

    output_image[mask_valid] = (
        output_image[mask_valid] * (1 - alpha) + overlay_f[mask_valid] * alpha
    )
    output_image = output_image.clip(0, 255).astype(np.uint8)

    cv2.imwrite(output_path, output_image)


# ──────────────────────────────────────────────────────────────────────
# Config Resolution
# ──────────────────────────────────────────────────────────────────────

def resolve_config_path(config_arg, sam2_module):
    """
    Resolve config path:
    1. Check if absolute path exists.
    2. Check if relative to sam2 package configs.
    """
    if os.path.isfile(config_arg):
        return config_arg

    sam2_root = Path(sam2_module.__file__).parent

    candidates = [
        sam2_root / "configs" / config_arg,
        sam2_root / "sam2_configs" / config_arg,
        sam2_root / ".." / "sam2_configs" / config_arg,
    ]

    for c in candidates:
        if c.exists():
            return str(c)

    # Return as-is if not found (build_sam2 may handle internal resolution)
    return config_arg


# ──────────────────────────────────────────────────────────────────────
# Image Discovery & Filtering
# ──────────────────────────────────────────────────────────────────────

def discover_images(input_dir: Path, only_images=None, max_images=None):
    """
    Find all images in a directory, deduplicate by stem, and optionally
    filter to a specific subset.

    Args:
        input_dir: Directory to scan for images.
        only_images: Optional list of filenames/stems to include.
        max_images: Optional cap on number of images returned.

    Returns:
        Sorted list of Path objects. Deterministic ordering.
    """
    image_extensions = ['*.png', '*.jpg', '*.jpeg']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(input_dir.glob(ext))

    # Deduplicate by stem, preferring .png > .jpg > .jpeg
    stem_to_path = {}
    ext_priority = {'.png': 0, '.jpg': 1, '.jpeg': 2}
    for p in image_paths:
        stem = p.stem
        ext = p.suffix.lower()
        if stem not in stem_to_path:
            stem_to_path[stem] = p
        else:
            current_ext = stem_to_path[stem].suffix.lower()
            if ext_priority.get(ext, 99) < ext_priority.get(current_ext, 99):
                stem_to_path[stem] = p

    image_paths = sorted(stem_to_path.values())

    # Filter to requested subset (match by stem or full filename)
    if only_images:
        target_stems = {Path(name).stem for name in only_images}
        image_paths = [p for p in image_paths if p.stem in target_stems]

    # Cap count for quick testing
    if max_images is not None and max_images > 0:
        image_paths = image_paths[:max_images]

    return image_paths


# ──────────────────────────────────────────────────────────────────────
# Single Image Processing
# ──────────────────────────────────────────────────────────────────────

def process_single_image(
    generator,
    img_path: Path,
    mask_path: Path,
    viz_path: Path,
    condition: str,
    level: str,
    profile_name: str,
    config_path: str,
    checkpoint_path: str,
    mask_root: Path,
    resume: bool,
    strict_resolution: bool,
):
    """
    Process one image through SAM2 mask generation.

    Handles:
        - Resume logic (skip if outputs exist and are valid)
        - Partial output recovery (viz missing but npz present, etc.)
        - Resolution validation
        - Runtime measurement
        - Structured logging

    Returns:
        A log dict suitable for append_image_log().
    """
    split_type = condition if condition != "clean" else "clean"
    log_row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "weather_condition": condition,
        "severity_level": level,
        "split_type": split_type,
        "input_filename": img_path.name,
        "input_path": str(img_path),
        "width": "",
        "height": "",
        "resolution_ok": "",
        "profile_used": profile_name,
        "checkpoint_used": checkpoint_path,
        "config_used": config_path,
        "start_time": "",
        "end_time": "",
        "runtime_seconds": "",
        "status": "",
        "number_of_masks_generated": "",
        "mask_output_path": str(mask_path),
        "viz_output_path": str(viz_path),
    }

    # ── Resume checks ─────────────────────────────────────────────
    npz_exists = mask_path.exists()
    viz_exists = viz_path.exists()
    npz_valid = False

    if npz_exists:
        try:
            data = np.load(str(mask_path), allow_pickle=True)
            masks_cached = data["masks"]
            npz_valid = True
        except Exception:
            print(f"  ⚠  Corrupt .npz detected: {mask_path.name} — will rerun")
            npz_valid = False

    if resume:
        if npz_valid and viz_exists:
            # Both outputs exist and npz is readable — skip entirely
            log_row["status"] = "skipped_existing"
            print(f"  SKIP (exists): {img_path.name}")
            return log_row

        if npz_valid and not viz_exists:
            # npz is good but viz is missing — regenerate viz only
            print(f"  REGEN VIZ: {img_path.name}")
            image_bgr = cv2.imread(str(img_path))
            if image_bgr is not None:
                masks_list = list(masks_cached) if isinstance(masks_cached, np.ndarray) else masks_cached
                viz_path.parent.mkdir(parents=True, exist_ok=True)
                save_dense_color_overlay(image_bgr, masks_list, str(viz_path))
                log_row["status"] = "regenerated_viz"
            else:
                log_row["status"] = "failed_read"
            return log_row

        # If viz exists but npz is missing/corrupt, fall through to full rerun

    # ── Read image ────────────────────────────────────────────────
    image_bgr = cv2.imread(str(img_path))
    if image_bgr is None:
        print(f"  ✗ Could not read: {img_path.name}")
        log_row["status"] = "failed_read"
        append_image_log(mask_root, log_row)
        return log_row

    h, w = image_bgr.shape[:2]
    log_row["width"] = w
    log_row["height"] = h
    resolution_ok = image_bgr.shape == CITYSCAPES_SHAPE
    log_row["resolution_ok"] = resolution_ok

    # ── Resolution validation ─────────────────────────────────────
    if not resolution_ok:
        if strict_resolution:
            print(
                f"  ✗ SKIP (bad resolution): {img_path.name} "
                f"— got {image_bgr.shape}, expected {CITYSCAPES_SHAPE}"
            )
            log_row["status"] = "skipped_bad_resolution"
            append_image_log(mask_root, log_row)
            return log_row
        else:
            print(
                f"  ⚠  WARNING: {img_path.name} has resolution "
                f"{image_bgr.shape}, expected {CITYSCAPES_SHAPE}. "
                f"Proceeding (--allow_non_cityscapes_resolution is set)."
            )

    # ── Run SAM2 ──────────────────────────────────────────────────
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    start_time = time.time()
    log_row["start_time"] = datetime.now(timezone.utc).isoformat()

    try:
        masks = generator.generate(image_rgb)
    except Exception as e:
        end_time = time.time()
        log_row["end_time"] = datetime.now(timezone.utc).isoformat()
        log_row["runtime_seconds"] = round(end_time - start_time, 2)
        log_row["status"] = "failed_sam2"
        print(f"  ✗ SAM2 failed for {img_path.name}: {e}")
        append_image_log(mask_root, log_row)
        return log_row

    end_time = time.time()
    runtime = round(end_time - start_time, 2)
    log_row["end_time"] = datetime.now(timezone.utc).isoformat()
    log_row["runtime_seconds"] = runtime
    log_row["number_of_masks_generated"] = len(masks)
    log_row["status"] = "processed"

    # ── Save outputs ──────────────────────────────────────────────
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    viz_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(str(mask_path), masks=masks)
    save_dense_color_overlay(image_bgr, masks, str(viz_path))

    print(
        f"  ✓ {img_path.name}  |  {len(masks)} masks  |  {runtime:.1f}s"
    )

    append_image_log(mask_root, log_row)
    return log_row


# ──────────────────────────────────────────────────────────────────────
# Directory Processing
# ──────────────────────────────────────────────────────────────────────

def process_directory(
    generator,
    input_dir: Path,
    mask_output_dir: Path,
    viz_output_dir: Path,
    condition: str,
    level: str,
    profile_name: str,
    config_path: str,
    checkpoint_path: str,
    mask_root: Path,
    resume: bool,
    strict_resolution: bool,
    only_images=None,
    max_images=None,
):
    """
    Process all matching images in a single directory.

    Returns:
        List of log dicts (one per image).
    """
    image_paths = discover_images(input_dir, only_images, max_images)

    if not image_paths:
        print(f"  No matching images in {input_dir}")
        return []

    print(f"  {len(image_paths)} image(s) to process in {input_dir}")

    mask_output_dir.mkdir(parents=True, exist_ok=True)
    viz_output_dir.mkdir(parents=True, exist_ok=True)

    logs = []
    for img_path in image_paths:
        mask_path = mask_output_dir / (img_path.stem + ".npz")
        viz_path = viz_output_dir / (img_path.stem + ".png")

        row = process_single_image(
            generator=generator,
            img_path=img_path,
            mask_path=mask_path,
            viz_path=viz_path,
            condition=condition,
            level=level,
            profile_name=profile_name,
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            mask_root=mask_root,
            resume=resume,
            strict_resolution=strict_resolution,
        )
        logs.append(row)

    return logs


# ──────────────────────────────────────────────────────────────────────
# Run Summary
# ──────────────────────────────────────────────────────────────────────

def summarize_run(all_logs: list, total_start: float):
    """Print a concise end-of-run summary to the console."""
    total_elapsed = round(time.time() - total_start, 1)

    processed = [r for r in all_logs if r.get("status") == "processed"]
    skipped_existing = [r for r in all_logs if r.get("status") == "skipped_existing"]
    skipped_resolution = [r for r in all_logs if r.get("status") == "skipped_bad_resolution"]
    regen_viz = [r for r in all_logs if r.get("status") == "regenerated_viz"]
    failed = [r for r in all_logs if r.get("status", "").startswith("failed")]

    runtimes = [
        r["runtime_seconds"] for r in processed
        if r.get("runtime_seconds") not in ("", None)
    ]
    avg_runtime = round(sum(runtimes) / len(runtimes), 1) if runtimes else 0

    print()
    print("=" * 60)
    print("  RUN SUMMARY")
    print("=" * 60)
    print(f"  Processed:            {len(processed)}")
    print(f"  Skipped (existing):   {len(skipped_existing)}")
    print(f"  Skipped (resolution): {len(skipped_resolution)}")
    print(f"  Regenerated viz only: {len(regen_viz)}")
    print(f"  Failed:               {len(failed)}")
    print(f"  ─────────────────────────────────")
    print(f"  Total images:         {len(all_logs)}")
    if runtimes:
        print(f"  Avg runtime/image:    {avg_runtime}s")
        print(f"  Min runtime:          {min(runtimes)}s")
        print(f"  Max runtime:          {max(runtimes)}s")
    print(f"  Total wall time:      {total_elapsed}s")
    print("=" * 60)
    print()


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run SAM2 automatic mask generation on Cityscapes clean "
                    "and weather-corrupted images for robustness evaluation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default run (balanced profile, levels 1,3,5)
  python scripts/run_sam2.py --checkpoint checkpoints/sam2.1_hiera_large.pt

  # Single image
  python scripts/run_sam2.py --checkpoint checkpoints/sam2.1_hiera_large.pt \\
      --only_images frankfurt_000001_064925_leftImg8bit.png

  # Two specific images
  python scripts/run_sam2.py --checkpoint checkpoints/sam2.1_hiera_large.pt \\
      --only_images img1.png,img2.png

  # Resume after interruption
  python scripts/run_sam2.py --checkpoint checkpoints/sam2.1_hiera_large.pt --resume
        """
    )

    # ── Paths ─────────────────────────────────────────────────────
    parser.add_argument(
        "--data_root", type=str, default="data",
        help="Root data directory containing clean/ and weather/ subfolders."
    )
    parser.add_argument(
        "--mask_root", type=str, default="masks/sam",
        help="Root directory for saving raw mask .npz files."
    )
    parser.add_argument(
        "--result_root", type=str, default="results/qualitative/sam2",
        help="Root directory for saving qualitative overlay PNGs."
    )

    # ── Model ─────────────────────────────────────────────────────
    parser.add_argument(
        "--config", type=str, default="sam2.1_hiera_b+.yaml",
        help="SAM2 config YAML filename or path."
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to SAM2 checkpoint file."
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device to run inference on (cuda/cpu)."
    )

    # ── Profile ───────────────────────────────────────────────────
    parser.add_argument(
        "--profile", type=str, default="balanced",
        choices=["balanced", "high_quality", "corrupt_permissive"],
        help="Inference profile controlling mask generator parameters. "
             "'balanced' (default) is practical for dissertation main runs. "
             "'high_quality' is heavier but still reasonable. "
             "'corrupt_permissive' is for ablation studies ONLY."
    )
    parser.add_argument(
        "--ablation_mode", action="store_true",
        help="Enable ablation mode: use --profile for clean images and "
             "'corrupt_permissive' for corrupted images. Results are NOT "
             "suitable for fair clean-vs-corrupt comparison. Clearly logged."
    )

    # ── Conditions & Levels ───────────────────────────────────────
    parser.add_argument(
        "--conditions", type=str, default="rain,fog,frost,snow,brightness",
        help="Comma-separated weather conditions to process."
    )
    parser.add_argument(
        "--levels", type=str, default="1,3,5",
        help="Comma-separated severity levels. Default: 1,3,5 for dissertation."
    )

    # ── Image Selection ───────────────────────────────────────────
    parser.add_argument(
        "--only_images", type=str, default=None,
        help="Comma-separated filenames or stems to process (e.g. "
             "'img1.png,img2.png'). Only these images will be processed "
             "across all conditions/levels."
    )
    parser.add_argument(
        "--max_images", type=int, default=None,
        help="Maximum number of images to process per directory. "
             "Useful for quick testing."
    )

    # ── Resume / Overwrite ────────────────────────────────────────
    parser.add_argument(
        "--resume", action="store_true", default=True,
        help="Skip images with existing valid outputs (default: enabled)."
    )
    parser.add_argument(
        "--no_resume", action="store_true",
        help="Disable resume — process all images even if outputs exist."
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Delete existing mask_root and result_root before running. "
             "Implies --no_resume."
    )
    parser.add_argument(
        "--allow_manifest_mismatch", action="store_true",
        help="Allow resuming even when the experiment config has changed. "
             "USE WITH CAUTION — may mix incompatible results."
    )

    # ── Resolution ────────────────────────────────────────────────
    parser.add_argument(
        "--allow_non_cityscapes_resolution", action="store_true",
        help="Process images even if they are not 1024x2048x3. "
             "Default behaviour is strict: non-Cityscapes images are skipped."
    )

    args = parser.parse_args()

    # ── Parse list arguments ──────────────────────────────────────
    levels = [int(x.strip()) for x in args.levels.split(",") if x.strip()]
    conditions = [x.strip() for x in args.conditions.split(",") if x.strip()]
    only_images = None
    if args.only_images:
        only_images = [x.strip() for x in args.only_images.split(",") if x.strip()]

    # ── Resolve flags ─────────────────────────────────────────────
    resume = args.resume and not args.no_resume
    if args.overwrite:
        resume = False
    strict_resolution = not args.allow_non_cityscapes_resolution

    # ── Resolve config ────────────────────────────────────────────
    config_path = resolve_config_path(args.config, sam2)

    # ── Determine profiles ────────────────────────────────────────
    clean_profile = args.profile
    corrupt_profile = args.profile  # SAME by default — this is critical for fairness

    if args.ablation_mode:
        corrupt_profile = "corrupt_permissive"
        print()
        print("!" * 60)
        print("  ABLATION MODE ENABLED")
        print("  Clean images:   profile = " + clean_profile)
        print("  Corrupt images: profile = " + corrupt_profile)
        print("  Results are NOT for main fair comparison!")
        print("!" * 60)
        print()

    clean_params = get_profile_params(clean_profile)
    corrupt_params = get_profile_params(corrupt_profile)

    # ── Startup banner ────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  SAM2 Automatic Mask Generation — Cityscapes Robustness")
    print("=" * 60)
    print(f"  Config:           {config_path}")
    print(f"  Checkpoint:       {args.checkpoint}")
    print(f"  Device:           {args.device}")
    print(f"  Profile (clean):  {clean_profile}")
    print(f"  Profile (corrupt):{corrupt_profile}")
    if clean_profile == corrupt_profile:
        print(f"  Fair comparison:  ✓ Same settings for clean & corrupt")
    else:
        print(f"  Fair comparison:  ✗ ABLATION MODE — different settings!")
    print(f"  Generator params: {clean_params}")
    if clean_profile != corrupt_profile:
        print(f"  Corrupt params:   {corrupt_params}")
    print(f"  Conditions:       {conditions}")
    print(f"  Levels:           {levels}")
    print(f"  Resume:           {resume}")
    print(f"  Overwrite:        {args.overwrite}")
    print(f"  Strict resolution:{strict_resolution}")
    if only_images:
        print(f"  Only images:      {only_images}")
    if args.max_images:
        print(f"  Max images/dir:   {args.max_images}")
    print(f"  Data root:        {args.data_root}")
    print(f"  Mask root:        {args.mask_root}")
    print(f"  Result root:      {args.result_root}")
    print("=" * 60)
    print()

    # ── Build manifest ────────────────────────────────────────────
    mask_root = Path(args.mask_root)
    result_root = Path(args.result_root)

    manifest_data = {
        "script": "run_sam2.py",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "data_root": str(args.data_root),
        "mask_root": str(args.mask_root),
        "result_root": str(args.result_root),
        "config_path": config_path,
        "checkpoint_path": args.checkpoint,
        "device": args.device,
        "profile": clean_profile,
        "corrupt_profile": corrupt_profile,
        "generator_params": clean_params,
        "corrupt_generator_params": corrupt_params,
        "conditions": conditions,
        "levels": levels,
        "overwrite": args.overwrite,
        "resume": resume,
        "strict_resolution": strict_resolution,
        "ablation_mode": args.ablation_mode,
        "only_images": only_images,
        "max_images": args.max_images,
    }

    # ── Overwrite cleanup ─────────────────────────────────────────
    if args.overwrite:
        for p in [mask_root, result_root]:
            if p.exists():
                print(f"Removing {p}...")
                shutil.rmtree(p)

    # ── Validate manifest ─────────────────────────────────────────
    load_or_validate_manifest(
        mask_root, manifest_data, resume, args.allow_manifest_mismatch
    )

    # ── Load SAM2 model ───────────────────────────────────────────
    print(f"Loading SAM2 model from {args.checkpoint}...")
    try:
        sam2_model = build_sam2(
            config_path, args.checkpoint,
            device=args.device, apply_postprocessing=False
        )
        generator_clean = build_mask_generator_from_profile(sam2_model, clean_profile)
        if clean_profile == corrupt_profile:
            generator_corrupt = generator_clean
        else:
            generator_corrupt = build_mask_generator_from_profile(sam2_model, corrupt_profile)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load SAM2 model: {e}")
        print("Check your config and checkpoint paths.")
        traceback.print_exc()
        return

    # ── Run ───────────────────────────────────────────────────────
    total_start = time.time()
    all_logs = []

    # 1. Clean data
    clean_dir = Path(args.data_root) / "clean"
    if clean_dir.exists():
        print(f"\n{'─'*60}")
        print(f"  CLEAN  |  profile={clean_profile}")
        print(f"{'─'*60}")
        logs = process_directory(
            generator=generator_clean,
            input_dir=clean_dir,
            mask_output_dir=mask_root / "clean",
            viz_output_dir=result_root / "clean",
            condition="clean",
            level="N/A",
            profile_name=clean_profile,
            config_path=config_path,
            checkpoint_path=args.checkpoint,
            mask_root=mask_root,
            resume=resume,
            strict_resolution=strict_resolution,
            only_images=only_images,
            max_images=args.max_images,
        )
        all_logs.extend(logs)
    else:
        print(f"⚠  Clean directory not found: {clean_dir}")

    # 2. Weather conditions
    for weather in conditions:
        for lvl in levels:
            base_input_dir = Path(args.data_root) / weather / f"level_{lvl}"
            if not base_input_dir.exists():
                continue

            # Support both layouts: weather/level_x/ and weather/level_x/clean/
            input_dir = base_input_dir / "clean"
            if not input_dir.exists():
                input_dir = base_input_dir

            print(f"\n{'─'*60}")
            print(f"  {weather.upper()} L{lvl}  |  profile={corrupt_profile}")
            if args.ablation_mode:
                print(f"  ⚠ ABLATION MODE — not for main comparison")
            print(f"{'─'*60}")

            logs = process_directory(
                generator=generator_corrupt,
                input_dir=input_dir,
                mask_output_dir=mask_root / weather / f"level_{lvl}",
                viz_output_dir=result_root / weather / f"level_{lvl}",
                condition=weather,
                level=str(lvl),
                profile_name=corrupt_profile,
                config_path=config_path,
                checkpoint_path=args.checkpoint,
                mask_root=mask_root,
                resume=resume,
                strict_resolution=strict_resolution,
                only_images=only_images,
                max_images=args.max_images,
            )
            all_logs.extend(logs)

    # ── Summary ───────────────────────────────────────────────────
    summarize_run(all_logs, total_start)
    print("✅ SAM2 processing complete!")


if __name__ == "__main__":
    main()
