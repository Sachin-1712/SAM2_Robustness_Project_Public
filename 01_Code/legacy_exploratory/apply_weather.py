"""
LEGACY / PROTOTYPE NOTE:
This script was an early prototype used for exploratory weather corruption tests.
It was NOT used for the final reported corruptions in the dissertation, which were
generated using a separate diffusion-based pipeline (Nano Banana Pro).
"""

"""
apply_weather.py

Generate *paper-style* corruption levels (5 levels) for a folder of clean images.

Outputs:
  data/
    clean/                      # input images
    rain/level_1..level_5/
    snow/level_1..level_5/
    fog/level_1..level_5/
    frost/level_1..level_5/
    brightness/level_1..level_5/

Also saves qualitative grids (2x3):
  results/qualitative/weather_grids/<image_stem>_<corruption>.png
  [Clean, Level 1, Level 2]
  [Level 3, Level 4, Level 5]

Notes:
- Uses Albumentations for rain/snow/fog/brightness
- Implements a custom "frost" overlay
- Reproducible: fixed seeds per image+corruption+level
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import cv2
import numpy as np

try:
    import albumentations as A
except ImportError as e:
    raise SystemExit(
        "Albumentations not installed. Run:\n"
        "  pip install albumentations opencv-python\n"
    ) from e


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


# -----------------------------
# Utilities
# -----------------------------
def list_images(folder: Path) -> list[Path]:
    return sorted([p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTS])


def imread_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Could not read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb


def imwrite_rgb(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)


def set_seed(seed: int) -> None:
    # For reproducibility across numpy + albumentations
    np.random.seed(seed)
    # import random; random.seed(seed) # Optional if using random module directly


def make_grid_2x3(images: list[np.ndarray], pad: int = 4) -> np.ndarray:
    """
    Create a 2x3 grid from exactly 6 images (Clean + 5 Levels).
    Row 1: Clean, Lvl 1, Lvl 2
    Row 2: Lvl 3, Lvl 4, Lvl 5
    """
    # If we have fewer than 6, pad with black or duplicate?
    # Logic: we expect clean + 5 levels = 6 images.
    
    targets = images[:6]
    while len(targets) < 6:
        # Pad with black image of same size if missing levels
        h, w = targets[0].shape[:2] if targets else (256, 256)
        targets.append(np.zeros((h, w, 3), dtype=np.uint8))
    
    # Assuming all images are same size
    h, w = targets[0].shape[:2]
    
    # Create canvas
    grid_h = 2 * h + pad
    grid_w = 3 * w + 2 * pad
    canvas = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
    
    # Positions: (row, col)
    # 0 -> (0,0), 1 -> (0,1), 2 -> (0,2)
    # 3 -> (1,0), 4 -> (1,1), 5 -> (1,2)
    
    for idx, img in enumerate(targets):
        r = idx // 3
        c = idx % 3
        
        y0 = r * (h + pad)
        x0 = c * (w + pad)
        
        # Resize if necessary (though they should be same)
        curr_h, curr_w = img.shape[:2]
        if (curr_h, curr_w) != (h, w):
             img = cv2.resize(img, (w, h))
             
        canvas[y0 : y0 + h, x0 : x0 + w] = img

    return canvas


def put_label(img: np.ndarray, text: str, xy=(12, 30)) -> np.ndarray:
    out = img.copy()
    # Black outline
    cv2.putText(out, text, xy, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
    # White text
    cv2.putText(out, text, xy, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return out


# -----------------------------
# Corruptions (5 levels)
# -----------------------------
def transform_rain(level: int) -> A.Compose:
    # level: 1..5 (light -> heavy/stormy)
    
    # Brightness reduction (darker for stormier)
    bright = {1: -0.05, 2: -0.10, 3: -0.15, 4: -0.20, 5: -0.25}[level]
    
    # Rain parameters
    # drop_length=(10, 20), drop_width=(1, 1), etc.
    if level == 1:
         rain_params = dict(drop_length=10, drop_width=1, blur_value=3, brightness_coefficient=0.9, rain_type='drizzle')
    elif level == 2:
         rain_params = dict(drop_length=15, drop_width=1, blur_value=4, brightness_coefficient=0.85, rain_type='default')
    elif level == 3:
         rain_params = dict(drop_length=20, drop_width=1, blur_value=5, brightness_coefficient=0.8, rain_type='default')
    elif level == 4:
         rain_params = dict(drop_length=25, drop_width=2, blur_value=6, brightness_coefficient=0.75, rain_type='heavy') 
    else: # level 5
         rain_params = dict(drop_length=30, drop_width=2, blur_value=7, brightness_coefficient=0.7, rain_type='torrential')

    transforms = [
        A.RandomRain(p=1.0, **rain_params),
        A.RandomBrightnessContrast(brightness_limit=(bright-0.05, bright+0.05), contrast_limit=(-0.2, -0.1), p=0.7),
        A.MotionBlur(blur_limit=(3, 5 + level), p=0.3 + 0.1*level),
    ]
    
    return A.Compose(transforms)


def transform_snow(level: int) -> A.Compose:
    # Snow: 1 (light flurries) -> 5 (blizzard/whiteout)
    
    snow_point = {1: 0.1, 2: 0.2, 3: 0.3, 4: 0.45, 5: 0.6}[level]
    brightness = {1: 1.1, 2: 1.2, 3: 1.25, 4: 1.3, 5: 1.4}[level]
    
    transforms = [
        A.RandomSnow(
            snow_point_lower=snow_point-0.05, 
            snow_point_upper=snow_point+0.05, 
            brightness_coeff=brightness, 
            p=1.0
        ),
        # Add some blur for falling snow effect
        A.GaussianBlur(blur_limit=(3, 3 + (level//2)*2), p=0.2 * level)
    ]
    return A.Compose(transforms)


def transform_fog(level: int) -> A.Compose:
    # Fog: 1 (light haze) -> 5 (dense fog)
    
    fog_coef_lower = {1: 0.1, 2: 0.2, 3: 0.3, 4: 0.45, 5: 0.6}[level]
    fog_coef_upper = {1: 0.2, 2: 0.3, 3: 0.45, 4: 0.6, 5: 0.75}[level]
    alpha_coef = {1: 0.1, 2: 0.2, 3: 0.3, 4: 0.4, 5: 0.5}[level]
    
    transforms = [
        A.RandomFog(
            fog_coef_lower=fog_coef_lower, 
            fog_coef_upper=fog_coef_upper, 
            alpha_coef=alpha_coef, 
            p=1.0
        ),
        # Desaturate as fog gets thicker
        A.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=-10*level, val_shift_limit=0, p=0.8)
    ]
    return A.Compose(transforms)


def transform_brightness(level: int) -> A.Compose:
    # Brightness: 1 (slight overexpose) -> 5 (severe overexpose/washed out)
    # Using RandomBrightnessContrast
    
    limit = {1: 0.1, 2: 0.2, 3: 0.3, 4: 0.4, 5: 0.5}[level]
    
    return A.Compose([
        A.RandomBrightnessContrast(
            brightness_limit=(limit, limit + 0.1), 
            contrast_limit=(-0.1, 0.1), 
            p=1.0
        )
    ])


def frost_overlay(img_rgb: np.ndarray, level: int, seed: int) -> np.ndarray:
    """
    Custom Frost implementation using noise patterns + ice colors.
    """
    set_seed(seed)
    h, w = img_rgb.shape[:2]

    # Generate frost mask logic (similar to previous but tuned)
    # Base noise
    noise = np.random.rand(h, w).astype(np.float32)
    
    # Kernel size for "crystals"
    k = {1: 5, 2: 9, 3: 13, 4: 17, 5: 21}[level]
    # Ensure k is odd
    if k % 2 == 0: k += 1
        
    blurred = cv2.GaussianBlur(noise, (k, k), 0)
    
    # Edges to simulate crystal structures
    edges = cv2.Laplacian(blurred, cv2.CV_32F, ksize=3)
    edges = np.abs(edges)
    edges = (edges - edges.min()) / (edges.max() - edges.min() + 1e-6)
    
    # Combine
    frost_map = 0.6 * blurred + 0.4 * edges
    
    # Thresholding based on level
    thresh = {1: 0.7, 2: 0.6, 3: 0.5, 4: 0.45, 5: 0.4}[level]
    mask = (frost_map > thresh).astype(np.float32)
    
    # Smooth the mask edges
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    
    # Frost color: Ice Blue/White
    frost_color = np.array([220, 240, 255], dtype=np.float32)
    
    # Blend
    img_float = img_rgb.astype(np.float32)
    strength = {1: 0.3, 2: 0.5, 3: 0.65, 4: 0.8, 5: 0.9}[level]
    
    # Reshape for broadcasting
    mask_3d = mask[..., None]
    
    out = img_float * (1 - mask_3d * strength) + frost_color * (mask_3d * strength)
    
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_corruption(img_rgb: np.ndarray, corruption: str, level: int, seed: int) -> np.ndarray:
    set_seed(seed)

    if corruption == "rain":
        aug = transform_rain(level)
        return aug(image=img_rgb)["image"]

    if corruption == "snow":
        aug = transform_snow(level)
        return aug(image=img_rgb)["image"]

    if corruption == "fog":
        aug = transform_fog(level)
        return aug(image=img_rgb)["image"]

    if corruption == "brightness":
        aug = transform_brightness(level)
        return aug(image=img_rgb)["image"]

    if corruption == "frost":
        return frost_overlay(img_rgb, level, seed=seed)

    raise ValueError(f"Unknown corruption: {corruption}")


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate multi-level weather corruptions (SAM-paper style).")
    parser.add_argument("--input_dir", type=str, default="data/clean", help="Folder with clean images.")
    parser.add_argument("--output_root", type=str, default="data", help="Root output folder.")
    parser.add_argument("--levels", type=int, default=5, help="Number of severity levels (default 5).")
    parser.add_argument(
        "--corruptions",
        type=str,
        default="rain,snow,fog,frost,brightness",
        help="Comma-separated list: rain,snow,fog,frost,brightness",
    )
    parser.add_argument("--overwrite", action="store_true", help="Delete output folders before regenerating.")
    parser.add_argument("--seed", type=int, default=1234, help="Base seed for reproducibility.")
    parser.add_argument("--run_root", type=str, default=None, help="Root folder for run (optional).")
    parser.add_argument("--grid_dir", type=str, default=None, help="Folder for resulting qualitative grids.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_root = Path(args.output_root)
    # Output for grids
    if args.grid_dir:
        grid_dir = Path(args.grid_dir)
    elif args.run_root:
        grid_dir = Path(args.run_root) / "results/qualitative/weather_grids"
    else:
        grid_dir = Path("results/qualitative/weather_grids")

    corruptions = [c.strip() for c in args.corruptions.split(",") if c.strip()]
    levels = int(args.levels)
    assert 1 <= levels <= 5, "Levels must be between 1 and 5."

    images = list_images(input_dir)
    if not images:
        raise SystemExit(f"No images found in: {input_dir}")

    print(f"Found {len(images)} images in {input_dir}")
    print(f"Corruptions: {corruptions} | Levels: 1..{levels}")
    
    # Handle overwrite / cleanup
    if args.overwrite:
        print("Cleaning up old corruption data...")
        for corr in corruptions:
            c_path = output_root / corr
            if c_path.exists():
                print(f"  Removing {c_path}")
                shutil.rmtree(c_path)
            
        if grid_dir.exists():
             print(f"  Removing {grid_dir}")
             shutil.rmtree(grid_dir)
             
    # Ensure directories exist
    grid_dir.mkdir(parents=True, exist_ok=True)

    for img_idx, img_path in enumerate(images):
        print(f"Processing {img_path.name}...")
        img_rgb = imread_rgb(img_path)

        for corr_idx, corr in enumerate(corruptions):
            # Collect images for grid: [Clean, Lvl1, Lvl2, Lvl3, Lvl4, Lvl5]
            grid_imgs = [put_label(img_rgb.copy(), "Clean")]

            for lv in range(1, levels + 1):
                # Unique but deterministic seed
                seed = args.seed + (img_idx * 1000) + (hash(corr) % 1000) + lv * 10
                
                out = apply_corruption(img_rgb, corr, lv, seed)

                # Save individual image
                out_dir = output_root / corr / f"level_{lv}" / "clean"
                out_dir.mkdir(parents=True, exist_ok=True)
                
                out_path = out_dir / img_path.name
                imwrite_rgb(out_path, out)

                grid_imgs.append(put_label(out, f"Lvl {lv}"))

            # Create 2x3 Grid
            grid = make_grid_2x3(grid_imgs)
            grid_out_path = grid_dir / f"{img_path.stem}_{corr}.png"
            imwrite_rgb(grid_out_path, grid)

    print("✅ Weather generation complete.")
    print(f"Grids saved to: {grid_dir.resolve()}")

if __name__ == "__main__":
    main()
