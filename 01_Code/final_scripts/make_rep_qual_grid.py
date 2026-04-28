import os
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def load_image_if_exists(img_path):
    if img_path.exists():
        try:
            return mpimg.imread(str(img_path))
        except Exception as e:
            print(f"Error reading {img_path}: {e}")
            return None
    return None

def make_grid_figure(run_root, image_name, out_dir, is_overlay=False):
    """
    Creates a composite grid.
    If is_overlay is True, loads from RESULTS_DIR.
    Else loads from DATA_DIR.
    """
    data_dir = run_root / "data"
    results_dir = run_root / "results" / "qualitative" / "sam2"
    
    COLUMNS = ["Rain", "Snow", "Fog", "Frost", "Brightness"]
    ROWS = ["Clean", "L1", "L3", "L5"]
    LEVEL_MAP = {"Clean": None, "L1": "level_1", "L3": "level_3", "L5": "level_5"}

    fig, axes = plt.subplots(len(ROWS), len(COLUMNS), figsize=(15, 8))
    
    # Use tightly packed subplots with no gaps
    plt.subplots_adjust(wspace=0.02, hspace=0.02, left=0.05, right=0.95, top=0.9, bottom=0.05)
    
    for r_idx, row_label in enumerate(ROWS):
        for c_idx, col_label in enumerate(COLUMNS):
            ax = axes[r_idx, c_idx]
            
            # Construct path
            condition_folder = col_label.lower()
            level_folder = LEVEL_MAP[row_label]
            
            base_search_dir = results_dir if is_overlay else data_dir
            
            if row_label == "Clean":
                img_path = base_search_dir / "clean" / image_name
            else:
                img_path = base_search_dir / condition_folder / level_folder / image_name
                
            img = load_image_if_exists(img_path)
            
            if img is not None:
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, "Missing\n" + row_label + " " + col_label, 
                        ha='center', va='center', fontsize=8, color='red')
                ax.set_facecolor('white')
            
            # Formatting
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            
            # Column titles (only on top row)
            if r_idx == 0:
                ax.set_title(col_label, fontsize=14, pad=10)
            
            # Row titles (only on first column)
            if c_idx == 0:
                ax.set_ylabel(row_label, fontsize=14, rotation=90, labelpad=15)

    out_dir.mkdir(parents=True, exist_ok=True)
    image_stem = Path(image_name).stem
    grid_type = "mask" if is_overlay else "corruption"
    out_path = out_dir / f"{grid_type}_grid_{image_stem}.png"
    
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.1)
    plt.close()
    return out_path

def main():
    parser = argparse.ArgumentParser(description="Generate qualitative result grids for SAM2 robustness evaluation.")
    parser.add_argument("--run_root", type=str, default=".", help="Root directory of the experiment run (containing data/ and results/)")
    parser.add_argument("--image_name", type=str, default="frankfurt_000001_064925_leftImg8bit.png", help="Filename of the representative image")
    parser.add_argument("--out_dir", type=str, default="report_assets/qualitative", help="Where to save the output grids")
    
    args = parser.parse_args()
    
    run_root = Path(args.run_root)
    out_dir = Path(args.out_dir)
    image_name = args.image_name
    
    print(f"Using run_root: {run_root}")
    print(f"Target image: {image_name}")
    
    # 1. Input Corruption Grid
    print("Generating Input Corruption Grid...")
    try:
        in_path = make_grid_figure(run_root, image_name, out_dir, is_overlay=False)
        print(f"Saved Input Grid: {in_path}")
    except Exception as e:
        print(f"Failed to generate input grid: {e}")
    
    # 2. SAM2 Overlay Grid
    print("Generating SAM2 Overlay Grid...")
    try:
        out_path = make_grid_figure(run_root, image_name, out_dir, is_overlay=True)
        print(f"Saved Overlay Grid: {out_path}")
    except Exception as e:
        print(f"Failed to generate overlay grid: {e}")

if __name__ == "__main__":
    main()

