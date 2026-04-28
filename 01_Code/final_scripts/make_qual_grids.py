import os
import cv2
import argparse
import numpy as np
from pathlib import Path

def put_label(img: np.ndarray, text: str, xy=(12, 30)) -> np.ndarray:
    out = img.copy()
    cv2.putText(out, text, xy, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(out, text, xy, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return out

def make_grid(cells, rows=4, cols=2, pad=4):
    h, w = cells[0].shape[:2]
    grid_h = rows * h + (rows - 1) * pad
    grid_w = cols * w + (cols - 1) * pad
    canvas = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
    
    for idx, img in enumerate(cells):
        if img is None: continue
        r = idx // cols
        c = idx % cols
        y0 = r * (h + pad)
        x0 = c * (w + pad)
        
        curr_h, curr_w = img.shape[:2]
        if (curr_h, curr_w) != (h, w):
            img = cv2.resize(img, (w, h))
        canvas[y0 : y0 + h, x0 : x0 + w] = img
        
    return canvas

def main():
    parser = argparse.ArgumentParser(description="Create qualitative grids combining clean and corrupted SAM2 masks.")
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--sam2_viz_root", default=None)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--num_images", type=int, default=3)
    parser.add_argument("--conditions", default="rain,fog,frost,snow,brightness",
                        help="Comma-separated conditions to include in grids")
    parser.add_argument("--run_root", default=None, help="Convenience arg to set data_root, sam2_viz_root, out_dir")
    parser.add_argument("--corruptions", default=None, help="Alias for --conditions")
    parser.add_argument("--levels", default="1,3,5", help="Comma-separated levels to include, e.g. 1,3,5")
    args = parser.parse_args()

    if args.run_root:
        run_root = Path(args.run_root)
        data_root = Path(args.data_root) if args.data_root else run_root / "data"
        sam_root = Path(args.sam2_viz_root) if args.sam2_viz_root else run_root / "results/qualitative/sam2"
        out_dir = Path(args.out_dir) if args.out_dir else run_root / "report_assets/qual"
    else:
        data_root = Path(args.data_root) if args.data_root else Path("data")
        sam_root = Path(args.sam2_viz_root) if args.sam2_viz_root else Path("results/qualitative/sam2")
        out_dir = Path(args.out_dir) if args.out_dir else Path("report_assets/qual")

    out_dir.mkdir(parents=True, exist_ok=True)

    cond_str = args.corruptions if args.corruptions else args.conditions
    conditions = [c.strip() for c in cond_str.split(",") if c.strip()]
    levels = [int(l.strip()) for l in args.levels.split(",") if l.strip()]

    clean_dir = data_root / "clean"
    images = sorted(clean_dir.glob("*.png"))[:args.num_images] # assuming .png

    if not images:
        print(f"No clean images found in {clean_dir}")
        return

    for img_path in images:
        stem = img_path.stem
        filename = img_path.name
        
        # Load clean images
        clean_input = cv2.imread(str(clean_dir / filename))
        if clean_input is None: continue
        clean_input = cv2.cvtColor(clean_input, cv2.COLOR_BGR2RGB)
        
        clean_sam_path = sam_root / "clean" / filename
        clean_sam = cv2.imread(str(clean_sam_path))
        if clean_sam is not None:
            clean_sam = cv2.cvtColor(clean_sam, cv2.COLOR_BGR2RGB)
        else:
            clean_sam = np.zeros_like(clean_input)

        clean_input_labeled = put_label(clean_input, "Clean Input")
        clean_sam_labeled = put_label(clean_sam, "Clean SAM2")

        for cond in conditions:
            cells = [clean_input_labeled, clean_sam_labeled]
            
            for lv in levels:
                # Corrupted input: data/<cond>/level_<nv>/clean/<filename>
                corr_in_path = data_root / cond / f"level_{lv}" / "clean" / filename
                if not corr_in_path.exists():
                     # Fallback to older struct if needed
                     corr_in_path = data_root / cond / f"level_{lv}" / filename

                corr_in = cv2.imread(str(corr_in_path))
                if corr_in is not None:
                    corr_in = cv2.cvtColor(corr_in, cv2.COLOR_BGR2RGB)
                    corr_in = put_label(corr_in, f"{cond.capitalize()} L{lv} Input")
                else:
                    corr_in = np.zeros_like(clean_input)
                    corr_in = put_label(corr_in, "Missing Input")
                    
                # SAM overlay: sam_root/<cond>/level_<nv>/<filename>
                corr_sam_path = sam_root / cond / f"level_{lv}" / filename
                corr_sam = cv2.imread(str(corr_sam_path))
                if corr_sam is not None:
                    corr_sam = cv2.cvtColor(corr_sam, cv2.COLOR_BGR2RGB)
                    corr_sam = put_label(corr_sam, f"{cond.capitalize()} L{lv} SAM2")
                else:
                    corr_sam = np.zeros_like(clean_input)
                    corr_sam = put_label(corr_sam, "Missing SAM2")

                cells.append(corr_in)
                cells.append(corr_sam)

            grid = make_grid(cells, rows=len(levels) + 1, cols=2)
            out_path = out_dir / f"{stem}_{cond}_qual.png"
            cv2.imwrite(str(out_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
            print(f"Saved {out_path}")

if __name__ == '__main__':
    main()
