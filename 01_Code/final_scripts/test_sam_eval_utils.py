import unittest
import numpy as np
import sys
from pathlib import Path

# Add current dir to path to import sam_eval_utils
sys.path.append(str(Path(__file__).parent))
import sam_eval_utils

class TestSamEvalUtils(unittest.TestCase):
    def setUp(self):
        # Create dummy masks for testing
        self.shape = (100, 100)
        self.mask_empty = np.zeros(self.shape, dtype=bool)
        self.mask_full = np.ones(self.shape, dtype=bool)
        
        # 50x100 half mask
        self.mask_half = np.zeros(self.shape, dtype=bool)
        self.mask_half[:50, :] = True
        
        # 25x100 quarter mask (subset of half)
        self.mask_quarter = np.zeros(self.shape, dtype=bool)
        self.mask_quarter[:25, :] = True

    def test_iou_identical(self):
        """IoU of identical masks should be 1.0"""
        iou = sam_eval_utils.compute_binary_iou(self.mask_half, self.mask_half)
        self.assertAlmostEqual(iou, 1.0)

    def test_iou_non_overlap(self):
        """IoU of non-overlapping masks should be 0.0"""
        mask_other_half = ~self.mask_half
        iou = sam_eval_utils.compute_binary_iou(self.mask_half, mask_other_half)
        self.assertAlmostEqual(iou, 0.0)

    def test_iou_partial_overlap(self):
        """IoU of known partial overlap (quarter inside half)"""
        # Intersection = quarter (2500 pixels)
        # Union = half (5000 pixels)
        # IoU = 2500 / 5000 = 0.5
        iou = sam_eval_utils.compute_binary_iou(self.mask_half, self.mask_quarter)
        self.assertAlmostEqual(iou, 0.5)

    def test_iou_empty(self):
        """IoU of two empty masks should be 0.0 (per implementation choice)"""
        iou = sam_eval_utils.compute_binary_iou(self.mask_empty, self.mask_empty)
        self.assertAlmostEqual(iou, 0.0)

    def test_dice_identical(self):
        """Dice of identical masks should be 1.0"""
        dice = sam_eval_utils.compute_binary_dice(self.mask_half, self.mask_half)
        self.assertAlmostEqual(dice, 1.0)

    def test_resize_preserves_discrete(self):
        """Nearest neighbor resize should preserve discrete labels (0/1)"""
        # Upscale 2x2 to 4x4
        small_mask = np.array([[1, 0], [0, 1]], dtype=np.int32)
        resized = sam_eval_utils.resize_mask_nearest(small_mask, (4, 4))
        
        # Check that all values are either 0 or 1 (no interpolation artifacts)
        unique_vals = np.unique(resized)
        for v in unique_vals:
            self.assertIn(v, [0, 1])
        
        # Check structure (blocky)
        expected = np.array([
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
            [0, 0, 1, 1]
        ], dtype=np.int32)
        np.testing.assert_array_equal(resized, expected)

    def test_masks_to_dense_label_map_ordering(self):
        """Check that larger masks are processed first but smaller ones fill unassigned"""
        # Logic: masks sorted by area descending. 
        # But smaller masks fill "only unassigned".
        # This means if a large mask covers everything, the small mask won't show up.
        # Wait, the logic is: "Fill only unassigned pixels".
        # So larger masks go first and claim their territory. 
        # This is consistent with SAM2 qualitative overlay logic.
        
        mask_large = {"segmentation": self.mask_full, "area": 10000}
        mask_small = {"segmentation": self.mask_half, "area": 5000}
        
        # Order: Large then Small.
        # Large claims full 100x100 as label 1.
        # Small tries to claim half as label 2 but all are already assigned.
        label_map = sam_eval_utils.masks_to_dense_label_map([mask_large, mask_small])
        self.assertTrue(np.all(label_map == 1))
        
        # Reverse check: if we pass them as Small, Large
        # The function sorts them internally!
        label_map_auto_sort = sam_eval_utils.masks_to_dense_label_map([mask_small, mask_large])
        self.assertTrue(np.all(label_map_auto_sort == 1))

if __name__ == '__main__':
    unittest.main()
