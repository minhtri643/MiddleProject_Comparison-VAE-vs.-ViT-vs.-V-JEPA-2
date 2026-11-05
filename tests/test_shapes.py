"""
Unit Tests for Output Grid Shapes and Mathematical Consistency.
"""

import unittest
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from src.analysis import compute_patch_pca, compute_cosine_map, compute_rankme, location_to_patch_index


class TestRepresentationMath(unittest.TestCase):

    def test_location_to_patch_index(self):
        grid_shape = (14, 14)
        row, col, flat_idx = location_to_patch_index(grid_shape, (0.0, 0.0))
        self.assertEqual((row, col, flat_idx), (0, 0, 0))

        row, col, flat_idx = location_to_patch_index(grid_shape, (1.0, 1.0))
        self.assertEqual((row, col, flat_idx), (13, 13, 195))

    def test_pca_reducer_shapes(self):
        dummy_grid = np.random.randn(16, 16, 1024).astype(np.float32)
        rgb_map, projected, var_pct, _ = compute_patch_pca(dummy_grid, n_components=3)
        self.assertEqual(rgb_map.shape, (16, 16, 3))
        self.assertEqual(projected.shape, (256, 3))
        self.assertTrue(0.0 <= var_pct <= 100.0)
        self.assertTrue(0.0 <= rgb_map.min() and rgb_map.max() <= 1.0)

    def test_cosine_similarity(self):
        dummy_grid = np.random.randn(32, 32, 4).astype(np.float32)
        cos_map, mean_cos, (r, c) = compute_cosine_map(dummy_grid, (0.5, 0.5))
        self.assertEqual(cos_map.shape, (32, 32))
        self.assertAlmostEqual(cos_map[r, c], 1.0, places=4)
        self.assertTrue(-1.0 <= mean_cos <= 1.0)

    def test_rankme_effective_rank(self):
        # Full rank matrix vs collapsed 1D line
        rng = np.random.default_rng(42)
        full_rank_data = rng.standard_normal((100, 20))
        rank_full = compute_rankme(full_rank_data)

        # 1D collapsed line
        v = rng.standard_normal((1, 20))
        collapsed_data = np.dot(rng.standard_normal((100, 1)), v)
        rank_collapsed = compute_rankme(collapsed_data)

        self.assertGreater(rank_full, rank_collapsed)
        self.assertAlmostEqual(rank_collapsed, 1.0, delta=0.5)


if __name__ == "__main__":
    unittest.main()
