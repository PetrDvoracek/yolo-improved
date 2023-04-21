import unittest
import torch
import random

import src.anchorbox


class Sigmoid(unittest.TestCase):
    values = torch.tensor([-10, -0.1, -0.0001, 0.1, 5])
    max_allowed_diff = 1e-5

    def test_scale(self):
        outs = src.anchorbox.sigmoid(Sigmoid.values)
        self.assertGreaterEqual(outs.min(), 0)
        self.assertLessEqual(outs.max(), 1)

    def test_inverse_of_small_values(self):
        small_values = torch.tensor([-10, -0.1, -0.0001, 0.1, 5])
        sigmoids = src.anchorbox.sigmoid(small_values)
        inversed_sigmoids = src.anchorbox.inverse_sigmoid(sigmoids)
        diff = torch.abs(small_values - inversed_sigmoids)
        self.assertLess(diff.max(), Sigmoid.max_allowed_diff)

    def test_inverse(self):
        sigmoids = src.anchorbox.sigmoid(Sigmoid.values)
        inversed_sigmoids = src.anchorbox.inverse_sigmoid(sigmoids)
        diff = torch.abs(Sigmoid.values - inversed_sigmoids)
        self.assertLess(diff.max(), Sigmoid.max_allowed_diff)


class CoordTransforms(unittest.TestCase):
    def test_inverse_pred_center_to_coords(self):
        pred_t = torch.tensor(0.4)
        cell_idx = torch.tensor(7)
        pred_to_coords = src.anchorbox.box_pred_center_to_coords(pred_t, cell_idx)
        coords_to_pred = src.anchorbox.box_coords_center_to_pred(
            pred_to_coords, cell_idx
        )
        self.assertAlmostEqual(pred_t.item(), coords_to_pred.item(), places=2)


class NNOutputTransform(unittest.TestCase):
    def test_xy_between_01(self):
        scale = 13
        n_anchors = 5
        nn_out = torch.rand(scale, scale, n_anchors, 25) * 10
        anchor = torch.rand_like(nn_out)
        coords = src.anchorbox.transform_nn_output_to_coords(
            scale=scale,
            label=nn_out,
            anchor_p_w=anchor[..., 2],
            anchor_p_h=anchor[..., 3],
        )[..., :2]
        self.assertGreaterEqual(coords.min(), 0)
        self.assertLessEqual(coords.max(), 1)


if __name__ == "__main__":
    unittest.main()
