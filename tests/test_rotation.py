"""Regression checks for the refiner's rotation conventions."""

import unittest

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from learning.rotation import axis_angle_to_matrix, rotation_6d_to_matrix


class RotationTests(unittest.TestCase):
    def test_axis_angle_matches_scipy(self):
        rng = np.random.default_rng(0)
        vectors = np.concatenate((np.zeros((1, 3)), np.eye(3) * np.pi,
                                  rng.normal(size=(100, 3)),
                                  rng.normal(size=(10, 3)) * 1e-9))
        expected = torch.from_numpy(Rotation.from_rotvec(vectors).as_matrix())
        for dtype in (torch.float32, torch.float64):
            actual = axis_angle_to_matrix(torch.tensor(vectors, dtype=dtype))
            torch.testing.assert_close(actual, expected.to(dtype), atol=2e-6, rtol=2e-6)

    def test_six_dimensional_representation_uses_rows(self):
        rotation = rotation_6d_to_matrix(torch.tensor([[0., 1., 0., 0., 0., 1.]]))
        expected = torch.tensor([[[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]]])
        torch.testing.assert_close(rotation, expected)

    def test_orthogonality_and_finite_gradients(self):
        generator = torch.Generator().manual_seed(0)
        for size, convert in ((3, axis_angle_to_matrix), (6, rotation_6d_to_matrix)):
            inputs = torch.randn(32, size, generator=generator, requires_grad=True)
            rotations = convert(inputs)
            identity = torch.eye(3).expand_as(rotations)
            torch.testing.assert_close(rotations @ rotations.transpose(-1, -2), identity)
            torch.testing.assert_close(torch.linalg.det(rotations), torch.ones(32))
            rotations.sum().backward()
            self.assertTrue(torch.isfinite(inputs.grad).all())
        zero = torch.zeros(2, 3, requires_grad=True)
        axis_angle_to_matrix(zero).sum().backward()
        self.assertTrue(torch.isfinite(zero.grad).all())


if __name__ == '__main__':
    unittest.main()
