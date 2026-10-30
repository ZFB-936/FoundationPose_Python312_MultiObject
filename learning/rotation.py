"""Rotation conversions used by the pose refiner, implemented with PyTorch."""

import torch
import torch.nn.functional as F


def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """Convert (..., 3) axis-angle vectors to (..., 3, 3) rotation matrices."""
    x, y, z = axis_angle.unbind(dim=-1)
    zero = torch.zeros_like(x)
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), dim=-1)
    skew = skew.reshape(axis_angle.shape[:-1] + (3, 3))
    angle = torch.linalg.vector_norm(axis_angle, dim=-1)[..., None, None]
    identity = torch.eye(3, dtype=axis_angle.dtype, device=axis_angle.device)
    return (identity + torch.sinc(angle / torch.pi) * skew
            + 0.5 * torch.sinc(angle / (2 * torch.pi)).square() * (skew @ skew))


def rotation_6d_to_matrix(rotation: torch.Tensor) -> torch.Tensor:
    """Convert (..., 6) vectors to matrices with orthonormal rows."""
    first, second = rotation[..., :3], rotation[..., 3:]
    row1 = F.normalize(first, dim=-1)
    row2 = F.normalize(second - (row1 * second).sum(dim=-1, keepdim=True) * row1,
                       dim=-1)
    row3 = torch.cross(row1, row2, dim=-1)
    return torch.stack((row1, row2, row3), dim=-2)
