import numpy as np
import torch


def quat2mat_torch(quaternions):
    """
    Converts given quaternion(s) to matrix/matrices.

    Args:
        quaternions (torch.Tensor): (x,y,z,w) vec4 float angles, shape (..., 4)

    Returns:
        torch.Tensor: (..., 3, 3) rotation matrices
    """
    inds = torch.tensor([3, 0, 1, 2], device=quaternions.device)
    q = quaternions[..., inds]

    n = torch.sum(q * q, dim=-1, keepdim=True)
    q = q * torch.sqrt(2.0 / n)
    q2 = torch.einsum('...i,...j->...ij', q, q)

    return torch.stack([
        1.0 - q2[..., 2, 2] - q2[..., 3, 3], q2[..., 1, 2] - q2[..., 3, 0], q2[..., 1, 3] + q2[..., 2, 0],
        q2[..., 1, 2] + q2[..., 3, 0], 1.0 - q2[..., 1, 1] - q2[..., 3, 3], q2[..., 2, 3] - q2[..., 1, 0],
        q2[..., 1, 3] - q2[..., 2, 0], q2[..., 2, 3] + q2[..., 1, 0], 1.0 - q2[..., 1, 1] - q2[..., 2, 2]
    ], dim=-1).reshape(*q.shape[:-1], 3, 3)


def quat2mat_numpy(quaternions):
    """
    Converts given quaternion(s) to matrix/matrices.

    Args:
        quaternions (np.ndarray): (x,y,z,w) vec4 float angles, shape (..., 4)

    Returns:
        np.ndarray: (..., 3, 3) rotation matrices
    """
    inds = np.array([3, 0, 1, 2])
    q = quaternions[..., inds]

    n = np.sum(q * q, axis=-1, keepdims=True)
    q = q * np.sqrt(2.0 / n)
    q2 = np.einsum('...i,...j->...ij', q, q)

    return np.stack([
        1.0 - q2[..., 2, 2] - q2[..., 3, 3], q2[..., 1, 2] - q2[..., 3, 0], q2[..., 1, 3] + q2[..., 2, 0],
        q2[..., 1, 2] + q2[..., 3, 0], 1.0 - q2[..., 1, 1] - q2[..., 3, 3], q2[..., 2, 3] - q2[..., 1, 0],
        q2[..., 1, 3] - q2[..., 2, 0], q2[..., 2, 3] + q2[..., 1, 0], 1.0 - q2[..., 1, 1] - q2[..., 2, 2]
    ], axis=-1).reshape(*q.shape[:-1], 3, 3)


def axisangle2quat_numpy(vecs):
    """
    Converts scaled axis-angle to quat for multiple vectors.

    Args:
        vecs (np.ndarray): (N, 3) array of axis-angle exponential coordinates

    Returns:
        np.ndarray: (N, 4) array of quaternions (x, y, z, w)
    """
    angles = np.linalg.norm(vecs, axis=1)
    axes = np.divide(vecs.T, angles, where=angles != 0).T  # Avoid division by zero

    quats = np.zeros((vecs.shape[0], 4))
    quats[:, 3] = np.cos(angles / 2.0)
    quats[:, :3] = axes * np.sin(angles / 2.0)[:, np.newaxis]
    quats[angles == 0, 3] = 1.0  # Handle zero-rotation case

    return quats


def axisangle2quat_torch(vecs):
    """
    Converts scaled axis-angle to quat for multiple vectors.

    Args:
        vecs (torch.Tensor): (N, 3) array of axis-angle exponential coordinates

    Returns:
        torch.Tensor: (N, 4) array of quaternions (x, y, z, w)
    """
    angles = torch.norm(vecs, dim=1)
    axes = vecs / angles.unsqueeze(1).where(angles.unsqueeze(1) != 0, torch.tensor(1.0))

    quats = torch.zeros((vecs.shape[0], 4), device=vecs.device)
    quats[:, 3] = torch.cos(angles / 2.0)
    quats[:, :3] = axes * torch.sin(angles / 2.0).unsqueeze(1)
    quats[angles == 0, 3] = 1.0  # Handle zero-rotation case

    return quats


def mat2quat_numpy(rmats):
    """
    Converts given rotation matrix to quaternion.

    Args:
        rmat (np.array): Rotation matrix with shape [..., 3, 3]

    Returns:
        np.array: Quaternion with shape [..., 4] (x, y, z, w)
    """
    """
    Converts given rotation matrices to quaternions.

    Args:
        rmats (np.array): array of shape [..., 3, 3] containing rotation matrices

    Returns:
        np.array: array of shape [..., 4] containing (x,y,z,w) float quaternion angles
    """
    """ Convert Rotation Matrix to Quaternion.  See rotation.py for notes """
    mat = np.asarray(rmats, dtype=np.float64)
    assert mat.shape[-2:] == (3, 3), "Invalid shape matrix {}".format(mat)

    Qxx, Qyx, Qzx = mat[..., 0, 0], mat[..., 0, 1], mat[..., 0, 2]
    Qxy, Qyy, Qzy = mat[..., 1, 0], mat[..., 1, 1], mat[..., 1, 2]
    Qxz, Qyz, Qzz = mat[..., 2, 0], mat[..., 2, 1], mat[..., 2, 2]
    # Fill only lower half of symmetric matrix
    K = np.zeros(mat.shape[:-2] + (4, 4), dtype=np.float64)
    K[..., 0, 0] = Qxx - Qyy - Qzz
    K[..., 1, 0] = Qyx + Qxy
    K[..., 1, 1] = Qyy - Qxx - Qzz
    K[..., 2, 0] = Qzx + Qxz
    K[..., 2, 1] = Qzy + Qyz
    K[..., 2, 2] = Qzz - Qxx - Qyy
    K[..., 3, 0] = Qyz - Qzy
    K[..., 3, 1] = Qzx - Qxz
    K[..., 3, 2] = Qxy - Qyx
    K[..., 3, 3] = Qxx + Qyy + Qzz
    K /= 3.0
    # TODO: vectorize this -- probably could be made faster
    q = np.empty(K.shape[:-2] + (4,))
    it = np.nditer(q[..., 0], flags=['multi_index'])
    while not it.finished:
        # Use Hermitian eigenvectors, values for speed
        vals, vecs = np.linalg.eigh(K[it.multi_index])
        # Select largest eigenvector, reorder to w,x,y,z quaternion
        q[it.multi_index] = vecs[[3, 0, 1, 2], np.argmax(vals)]
        # Prefer quaternion with positive w
        # (q * -1 corresponds to same rotation as q)
        if q[it.multi_index][0] < 0:
            q[it.multi_index] *= -1
        it.iternext()
    inds = np.array([1, 2, 3, 0])
    return q[..., inds]


def quat2axisangle_numpy(quats):
    """
    Converts quaternions to axis-angle format.
    Returns unit vector directions scaled by their angles in radians.

    Args:
        quats (np.array): (N, 4) array of quaternions (x,y,z,w)

    Returns:
        np.array: (N, 3) array of axis-angle exponential coordinates
    """
    # Clip quaternion w components
    quats[:, 3] = np.clip(quats[:, 3], -1.0, 1.0)

    # Calculate denominator
    den = np.sqrt(1.0 - quats[:, 3] ** 2)

    # Handle zero degree rotations
    zero_rotations = np.isclose(den, 0.0)
    axis_angles = np.zeros_like(quats[:, :3])

    # Calculate axis-angle for non-zero rotations
    non_zero_rotations = ~zero_rotations
    axis_angles[non_zero_rotations] = (quats[non_zero_rotations, :3] *
                                       (2.0 * np.arccos(quats[non_zero_rotations, 3])[:, np.newaxis]) /
                                       den[non_zero_rotations][:, np.newaxis])

    return axis_angles


def quat2axisangle_torch(quats):
    """
    Converts quaternions to axis-angle format.
    Returns unit vector directions scaled by their angles in radians.

    Args:
        quats (torch.Tensor): (N, 4) array of quaternions (x,y,z,w)

    Returns:
        torch.Tensor: (N, 3) array of axis-angle exponential coordinates
    """
    # Clip quaternion w components
    quats[:, 3] = torch.clip(quats[:, 3], -1.0, 1.0)

    # Calculate denominator
    den = torch.sqrt(1.0 - quats[:, 3] ** 2)

    # Handle zero degree rotations
    zero_rotations = torch.isclose(den, torch.tensor(0.0))
    axis_angles = torch.zeros_like(quats[:, :3])

    # Calculate axis-angle for non-zero rotations
    non_zero_rotations = ~zero_rotations
    axis_angles[non_zero_rotations] = (quats[non_zero_rotations, :3] *
                                       (2.0 * torch.arccos(quats[non_zero_rotations, 3])[:, np.newaxis]) /
                                       den[non_zero_rotations][:, np.newaxis])

    return axis_angles