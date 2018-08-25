import numpy as np
import scipy.linalg as la


def _inverse(R, t, s=1.0):
    assert la.norm(R.T - la.inv(R), 2) < 1e-4, "Rotation matrix must be orthogonal."  # Sanity check.

    # t should be a column vector.
    if len(t.shape) == 1:
        t = t[:, None]

    R_inv = R.T
    t_inv = -R.T.dot(t) / s
    s_inv = 1.0 / s

    return R_inv, t_inv, s_inv


def camera_extrinsics(R, t):
    R_inv, t_inv, s_inv = _inverse(R, t, s=1.0)

    cam_pos = t_inv.ravel()
    np.hstack((s_inv * R_inv, t_inv))
