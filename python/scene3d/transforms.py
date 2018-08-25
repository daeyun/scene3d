import math
import warnings

import numpy
import numpy as np
import scipy.linalg as la
import scipy.ndimage as spndim

from scene3d import log


def translate(x_or_xyz, y=None, z=None):
    """
    translate produces a translation by (x, y, z) .
    http://www.labri.fr/perso/nrougier/teaching/opengl/

    Parameters
    ----------
    x, y, z
        Specify the x, y, and z coordinates of a translation vector.
    """
    if isinstance(x_or_xyz, np.ndarray):
        assert x_or_xyz.size == 3
        x, y, z = x_or_xyz
    else:
        x = x_or_xyz
        if y is None: y = x
        if z is None: z = x
    T = [[1, 0, 0, x],
         [0, 1, 0, y],
         [0, 0, 1, z],
         [0, 0, 0, 1]]
    return np.array(T, dtype=np.float64)


def scale(x, y=None, z=None):
    """
    scale produces a non uniform scaling along the x, y, and z axes. The three
    parameters indicate the desired scale factor along each of the three axes.
    http://www.labri.fr/perso/nrougier/teaching/opengl/

    Parameters
    ----------
    x, y, z
        Specify scale factors along the x, y, and z axes, respectively.
    """
    if y is None: y = x
    if z is None: z = x
    S = [[x, 0, 0, 0],
         [0, y, 0, 0],
         [0, 0, z, 0],
         [0, 0, 0, 1]]
    return np.array(S, dtype=np.float64)


def xrotate(theta, deg=True):
    """
    http://www.labri.fr/perso/nrougier/teaching/opengl/
    """
    if deg:
        theta = np.deg2rad(theta)

    is_batch = (isinstance(theta, np.ndarray) and theta.size > 1) or np.array(theta).size > 1

    cosT = np.cos(theta)
    sinT = np.sin(theta)

    if is_batch:
        assert theta.ndim == 1
        R = np.tile(np.eye(4), (theta.shape[0], 1, 1))
        R[:, 1, 1] = cosT
        R[:, 2, 2] = cosT
        R[:, 1, 2] = -sinT
        R[:, 2, 1] = sinT
    else:
        R = numpy.array(
            [[1.0, 0.0, 0.0, 0.0],
             [0.0, cosT, -sinT, 0.0],
             [0.0, sinT, cosT, 0.0],
             [0.0, 0.0, 0.0, 1.0]], dtype=np.float64)
    return R


def yrotate(theta, deg=True):
    """
    http://www.labri.fr/perso/nrougier/teaching/opengl/
    """
    if deg:
        theta = np.deg2rad(theta)
    cosT = math.cos(theta)
    sinT = math.sin(theta)
    R = numpy.array(
        [[cosT, 0.0, sinT, 0.0],
         [0.0, 1.0, 0.0, 0.0],
         [-sinT, 0.0, cosT, 0.0],
         [0.0, 0.0, 0.0, 1.0]], dtype=np.float64)
    return R


def zrotate(theta, deg=True):
    """
    http://www.labri.fr/perso/nrougier/teaching/opengl/
    """
    if deg:
        theta = np.deg2rad(theta)

    is_batch = (isinstance(theta, np.ndarray) and theta.size > 1) or np.array(theta).size > 1

    cosT = np.cos(theta)
    sinT = np.sin(theta)

    if is_batch:
        assert theta.ndim == 1
        R = np.tile(np.eye(4), (theta.shape[0], 1, 1))
        R[:, 0, 0] = cosT
        R[:, 1, 1] = cosT
        R[:, 0, 1] = -sinT
        R[:, 1, 0] = sinT
    else:
        R = numpy.array(
            [[cosT, -sinT, 0.0, 0.0],
             [sinT, cosT, 0.0, 0.0],
             [0.0, 0.0, 1.0, 0.0],
             [0.0, 0.0, 0.0, 1.0]], dtype=np.float64)
    return R


def unit_vector_(data, axis=None, out=None):
    """
    http://www.lfd.uci.edu/~gohlke/code/transformations.py.html
    """
    if out is None:
        data = numpy.array(data, dtype=numpy.float64, copy=True)
        if data.ndim == 1:
            data /= math.sqrt(numpy.dot(data, data))
            return data
    else:
        if out is not data:
            out[:] = numpy.array(data, copy=False)
        data = out
    length = numpy.atleast_1d(numpy.sum(data * data, axis))
    numpy.sqrt(length, length)
    if axis is not None:
        length = numpy.expand_dims(length, axis)
    data /= length
    if out is None:
        return data


def rotation_matrix2(angle, direction, point=None):
    """
    http://www.lfd.uci.edu/~gohlke/code/transformations.py.html
    """
    angle = angle / 180 * np.pi
    sina = math.sin(angle)
    cosa = math.cos(angle)
    direction = unit_vector_(direction[:3])
    # rotation matrix around unit vector
    R = numpy.diag([cosa, cosa, cosa])
    R += numpy.outer(direction, direction) * (1.0 - cosa)
    direction *= sina
    R += numpy.array([[0.0, -direction[2], direction[1]],
                      [direction[2], 0.0, -direction[0]],
                      [-direction[1], direction[0], 0.0]])
    M = numpy.identity(4)
    M[:3, :3] = R
    if point is not None:
        # rotation not around origin
        point = numpy.array(point[:3], dtype=numpy.float64, copy=False)
        M[:3, 3] = point - numpy.dot(R, point)
    return M


def rotation_matrix(angle, direction, point=None, deg=True):
    """
    Based on http://www.lfd.uci.edu/~gohlke/code/transformations.py.html
    """
    assert direction.ndim in (1, 2)
    is_batch = direction.ndim == 2
    if is_batch:
        assert angle.shape[0] == direction.shape[0]
        assert direction.shape[1] == 3
        if point is not None:
            assert point.shape[1] == 3

    if deg:
        angle = np.deg2rad(angle)

    if np.issubdtype(direction.dtype, np.integer):
        direction = direction.astype(np.float64)

    sina = np.sin(angle)
    cosa = np.cos(angle)

    # rotation matrix around unit vector
    if is_batch:
        axis = np.divide(direction, la.norm(direction, ord=2, axis=1, keepdims=True))
        # If direction is (0,0,0), return identity.
        nans = np.any(np.isnan(axis), axis=1)

        R = np.zeros((cosa.shape[0], 3, 3))
        R[:, [0, 1, 2], [0, 1, 2]] = cosa[:, None]
        R += np.matmul(axis[:, :, None], axis[:, None, :]) * (1.0 - cosa)[:, None, None]
        axis *= sina[:, None]
    else:
        axis = np.divide(direction, la.norm(direction, ord=2))
        if np.any(np.isnan(axis)):
            return np.eye(4)

        if isinstance(cosa, numpy.ndarray):
            cosa = float(cosa)
        R = numpy.diag([cosa, cosa, cosa])
        R += numpy.outer(axis, axis) * (1.0 - cosa)
        axis *= sina

    if is_batch:
        R[:, [2, 0, 1], [1, 2, 0]] += axis
        R[:, [1, 2, 0], [2, 0, 1]] -= axis
        M = np.zeros((angle.shape[0], 4, 4))
        M[:, :3, :3] = R
        M[:, 3, 3] = 1
    else:
        R += numpy.array([[0.0, -axis[2], axis[1]],
                          [axis[2], 0.0, -axis[0]],
                          [-axis[1], axis[0], 0.0]])
        M = np.eye(4)
        M[:3, :3] = R

    if point is not None:
        # rotation not around origin
        if is_batch:
            M[:, :3, 3] = point - np.matmul(R, point[:, :, None]).squeeze()
        else:
            point = numpy.array(point[:3], dtype=numpy.float64, copy=False)
            M[:3, 3] = point - numpy.dot(R, point)

    if is_batch:
        M[nans] = np.eye(4)

    return M


def angle(v1, v2, axis=0, deg=False, ref_plane=None):
    """
    Angle between two vectors.
    :param axis: 0 if column vectors, 1 if row vectors.
    :param deg: Returns angle in degrees if True, radians if False.
    :param ref_plane: If set, returns a signed angle for right-handed rotation with respect to this plane.
    """
    v1n = np.divide(v1, la.norm(v1, ord=2, axis=axis, keepdims=True))
    v2n = np.divide(v2, la.norm(v2, ord=2, axis=axis, keepdims=True))

    # More numerically stable than arccos.
    dotprod = (v1n * v2n).sum(axis=axis)
    crossprod = np.cross(v1n, v2n, axis=axis)
    ret = np.arctan2(la.norm(crossprod, ord=2, axis=axis, keepdims=True), dotprod)

    if deg:
        ret = np.rad2deg(ret)
    if ref_plane is not None:
        ret *= np.sign((crossprod * ref_plane).sum(axis=axis))
    return ret


def frustum(left, right, bottom, top, znear, zfar):
    """
    http://www.labri.fr/perso/nrougier/teaching/opengl/
    """
    assert (right != left)
    assert (bottom != top)
    assert (znear != zfar)

    M = np.zeros((4, 4), dtype=np.float64)
    M[0, 0] = +2.0 * znear / (right - left)
    M[2, 0] = (right + left) / (right - left)
    M[1, 1] = +2.0 * znear / (top - bottom)
    M[3, 1] = (top + bottom) / (top - bottom)
    M[2, 2] = -(zfar + znear) / (zfar - znear)
    M[3, 2] = -2.0 * znear * zfar / (zfar - znear)
    M[2, 3] = -1.0
    return M


def perspective(fovy, aspect, znear, zfar):
    """
    http://www.labri.fr/perso/nrougier/teaching/opengl/
    """
    assert (znear != zfar)
    h = np.tan(fovy / 360.0 * np.pi) * znear
    w = h * aspect
    return frustum(-w, w, -h, h, znear, zfar)


def lookat_matrix(cam_xyz, obj_xyz, up=(0, 0, 1)):
    """
    Right-handed rigid transform. -Z points to object.

    :param cam_xyz: (3,)
    :param obj_xyz: (3,)
    :return: (3,4)
    """
    cam_xyz = np.array(cam_xyz)
    obj_xyz = np.array(obj_xyz)
    F = obj_xyz - cam_xyz
    f = F / la.norm(F)

    up = np.array(up)
    u = up / la.norm(up)

    s = np.cross(f, u)
    s /= la.norm(s)

    u = np.cross(s, f)

    R = np.vstack((s, u, -f))

    M = np.hstack([R, np.zeros((3, 1))])
    T = np.eye(4)
    T[:3, 3] = -cam_xyz
    MT = M.dot(T)

    return MT


def apply_Rt(Rt, pts, inverse=False):
    """
    :param Rt: (3,4)
    :param pts: (n,3)
    :return:
    """
    if inverse:
        R = Rt[:, :3].T
        t = -R.dot(Rt[:, 3, None])
        Rtinv = np.hstack((R, t))
        return Rtinv.dot(np.vstack((pts.T, np.ones((1, pts.shape[0]))))).T
    return Rt.dot(np.vstack((pts.T, np.ones((1, pts.shape[0]))))).T


def ortho44(left, right, bottom, top, znear, zfar):
    assert (right != left)
    assert (bottom != top)
    assert (znear != zfar)

    return np.array([
        [2.0 / (right - left), 0, 0, -(right + left) / float(right - left)],
        [0, 2.0 / (top - bottom), 0, -(top + bottom) / float(top - bottom)],
        [0, 0, -2.0 / (zfar - znear), -(zfar + znear) / float(zfar - znear)],
        [0, 0, 0, 1]
    ], dtype=np.float64)


def apply34(M, pts):
    assert 2 == len(pts.shape)
    assert pts.shape[1] == 3

    assert M.shape == (3, 4)
    Mpts = M.dot(np.vstack((pts.T, np.ones((1, pts.shape[0])))))
    return Mpts.T


def apply44(M, pts):
    assert 2 == len(pts.shape)
    assert pts.shape[1] == 3

    if M.shape == (4, 4):
        Mpts = M.dot(np.vstack((pts.T, np.ones((1, pts.shape[0])))))
        return (Mpts[:3, :] / Mpts[3, :]).T
    else:
        raise NotImplementedError()


def normalize_mesh_vertices(mesh, up='+z'):
    # pts = mesh['v'][mesh['f']]
    # a = la.norm(pts[:, 0, :] - pts[:, 1, :], 2, axis=1)
    # b = la.norm(pts[:, 1, :] - pts[:, 2, :], 2, axis=1)
    # c = la.norm(pts[:, 2, :] - pts[:, 0, :], 2, axis=1)
    # s = (a + b + c) / 2.0
    # areas_sq = s * (s - a) * (s - b) * (s - c)
    # areas_sq = np.abs(areas_sq)
    # areas = np.sqrt(areas_sq)
    # areas = np.tile(areas, 3)

    pts = mesh['v'][mesh['f'].ravel()]
    # weighted_std = stats.weighted_std(areas, pts)
    # weighted_mean = stats.weighted_mean(areas, pts)

    t = -(mesh['v'].max(0) + mesh['v'].min(0)) / 2

    furthest = la.norm(pts + t, ord=2, axis=1).max()
    # sigma = 2 * weighted_std

    scale = 1.0 / furthest

    M = np.array([
        [1, 0, 0, t[0]],
        [0, 1, 0, t[1]],
        [0, 0, 1, t[2]],
        [0, 0, 0, 1.0 / scale]
    ])

    if up == '+z':
        R = np.eye(4)
    elif up == '-z':
        R = xrotate(180)
    elif up == '+y':
        R = xrotate(90)
    elif up == '-y':
        R = xrotate(-90)
    elif up == '+x':
        R = yrotate(90)
    elif up == '-x':
        R = yrotate(-90)
    else:
        raise RuntimeError('Unrecognized up axis: {}'.format(up))

    return R.dot(M)


def cam_pts_from_ortho_depth(depth, trbl=(1, 1, -1, -1)):
    assert depth.ndim == 2
    d = depth.copy()
    if hasattr(depth, 'mask'):
        d[depth.mask] = np.nan
    im_wh = d.shape[::-1]

    newd = np.concatenate((np.indices(d.shape), d[None, :, :].data), axis=0).astype(np.float64)

    impts = np.vstack((newd[1, :, :].ravel(), newd[0, :, :].ravel(), newd[2, :, :].ravel())).T

    # important.
    impts[:, :2] += 0.5

    valid_inds = np.logical_not(np.isnan(impts[:, 2]))
    impts = impts[valid_inds, :].astype(np.float64)

    top, right, bottom, left = trbl

    impts[:, 0] *= (right - left) / im_wh[0]
    impts[:, 1] *= -(top - bottom) / im_wh[1]
    impts[:, 0] += left
    impts[:, 1] += top
    impts[:, 2] *= -1

    return impts


def cam_pts_from_perspective_depth(depth, K, trbl=(1, 1, -1, -1)):
    """
    Point cloud from opengl depth image.
    See http://www.songho.ca/opengl/gl_projectionmatrix.html

    p = K*[x, y, z].T  where z=-d
    x_n, y_n = p[:2]/z

    :param depth:
    :param K: (3, 3)
    :param trbl:
    :return:
    """
    assert K.shape == (3, 3)
    d = depth.copy()
    if hasattr(depth, 'mask'):
        d[depth.mask] = np.nan
    im_wh = d.shape[::-1]

    finite = np.isfinite(d)
    y, x = np.where(finite)
    z = d[finite]
    impts = np.stack([x, y, z], axis=1)

    # important.
    impts[:, :2] += 0.5

    top, right, bottom, left = trbl
    impts[:, 0] *= (right - left) / im_wh[0]
    impts[:, 1] *= -(top - bottom) / im_wh[1]
    impts[:, 0] += left
    impts[:, 1] += top
    impts[:, :2] *= impts[:, [2]]
    impts[:, 2] *= -1
    pts = la.inv(K).dot(impts.T).T[:, :2]
    pts = np.concatenate([pts, -z[:, None]], axis=1)

    return pts


def quaternion_matrix(quaternion):
    """Return homogeneous rotation matrix from quaternion.
    http://www.lfd.uci.edu/~gohlke/code/transformations.py.html
    """
    _EPS = numpy.finfo(float).eps * 4.0

    q = numpy.array(quaternion, dtype=numpy.float64, copy=True)
    n = numpy.dot(q, q)
    if n < _EPS:
        return numpy.identity(4)
    q *= math.sqrt(2.0 / n)
    q = numpy.outer(q, q)
    return numpy.array([
        [1.0 - q[2, 2] - q[3, 3], q[1, 2] - q[3, 0], q[1, 3] + q[2, 0], 0.0],
        [q[1, 2] + q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] - q[1, 0], 0.0],
        [q[1, 3] - q[2, 0], q[2, 3] + q[1, 0], 1.0 - q[1, 1] - q[2, 2], 0.0],
        [0.0, 0.0, 0.0, 1.0]])


def random_rotation(return_4x4=False):
    v = np.random.randn(4)
    Q = quaternion_matrix(v / la.norm(v, ord=2))

    assert np.isclose(la.det(Q), 1.0)
    assert np.isclose(Q[3, 3], 1.0)

    return Q if return_4x4 else Q[:3, :3]


def xyz_to_sph(xyz):
    """
    :return: radius, inclination, azimuth. In radians.
    0 <= inclination <= pi
    -pi < azimuth <= pi
    """
    r = la.norm(xyz, ord=2, axis=-1, keepdims=True).reshape(-1, 1)
    assert (r != 0).all()
    xyz = xyz.reshape(-1, 3)
    inclination = np.arccos(xyz[:, 2, None] / r)
    azimuth = np.arctan2(xyz[:, 1, None], xyz[:, 0, None])
    return np.hstack((r, inclination, azimuth)).reshape(xyz.shape)


def sph_to_xyz(sph, is_input_radians=True):
    """
    Spherical coordinates in radians to xyz coordinates.
    (radius, inclination, azimuth)
    """
    if isinstance(sph, (tuple, list)):
        sph = np.array(sph, dtype=np.float64).copy()
    input_dim = sph.ndim
    if input_dim == 1:
        sph = sph.reshape(1, 3).copy()

    if not is_input_radians:
        sph = sph.copy()
        rad = np.deg2rad(sph[:, 1:])
        sph[:, 1:] = rad

    r, inclination, azimuth = sph[:, 0], sph[:, 1], sph[:, 2]
    x = r * np.sin(inclination) * np.cos(azimuth)
    y = r * np.sin(inclination) * np.sin(azimuth)
    z = r * np.cos(inclination)
    xyz = np.stack((x, y, z), axis=1)
    assert xyz.shape == sph.shape

    if input_dim == 1:
        return xyz.reshape(3).copy()
    return xyz


def unit_vector(vec) -> np.ndarray:
    if isinstance(vec, (list, tuple)):
        vec = np.array(vec)

    if vec.ndim == 1:
        return vec / la.norm(vec, ord=2)
    elif vec.ndim == 2:
        return vec / la.norm(vec, ord=2, axis=1, keepdims=True)
    else:
        raise NotImplementedError()


def spherical_coord_align_rotation(v1, v2):
    """
    Returns a rotation matrix that aligns v1 to v2 such that the rotation can be parametrized as spherical coordinates.
    i.e. Rotates freely around the z-axis, and then adjusts elevation.
    :param v1: Source vector.
    :param v2: Target vector.
    :return: Rotation matrix.
    """
    a = unit_vector(v1)
    b = unit_vector(v2)
    a[2] = b[2] = 0
    azimuth = angle(a, b, deg=False, ref_plane=np.array([0, 0, 1]))

    assert -np.pi <= azimuth <= np.pi
    xyrot = zrotate(azimuth, deg=False)[:3, :3]
    vs = v1.dot(xyrot.T)
    elaxis = np.cross(vs, v2)
    elevation = angle(vs, v2, deg=False, ref_plane=elaxis)

    assert -np.pi <= elevation <= np.pi
    elrot = rotation_matrix(elevation, elaxis, deg=False)[:3, :3]
    Q = elrot.dot(xyrot)
    assert np.allclose(la.det(Q), 1.0)
    return Q


def pca_svd(X):
    """
    :param X: (n, d)
    :return: principal components (columns) and singular values.
    """
    _, s, V = la.svd(X - X.mean(axis=0))
    return V.T, s


def depth_normals(depth, worldpts, viewdir, window_size=5, min_near_pts=4, visualize=False, ax=None):
    """
    :param viewdir: vector from camera to object. normals direction will be less than 90 degrees from viewdir.
    :return: 3-channel images filled with inward normals and worldpts. There may be discarded points.
    """
    import skimage

    assert window_size % 2 == 1
    assert len(viewdir.shape) == 1
    assert np.isclose(la.norm(viewdir, 2), 1.0)
    assert len(depth.shape) == 2
    assert len(worldpts.shape) == 2
    assert depth.shape[0] > 4
    assert worldpts.shape[1] == 3
    assert (~np.isnan(depth)).sum() == worldpts.shape[0]

    imxyz = np.full((depth.shape[0], depth.shape[1], 3), np.nan, dtype=np.float32)
    imxyz[~np.isnan(depth)] = worldpts

    npts = worldpts.shape[0]
    nthsmallest = max(3, int(npts / 40))  # rough heuristic
    gap = np.partition(la.norm(worldpts[:-1, :] - worldpts[1:, :], 2, axis=1),
                       nthsmallest)[nthsmallest]
    maxdist = gap * (2 ** 0.5) * ((5 - 1) / 2) * 1.02  # rough heuristic

    padding_left, padding_right = int((window_size - 1) / 2), int((window_size - 1) / 2 + 0.5)
    windows = skimage.util.view_as_windows(
        np.pad(imxyz, [[padding_left, padding_right], [padding_left, padding_right], [0, 0]],
               mode='constant', constant_values=np.nan), (window_size, window_size, 3), step=1)

    normals = np.full(imxyz.shape, np.nan, dtype=np.float32)

    for idx in np.ndindex(imxyz.shape[0], imxyz.shape[1], 1):
        win = windows[idx].view()
        # filter center
        current = win[padding_left, padding_left]

        if np.isnan(current[0]):
            continue
        # assuming nan if first channel has nan.
        valid = ~np.isnan(win[:, :, 0])
        if valid.sum() < min_near_pts:
            continue
        pts = win[valid] - current
        dists = la.norm(win[valid] - current, ord=2, axis=1)
        valid = dists < maxdist
        if valid.sum() < min_near_pts:
            continue
        pts = pts[valid]
        pc, s = pca_svd(pts)
        normal = pc[:, 2]
        normal /= la.norm(normal, ord=2)
        normals[idx[:2]] = normal

    flip = (normals * viewdir).sum(2) > 0
    normals[flip] *= -1

    if visualize:
        from scene3d import geom3d
        pts = imxyz[~np.isnan(normals).any(axis=2)]
        vecs = normals[~np.isnan(normals).any(axis=2)]
        edges = np.stack((pts, pts + vecs * 0.05), axis=1)
        ax = geom3d.pts(pts, markersize=30, ax=ax)

        start = la.norm(worldpts - worldpts.mean(0), 2, 1).max() * 1.2 * viewdir.reshape(1, 3)
        end = worldpts.mean(0).reshape(1, 3) * 0.2 + start * 0.8
        geom3d.draw_arrow_3d(start, end, ax=ax)
        geom3d.edge_3d(edges, ax=ax, colors='green', linewidths=0.5)
        geom3d.pts(pts + vecs * 0.1, ax=ax, color='red')

    imxyz[np.isnan(normals)] = np.nan

    return normals, imxyz


def rescale_and_recenter(image, hw=(64, 64), padding=1, return_scale_and_center=False):
    assert 2 == len(image.shape)
    # Crop margins with nan values.
    y, x = np.where(~np.isnan(image))

    in_out_ratio = np.array(image.shape) / np.array(hw)
    assert np.allclose(in_out_ratio, in_out_ratio[0])
    in_out_ratio = in_out_ratio[0]

    try:
        h, w = y.max() - y.min() + 1, x.max() - x.min() + 1
        center = int(y.min() + h / 2), int(x.min() + w / 2)
        ystart = int(center[0] - h / 2)
        yend = int(center[0] + h / 2)
        xstart = int(center[1] - w / 2)
        xend = int(center[1] + w / 2)
        roi = image[ystart:yend, xstart:xend]

        # Resize.
        ratios = np.array(roi.shape) / np.array(hw)
        longest_axis = np.argmax(ratios)
        im_scale = (hw[longest_axis] - padding * 2) / roi.shape[longest_axis] + 1e-8

        # Ignore the "the output shape of zoom() is calculated with round() instead of int()" warning.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r'.*?output shape of zoom.*', category=UserWarning)
            resized = spndim.zoom(roi, zoom=im_scale, order=0, mode='constant', cval=np.nan)

        # Sanity check. Might not be correct.
        assert max(resized.shape) == max(hw)

        # Depth values are rescaled so that this is a rigid 3D transformation.
        value_scale = im_scale * in_out_ratio
        resized *= value_scale

        h, w = resized.shape
        output = np.full(hw, np.nan)
        hstart, wstart = int((output.shape[0] - h) / 2 + 0.5), int((output.shape[1] - w) / 2 + 0.5)
        output[hstart:hstart + h, wstart:wstart + w] = resized
    except Exception as ex:
        log.warn(str(ex))
        output = np.full(hw, np.nan)

    assert np.allclose(output.shape, hw)

    if return_scale_and_center:
        assert isinstance(center, tuple) and im_scale > 0.0
        out_center = tuple(np.array(center, dtype=np.float64) / in_out_ratio)
        # Center is in output image coordinates. i.e. not affected by the size of the original image.
        return output, (value_scale, out_center)
    return output


def denormalize_depth_image(image, inv_scale, im_center_yx, depth_mean):
    depth_denorm = image * inv_scale + depth_mean
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r'.*?output shape of zoom.*', category=UserWarning)
        resized = spndim.zoom(depth_denorm, zoom=inv_scale, order=0, mode='constant', cval=np.nan)

    ret = np.full(image.shape, np.nan, dtype=image.dtype)

    center_y, center_x = im_center_yx
    h, w = resized.shape
    ystart, xstart = np.rint((center_y - h * 0.5, center_x - w * 0.5)).astype(int)
    ret[ystart:ystart + h, xstart:xstart + w] = resized

    return ret


def depth_image_centroid_and_mean_dist(image, trbl=(1, 1, -1, -1)):
    pts = cam_pts_from_ortho_depth(image, trbl=trbl)
    centroid_xyz = pts.mean(axis=0)

    # # `im_scale` is the same as the image zoom factor in rescale_and_recenter. `padding` values should match.
    # # Should be computed exactly the same way.
    # im_hw = image.shape[::-1]
    # y, x = np.where(~np.isnan(image))
    # h, w = y.max() - y.min() + 1, x.max() - x.min() + 1
    # center = int(y.min() + h / 2), int(x.min() + w / 2)
    # ystart = int(center[0] - h / 2)
    # yend = int(center[0] + h / 2)
    # xstart = int(center[1] - w / 2)
    # xend = int(center[1] + w / 2)
    # roi = image[ystart:yend, xstart:xend]
    # assert roi.shape == (h, w)
    # ratios = np.array((h, w)) / np.array(im_hw)
    # longest_axis = np.argmax(ratios)
    # im_scale = (im_hw[longest_axis] - padding * 2) / roi.shape[longest_axis] + 1e-8
    # inv_im_scale = 1.0 / im_scale

    mean_dist = la.norm(pts - centroid_xyz, axis=1, ord=2).mean()
    return tuple(centroid_xyz), mean_dist


def cam_pts_from_normalized_ortho_depth(image, target_centroid_xyz, target_mean_dist, trbl=(1, 1, -1, -1)):
    pts = cam_pts_from_ortho_depth(image, trbl=trbl)
    input_centroid_xyz = pts.mean(axis=0)
    input_mean_dist = la.norm(pts - input_centroid_xyz, axis=1, ord=2).mean()
    target_scale = target_mean_dist / input_mean_dist
    pts = (pts - input_centroid_xyz) * target_scale + target_centroid_xyz
    return pts


def normalize_depth_image(image, hw=(64, 64)):
    # Depth image is resized, recentered, and rescaled so
    # that the resulting transformation is rigid in 3D.
    padding = 0
    resized_image = rescale_and_recenter(image, hw=hw, padding=padding)

    valid = ~np.isnan(resized_image)
    if valid.sum() <= 0:
        # Empty image.
        return resized_image

    # Returns a mean-centered image.
    depth_mean = resized_image[valid].mean()
    resized_image -= depth_mean

    return resized_image


def apply44_mesh(T, mesh):
    v = apply44(T, mesh['v'])
    return {'v': v, 'f': mesh['f'].copy()}


def apply34_mesh(T, mesh):
    v = apply34(T, mesh['v'])
    return {'v': v, 'f': mesh['f'].copy()}


def find_similarity_transform(source_points, target_points):
    assert source_points.shape == target_points.shape
    source_centroid = source_points.mean(0)
    target_centroid = target_points.mean(0)

    source_recentered = source_points - source_centroid
    target_recentered = target_points - target_centroid

    scale = la.norm(target_recentered, ord=2, axis=1).max() / la.norm(source_recentered, ord=2, axis=1).max()

    u, _, vt = la.svd(source_recentered.T.dot(target_recentered))

    R = vt.T.dot(u.T)
    t = target_centroid - R.dot(source_centroid * scale)

    M = np.hstack([scale * R, t[:, None]])

    return R, scale, t, M


def normals_from_ortho_depth(depth, trbl=(1, 1, -1, -1), is_inward=False):
    assert depth.ndim == 2
    im_hw = depth.shape
    cam_h = trbl[0] - trbl[2]
    cam_w = trbl[1] - trbl[3]
    dy, dx = np.gradient(depth, cam_h / im_hw[0], cam_w / im_hw[1])
    dxyz = np.stack([-dx, dy, -np.ones_like(dx)], axis=2)
    norm = la.norm(dxyz, ord=2, axis=2, keepdims=True)
    dxyz /= norm
    if not is_inward:
        dxyz = -dxyz
    return dxyz


def oriented_pcl_from_ortho_depth(depth, mask, inward_normals=False, trbl=(1, 1, -1, -1)):
    assert depth.shape == mask.shape
    assert mask.ndim == 2
    pts_im = cam_pts_from_ortho_depth(depth=depth, trbl=trbl)
    pts = pts_im[mask.ravel()]
    normals_im = normals_from_ortho_depth(depth, trbl=trbl, is_inward=inward_normals)
    normals = normals_im[mask]
    assert pts.shape == normals.shape
    return pts, normals


def patch_radius_from_ortho_depth(depth, trbl=(1, 1, -1, -1)):
    im_hw = depth.shape
    cam_h = trbl[0] - trbl[2]
    cam_w = trbl[1] - trbl[3]
    dy, dx = np.gradient(depth)
    im_dy = cam_h / im_hw[0]
    im_dx = cam_w / im_hw[1]

    y_r = np.sqrt(np.square(dy) + np.square(im_dy))
    x_r = np.sqrt(np.square(dx) + np.square(im_dx))
    r = np.sqrt(np.square(x_r) + np.square(y_r))
    return r


def pad_image(im, pad, fill_value):
    pad_width = [(pad, pad), (pad, pad)]
    if im.ndim > 2:
        pad_width.append((0, 0))
    ret = np.pad(im, pad_width=pad_width, mode='constant', constant_values=fill_value)
    return ret


def visible_border_for_cropping(im, square=True, mask=None, padding=0, minimum_wh=None, background_value=0.0):
    # TODO(daeyun): check any off-by-one error.

    if square:
        assert minimum_wh[0] == minimum_wh[1]

    if mask is None:
        if np.isnan(background_value):
            mask = ~np.isnan(mask)
        else:
            mask = im != background_value
        if mask.ndim > 2:
            mask = np.any(mask, axis=2)
        assert np.any(mask)
    assert mask.ndim == 2

    _, x = np.where(np.any(mask, axis=0, keepdims=True))
    y, _ = np.where(np.any(mask, axis=1, keepdims=True))

    top, bottom = y.min(), y.max()
    left, right = x.min(), x.max()

    if square:
        vr = ((bottom - top) / 2.0)
        hr = ((right - left) / 2.0)
        if vr > hr:
            mid = np.ceil((right + left) / 2.0)
            left = mid - vr
            right = mid + vr
        if vr < hr:
            mid = np.ceil((bottom + top) / 2.0)
            top = mid - hr
            bottom = mid + hr

    if minimum_wh is not None:
        w = right - left + 1
        if w + 2 * padding < minimum_wh[0]:
            before = np.floor(minimum_wh[0] - (w + 2 * padding)) / 2.0
            after = np.ceil(minimum_wh[0] - (w + 2 * padding)) / 2.0
            left -= before
            right += after

        h = bottom - top + 1
        if w + 2 * padding < minimum_wh[1]:
            before = np.floor(minimum_wh[1] - (h + 2 * padding)) / 2.0
            after = np.ceil(minimum_wh[1] - (h + 2 * padding)) / 2.0
            top -= before
            bottom += after

    top, bottom, left, right = int(top), int(bottom), int(left), int(right)
    assert right - left == bottom - top

    return top, bottom, left, right


def crop_visible(im, square=True, mask=None, padding=0, minimum_wh=None, white_background=False):
    """
    If `im` is a list of images, crop based on the parameters of the first image.
    """
    return_list = False
    if isinstance(im, (list, tuple)):
        return_list = True
        images = im
        im = images[0]
    else:
        images = [im]

    if white_background:
        if np.issubdtype(im.dtype, np.float):
            background_value = 1.0
        elif np.issubdtype(im.dtype, np.uint8):
            background_value = 255
    else:
        if np.issubdtype(im.dtype, np.float):
            background_value = 0.0
        elif np.issubdtype(im.dtype, np.uint8):
            background_value = 0

    for image_i in images:
        assert image_i.shape == im.shape, image_i.shape

    top, bottom, left, right = visible_border_for_cropping(im=im, square=square, mask=mask, padding=padding, minimum_wh=minimum_wh, background_value=background_value)

    wh = im.shape[:2][::-1]

    if top < 0 or left < 0 or right >= wh[0] or right >= wh[1]:
        raise NotImplementedError('Cropping outside image coordinates: {}, {}, {}, {}'.format(top, bottom, left, right))

    if padding != 0:
        top -= padding
        bottom += padding
        left -= padding
        right += padding

    ret = [image_i[top:bottom + 1, left:right + 1].copy() for image_i in images]

    if square:
        for ret_i in ret:
            assert ret_i.shape[0] == ret_i.shape[1], ret_i.shape

    if minimum_wh is not None:
        for ret_i in ret:
            assert ret_i.shape[0] >= minimum_wh[1] and ret_i.shape[1] >= minimum_wh[0]

    if not return_list:
        assert len(ret) == 1
        return ret[0]
    return ret


