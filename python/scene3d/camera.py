import numpy as np
import scipy.linalg as la
import hashlib


class Camera(object):
    def __init__(self, R, t, s, sRt_scale=None, is_world_to_cam=True):
        """
        :param R: (3,3) Rotation matrix. Must be orthogonal.
        :param t: (3,) Translation vector.
        :param s: Pre-transformation scale. Applied before rotating and translating.
        :param sRt_scale: Post-transformation scale. Both R and t are scaled by this amount. Not the same as `s`.
            Changes t and s.
        :param is_world_to_cam:
        """
        if not is_world_to_cam:
            raise NotImplementedError()
        self.is_world_to_cam = is_world_to_cam
        self.R = R
        self.t = t[:, None] if len(t.shape) == 1 else t
        self.s = s

        if sRt_scale is None:
            self.sRt_scale = 1.0
        else:
            assert isinstance(sRt_scale, float)
            self.sRt_scale = sRt_scale

        # if self.sRt_scale is not None:
        #     assert isinstance(self.sRt_scale, float)
        #     self.t *= self.sRt_scale
        #     self.s *= self.sRt_scale

        self.R_inv, self.t_inv, self.s_inv = self._inverse(self.R, self.t, self.s)

        self.pos = self.t_inv.ravel()

        self.viewdir = self.cam_to_world(np.array([0, 0, -1]).reshape(-1, 3)).ravel() - self.pos
        self.viewdir /= la.norm(self.viewdir)

        self.up_vector = self.cam_to_world(np.array([0, 1, 0]).reshape(-1, 3)).ravel() - self.pos
        self.up_vector /= la.norm(self.up_vector)

    def Rt(self):
        return np.hstack((self.R, self.t))

    def Rt_inv(self):
        return np.hstack((self.R_inv, self.t_inv))

    def sRt(self):
        return np.hstack((self.s * self.R, self.t))

    def sRt_inv(self):
        return np.hstack((self.s_inv * self.R_inv, self.t_inv))

    def world_to_cam(self, xyzs):
        return self.sRt().dot(self._hom(xyzs).T).T * self.sRt_scale

    def cam_to_world(self, xyzs):
        return self.sRt_inv().dot(self._hom(xyzs / self.sRt_scale).T).T

    def sRt_hash(self):
        """
        :return: SHA1 hash of sRt matrix ignoring very small numerical differences.
        """
        sRt_flat = np.ascontiguousarray(self.sRt(), dtype=np.float64).ravel()
        values = ['{:.4f}'.format(item) for item in sRt_flat]

        # '-0.0000' and '0.0000' are replaced by '0'
        values = ['0' if float(value) == 0.0 else value for value in values]
        values_string = ','.join(values)

        sha1 = hashlib.sha1()
        sha1.update(values_string.encode('utf8'))
        ret = sha1.hexdigest()

        return ret

    @classmethod
    def _inverse(cls, R: np.ndarray, t: np.ndarray, s: float = 1):
        cls._check_orthogonal(R)
        if len(t.shape) == 1:
            t = t[:, None]

        R_inv = R.T
        t_inv = -R.T.dot(t) / s
        s_inv = 1.0 / s

        return R_inv, t_inv, s_inv

    @classmethod
    def _hom(cls, pts):
        assert pts.shape[1] in [2, 3]
        return np.hstack((pts, np.ones((pts.shape[0], 1))))

    @classmethod
    def _hom_inv(cls, pts):
        assert pts.shape[1] in [3, 4]
        return pts[:, :-1] / pts[:, -1, None]

    @classmethod
    def _check_orthogonal(cls, R, eps=1e-4):
        assert la.norm(R.T - la.inv(R), 2) < eps


class OrthographicCamera(Camera):
    @classmethod
    def from_Rt(cls, Rt: np.ndarray, s: float = 1.0, trbl=(1, 1, -1, -1), wh=(64, 64), sRt_scale=None, is_world_to_cam=True):
        return OrthographicCamera(Rt[:, :3], Rt[:, 3], s=s, trbl=trbl, wh=wh, is_world_to_cam=is_world_to_cam, sRt_scale=sRt_scale)

    @classmethod
    def identity(cls, wh):
        return cls.from_Rt(Rt=np.eye(3, 4, dtype=np.float64), s=1.0, wh=wh)

    def __init__(self, R: np.ndarray, t: np.ndarray, s: float = 1.0, trbl=(1, 1, -1, -1),
                 wh=(64, 64), sRt_scale=None, is_world_to_cam=True):
        super().__init__(R, t, s, sRt_scale=sRt_scale, is_world_to_cam=is_world_to_cam)
        self.trbl = trbl
        self.wh = wh

    def world_to_image(self, xyzs, filter_invalid=False, return_z=False):
        campts = self.world_to_cam(xyzs)
        xy, valid_inds = self.cam_to_image(campts, filter_invalid=False)
        depth = campts[:, 2]

        if filter_invalid:
            xy = xy[valid_inds]
            depth = depth[valid_inds]

        if return_z:
            return xy, valid_inds, depth
        return xy, valid_inds

    def cam_to_image(self, xyzs, filter_invalid=False):
        xy = xyzs[:, :2].copy()
        xy[:, 1] *= -1
        xy[:, 0] /= self.trbl[1] - self.trbl[3]
        xy[:, 1] /= self.trbl[0] - self.trbl[2]
        xy[:, 0] *= self.wh[0] - 1
        xy[:, 1] *= self.wh[1] - 1
        xy[:, 0] += (self.wh[0] - 1) / 2
        xy[:, 1] += (self.wh[1] - 1) / 2
        xy = np.round(xy)
        xy = xy.astype(np.int32)

        valid_inds = np.all((xy >= 0) & (xy < self.wh), axis=1)

        if filter_invalid:
            return xy[valid_inds], np.where(valid_inds)[0]
        return xy, np.where(valid_inds)[0]


class PerspectiveCamera(Camera):
    def __init__(self, R: np.ndarray, t: np.ndarray, K: np.ndarray = None, s: float = 1.0, sRt_scale=None, is_world_to_cam=True):
        assert R.shape == (3, 3)
        assert K.shape == (3, 3)
        if len(t.shape) == 1:
            t = t[None, :]
        assert t.shape == (3, 1)

        if sRt_scale is not None:
            raise NotImplementedError()

        self._check_orthogonal(R)

        self.K = K
        self.K_inv = la.inv(K)

        # self.{R, t, s} will always be world_to_cam regardless of initialization.
        if is_world_to_cam:
            self.R, self.t, self.s = R, t, s
            self.R_inv, self.t_inv, self.s_inv = self._inverse(R, t, s)
        else:
            self.R_inv, self.t_inv, self.s_inv = R, t, s
            self.R, self.t, self.s = self._inverse(R, t, s)

    def __str__(self):
        return 'R:\n{}\nt:\n{}\nK:\n{}'.format(self.R, self.t, self.K)

    def _check_orthogonal(self, R, eps=1e-4):
        assert la.norm(R.T - la.inv(R), 2) < eps

    def _inverse(self, R: np.ndarray, t: np.ndarray, s: float = 1):
        raise NotImplementedError  # TODO
        self._check_orthogonal(R)
        if len(t.shape) == 1:
            t = t[None, :]

        R_inv = R.T
        t_inv = -R.T.dot(t) / s
        s_inv = 1.0 / s

        return R_inv, t_inv, s_inv

    def _hom(self, pts):
        assert pts.shape[1] in [2, 3]
        return np.hstack((pts, np.ones((pts.shape[0], 1))))

    def _hom_inv(self, pts):
        assert pts.shape[1] in [3, 4]
        return pts[:, :-1] / pts[:, -1, None]

    def position(self):
        return (-self.R.T.dot(self.t) / self.s).ravel()

    def image_to_world(self, xys):
        if xys.shape[1] == 2:
            xys = self._hom(xys)
        return self.cam_to_world(self.K_inv.dot(xys.T).T)

    def cam_to_world(self, xyzs):
        sRt_inv = np.hstack((self.s_inv * self.R_inv, self.t_inv))
        xyzs = self._hom(xyzs)
        return sRt_inv.dot(xyzs.T).T

    def world_to_cam(self, xyzs):
        sRt = np.hstack((self.s * self.R, self.t))
        return sRt.dot(self._hom(xyzs).T).T

    def world_to_image(self, xyzs):
        # return self._hom_inv(self.K.dot(self.world_to_cam(xyzs).T).T)
        P = self.projection_mat34()
        return self._hom_inv(P.dot(self._hom(xyzs).T).T)

    def projection_mat34(self):
        return self.K.dot(np.hstack((self.s * self.R, self.t)))


def camera_fixation_centroid(cameras):
    lines = []
    for cam in cameras:
        pos = cam.position()
        imcenter = cam.image_to_world(np.array([[cam.K[0, 2], cam.K[1, 2], 1]]))
        v = (pos - imcenter)
        lines.append(np.hstack((pos[None, :], v / la.norm(v, 2))))

    A = []
    b = []
    for line in lines:
        A.append([[line[0, 4], -line[0, 3], 0], [line[0, 5], 0, -line[0, 3]]])
        b.extend([la.det(np.vstack((line[0, (0, 1)], line[0, (3, 4)]))),
                  la.det(np.vstack((line[0, (0, 2)], line[0, (3, 5)])))])
    A = np.vstack(A)
    b = np.array(b)[:, None]
    x = la.lstsq(A, b)[0].ravel()
    return x
