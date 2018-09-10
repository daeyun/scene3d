import time

import numpy as np
import matplotlib.pyplot as pt
import cv2

from scene3d import io_utils
from scene3d import epipolar


def main():
    ldi = io_utils.read_array_compressed('../resources/depth_render/0004d52d1aeeb8ae6de39d6bd993e992/000003_ldi.bin')

    front_depth = ldi[0].copy()
    back_depth = ldi[1].copy()

    in_rgb = cv2.cvtColor(cv2.imread('../resources/rgb/0004d52d1aeeb8ae6de39d6bd993e992/000003_mlt.png'), cv2.COLOR_BGR2RGB)
    in_rgb = cv2.resize(in_rgb, dsize=(front_depth.shape[1], front_depth.shape[0]), interpolation=cv2.INTER_LINEAR)
    in_rgb = (in_rgb / 255.0).astype(np.float32)

    camera_filename = '../resources/depth_render/0004d52d1aeeb8ae6de39d6bd993e992/000003_cam.txt'

    start = time.time()
    out = epipolar.feature_transform(in_rgb, front_depth, back_depth, camera_filename, 300, 300)
    print('elapsed: {}'.format(time.time() - start))
    start = time.time()
    out = epipolar.feature_transform(in_rgb, front_depth, back_depth, camera_filename, 300, 300)
    print('elapsed: {}'.format(time.time() - start))

    pt.imshow(out)
    pt.show()


if __name__ == '__main__':
    main()
