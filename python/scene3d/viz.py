from scene3d import io_utils
from scene3d import geom2d
import cv2
import typing
import numpy as np
import matplotlib.pyplot as pt


def find_common_y_data(global_steps_list: typing.Sequence[typing.Sequence[int]], y_data_list: typing.Sequence[typing.Sequence[float]]):
    assert len(global_steps_list) == len(y_data_list)
    raise NotImplementedError()


def movingaverage(interval, window_size):
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(interval, window, 'same')


def remove_matching(a, value):
    i = a != value
    if i.any():
        a = a[i]
    return a


def colorize_model_ids(m, ref=()):
    cat = m.copy()
    if ref:
        unique_values = np.unique(np.concatenate([item.ravel() for item in ref], axis=0))
    else:
        unique_values = np.unique(cat)
    unique_values = remove_matching(unique_values, 65536)
    unique_values = remove_matching(unique_values, 0)
    argsort = np.argsort(unique_values)
    cat_mappings = dict(zip(unique_values, argsort))
    cat_mappings[0] = np.nan
    cat = cat.astype(np.float64)
    for i in range(cat.size):
        cat.flat[i] = cat_mappings[cat.flat[i]]
    pt.imshow(cat)
    pt.clim(argsort.min(), argsort.max())
    return cat


def display_example(house_id, cam_i, outdir):
    io_utils.ensure_dir_exists(outdir)

    basepath = '/data2/scene3d/v8/renderings/{}/{}'.format(house_id, cam_i)
    rgb_path = '/data2/pbrs/mlt_v2/{}/{}_mlt.png'.format(house_id, cam_i)

    rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)

    a = io_utils.read_array_compressed('{}_ldi.bin'.format(basepath), dtype=np.float32)
    b = io_utils.read_array_compressed('{}_model.bin'.format(basepath), dtype=np.uint16)
    c = io_utils.read_array_compressed('{}_ldi-o.bin'.format(basepath), dtype=np.float32)
    d = io_utils.read_array_compressed('{}_model-o.bin'.format(basepath), dtype=np.uint16)
    e = io_utils.read_array_compressed('{}_oit.bin'.format(basepath), dtype=np.float32)
    f = io_utils.read_array_compressed('{}_n.bin'.format(basepath), dtype=np.float32)

    pt.figure()
    pt.title('RGB')
    pt.imshow(rgb)
    pt.xticks([])
    pt.yticks([])
    pt.savefig('{}/0.png'.format(outdir), bbox_inches='tight')

    pt.figure()
    pt.title('depth')
    pt.imshow(a[0])
    pt.xticks([])
    pt.yticks([])
    pt.savefig('{}/1.png'.format(outdir), bbox_inches='tight')

    pt.figure()
    pt.title('depth - instance exit rule')
    pt.imshow(a[1])
    pt.xticks([])
    pt.yticks([])
    pt.savefig('{}/2.png'.format(outdir), bbox_inches='tight')

    pt.figure()
    pt.title('depth - last exit rule')
    pt.imshow(a[2])
    pt.xticks([])
    pt.yticks([])
    pt.savefig('{}/3.png'.format(outdir), bbox_inches='tight')

    pt.figure()
    pt.title('depth - disoccluded background')
    pt.imshow(a[3])
    pt.xticks([])
    pt.yticks([])
    pt.savefig('{}/4.png'.format(outdir), bbox_inches='tight')

    pt.figure()
    colorize_model_ids(b[0], [b[0], b[2], d[0], d[2]])
    pt.xticks([])
    pt.yticks([])
    pt.title('model id (category)')
    pt.savefig('{}/5.png'.format(outdir), bbox_inches='tight')

    pt.figure()
    colorize_model_ids(b[2], [b[0], b[2], d[0], d[2]])
    pt.xticks([])
    pt.yticks([])
    pt.title('model id (category) - last exit')
    pt.savefig('{}/6.png'.format(outdir), bbox_inches='tight')

    pt.figure()
    pt.title('thickness - inward normal direction, instance exit')
    pt.imshow(e)
    pt.xticks([])
    pt.yticks([])
    pt.savefig('{}/7.png'.format(outdir), bbox_inches='tight')

    pt.figure()
    pt.title('surface normal')
    normal = f.transpose([1, 2, 0]) * 0.5 + 0.5
    normal[~np.isfinite(normal)] = 1
    pt.imshow(normal)
    pt.xticks([])
    pt.yticks([])
    pt.savefig('{}/8.png'.format(outdir), bbox_inches='tight')

    pt.figure()
    pt.title('height map')
    pt.imshow(c[0])
    pt.xticks([])
    pt.yticks([])
    pt.colorbar()
    pt.savefig('{}/9.png'.format(outdir), bbox_inches='tight')

    pt.figure()
    pt.title('height map - instance exit')
    pt.imshow(c[1])
    pt.xticks([])
    pt.yticks([])
    pt.colorbar()
    pt.savefig('{}/10.png'.format(outdir), bbox_inches='tight')

    pt.figure()
    pt.title('height map - last exit')
    pt.imshow(c[2])
    pt.xticks([])
    pt.yticks([])
    pt.colorbar()
    pt.savefig('{}/11.png'.format(outdir), bbox_inches='tight')

    pt.figure()
    colorize_model_ids(d[0], [b[0], b[2], d[0], d[2]])
    pt.xticks([])
    pt.yticks([])
    pt.title('overhead model id (category)')
    pt.savefig('{}/12.png'.format(outdir), bbox_inches='tight')

    pt.figure()
    colorize_model_ids(d[2], [b[0], b[2], d[0], d[2]])
    pt.xticks([])
    pt.yticks([])
    pt.title('overhead model id (category) - last exit')
    pt.savefig('{}/13.png'.format(outdir), bbox_inches='tight')

    pt.figure()
    aabb = io_utils.read_txt_array('{}_aabb.txt'.format(basepath))
    ax = None
    ax = geom2d.draw_rectangles(aabb[None, 0, [0, 2, 3, 5]], ax=ax, edgecolor='black', linewidth=3)
    ax = geom2d.draw_rectangles(aabb[None, 1, [0, 2, 3, 5]], ax=ax, edgecolor='red', linestyle='--', linewidth=1)
    ax = geom2d.draw_rectangles(aabb[None, 2, [0, 2, 3, 5]], ax=ax, edgecolor='blue', linestyle='--', linewidth=1)
    ax = geom2d.draw_rectangles(aabb[None, 3, [0, 2, 3, 5]], ax=ax, edgecolor='green', linestyle='--', linewidth=1)
    pt.gca().invert_yaxis()
    pt.title('overhead bounding boxes')
    pt.savefig('{}/14.png'.format(outdir), bbox_inches='tight')

    pt.close('all')
