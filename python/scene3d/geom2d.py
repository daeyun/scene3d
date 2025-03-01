"""
2D visualization utils
"""

import matplotlib.pyplot as pt
import PIL
import matplotlib.ticker as pt_ticker
import numpy as np
from matplotlib.patches import Polygon
import matplotlib.cm as cm
from matplotlib.patches import Rectangle

cmap_viridis = cm.get_cmap('viridis')
cmap_viridis_array = np.array([cmap_viridis(item)[:3] for item in np.arange(0, 1, 1.0 / 256)]).astype(np.float32)

# https://matplotlib.org/users/dflt_style_changes.html
line_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
               '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
               '#bcbd22', '#17becf']


def draw_triangles(triangles, ax=None, facecolor='blue', alpha=1):
    """
    :param triangles: (n,3,2) 2d triangles.
    :return:
    """
    assert triangles.shape[1] == 3
    assert triangles.shape[2] == 2

    if ax is None:
        fig = pt.figure()
        ax = fig.gca()

    for i in range(triangles.shape[0]):
        pts = np.array(triangles[i, :, :])
        p = Polygon(pts, closed=True, facecolor=facecolor, alpha=alpha)
        ax.add_patch(p)

    ax.set_xlim([triangles[:, :, 0].min(), triangles[:, :, 0].max()])
    ax.set_ylim([triangles[:, :, 1].min(), triangles[:, :, 1].max()])


def pts(xy, ax=None, markersize=10, color='r'):
    if ax is None:
        fig = pt.figure()
        ax = fig.gca()

    ax.scatter(xy[:, 0], xy[:, 1], marker='.', s=markersize, c=color)
    return ax


def draw_depth(depth: np.ma.core.MaskedArray, in_order='hw', grid_width=None, ax=None,
               clim=None, nan_color='y', cmap='gray', grid=64, show_colorbar_ticks=True, show_colorbar=True, colorbar_size='3%', simple_ticks=True,
               nan_alpha=None, alpha=None):
    if np.any(np.array(depth.shape) <= 1):
        depth = depth.squeeze()

    in_order = in_order.lower()
    if (np.array(depth.shape) > 1).sum() == 3:
        in_order = 'chw'

    if in_order != 'hw':
        depth = montage(depth, in_order=in_order, gridwidth=grid_width)

    g = cm.get_cmap(cmap, 1024 * 2)
    g.set_bad(nan_color, alpha=nan_alpha)

    if ax is None:
        fig = pt.figure()
        ax = fig.gca()

    pt.grid(True)
    for tic in ax.xaxis.get_major_ticks():
        tic.tick1On = tic.tick2On = False
        tic.label1On = tic.label2On = False
    for tic in ax.yaxis.get_major_ticks():
        tic.tick1On = tic.tick2On = False
        tic.label1On = tic.label2On = False

    if grid is None:
        pt.grid(False)
    else:
        ax.xaxis.set_major_locator(pt_ticker.MultipleLocator(base=grid))
        ax.yaxis.set_major_locator(pt_ticker.MultipleLocator(base=grid))

    ii = ax.imshow(depth, cmap=g, interpolation='nearest', aspect='equal', alpha=alpha)
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=colorbar_size, pad=0.05)
    # cb = ax.figure.colorbar(ii)
    cb = pt.colorbar(ii, cax=cax, format='%.2g')
    cb.ax.tick_params(labelsize=8)

    if simple_ticks:
        mask = ~np.isnan(depth)
        values = depth[mask]
        tick_min = values.min()
        tick_max = values.max()

        if np.isclose(tick_min, 0):
            tick_min = 0.0
        if np.isclose(tick_max, 0):
            tick_max = 0.0

        if tick_min < 0 and tick_max > 0:
            tick_values = [tick_min, 0, tick_max]
        else:
            tick_values = [tick_min, tick_max]

        cb.set_ticks(tick_values)

    if not show_colorbar_ticks:
        for label in cb.ax.xaxis.get_ticklabels():
            label.set_visible(False)
        for label in cb.ax.yaxis.get_ticklabels():
            label.set_visible(False)

    if clim is not None:
        # fig.clim(clim[0 ], clim[-1])
        cb.set_clim(clim)

    if not show_colorbar:
        cax.set_visible(False)

    return ax


def montage(images, in_order='chw', gridwidth=None, empty_value=0):
    images = np.array(images)

    assert images.ndim == len(in_order)

    target_order = 'chw'
    if in_order.lower() != target_order:
        images = images.transpose([in_order.lower().index(ch) for ch in target_order])

    images = [images[i] for i in range(images.shape[0])]
    imtype = images[0].dtype

    if gridwidth is None:
        gridwidth = int(np.ceil(np.sqrt(len(images))))
    gridheight = int(np.ceil(len(images) / gridwidth))
    remaining = gridwidth * gridheight
    rows = []
    while remaining > 0:
        rowimgs = images[:gridwidth]
        images = images[gridwidth:]
        nblank = gridwidth - len(rowimgs)
        empty_block = np.zeros(rowimgs[0].shape)
        empty_block[:] = empty_value
        rowimgs.extend([empty_block] * nblank)
        remaining -= gridwidth
        row = np.hstack(rowimgs)
        rows.append(row)
    m = np.vstack(rows)
    return m.astype(imtype)


def display_montage(images, in_order='chw', gridwidth=None, empty_value=0):
    feat_montage = montage(images, in_order=in_order, gridwidth=gridwidth, empty_value=empty_value)

    pt.imshow(feat_montage)

    ax = pt.gca()
    ax.set_xticks(np.arange(0, feat_montage.shape[1], images.shape[2]))
    ax.set_yticks(np.arange(0, feat_montage.shape[0], images.shape[1]))
    ax.grid(which='major', color='w', linestyle='-', linewidth=1)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    return ax


def apply_colormap(img, cmap_name='viridis'):
    if cmap_name != 'viridis':
        raise NotImplementedError(cmap_name)
    assert img.ndim == 2

    nan_mask = np.isnan(img)
    min_value = img[~nan_mask].min()
    max_value = img[~nan_mask].max()

    im = np.round((img - min_value) / (max_value - min_value) * 255).astype(np.uint8)
    ret = cmap_viridis_array[im]
    ret[nan_mask, :] = 1
    return ret


def draw_rectangles(rects: np.ndarray, ax=None, **kargs):
    """
    :param rects: Each row contains x1,y1,x2,y2
    """
    if ax is None:
        ax = pt.gca()

    for i in range(rects.shape[0]):
        x = min(rects[i, 0], rects[i, 2])
        y = min(rects[i, 1], rects[i, 3])
        w = abs(rects[i, 0] - rects[i, 2])
        h = abs(rects[i, 1] - rects[i, 3])
        if w * h < 0.5:
            continue  # TODO: temporary
        r = Rectangle((x, y), w, h, alpha=1, facecolor='none', fill=None, **kargs)
        ax.add_patch(r)
    ax.scatter(rects[:, 0], rects[:, 1], s=0)
    ax.scatter(rects[:, 2], rects[:, 3], s=0)
    ax.axis('equal')

    return ax


def plot_with_std(x, y_all, std=1, band_color='#FF9848', label=None):
    assert x.ndim == 1
    assert y_all.ndim == 2

    y_mean = y_all.mean(axis=0)
    y_std = y_all.std(axis=0) * std

    pt.plot(x, y_mean, label=label)
    pt.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.5, facecolor=band_color, linewidth=0, antialiased=True)
