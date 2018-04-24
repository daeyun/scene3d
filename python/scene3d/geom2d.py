"""
2D visualization utils
"""

import matplotlib.pyplot as pt
import matplotlib.ticker as pt_ticker
import numpy as np
from matplotlib.patches import Polygon
import matplotlib.cm as cm


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
