import numpy as np
import PIL.Image
import IPython.display
import textwrap
import numpy as np

import IPython.display
import matplotlib.pyplot as pt
import PIL.Image
from IPython.core import display


def html(html_content):
    display.display(display.HTML(textwrap.dedent(html_content.strip())))


def textarea(content='', width='100%', height='100px'):
    html(html_content='''
        <textarea style="border:1px solid #bbb; padding:0; margin:0; line-height:1; font-size:75%; font-family:monospace; width:{width}; height:{height}">{content}</textarea>
    '''.format(width=width, height=height, content=content))


def horizontal_line(color='#bbb', size=1):
    html(html_content='''
        <hr style="border-color: {color}; color: {color}; size={size}">
    '''.format(color=color, size=size))


def quick_show_rgb(arr: np.ndarray):
    if arr.ndim == 3 and arr.shape[0] == 3:
        arr = arr.transpose(1, 2, 0)
    if arr.dtype == np.float32:
        arr = (arr * 255).astype(np.uint8)

    im = PIL.Image.fromarray(arr)
    IPython.display.display(im)


def remove_ticks(ax=None):
    if ax is None:
        ax = pt.gca()
    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])
    return ax

