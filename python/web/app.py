import os.path
import re
import glob

from flask import Flask
from flask import send_from_directory
from flask import send_file

import scene3d.config
import random
import imageio
import io
from scene3d import suncg_utils
from scene3d import geom2d
from scene3d import pbrs_utils
from scene3d import io_utils
from scene3d.dataset import v2
import numpy as np
from scipy import misc

app = Flask(__name__)

filenames = pbrs_utils.load_pbrs_filenames()
ds_v2_0 = v2.MultiLayerDepth_0(train=True, subtract_mean=False)


@app.route('/image/scene3d/v2_0/<index>')
def make_image_v2_0(index):
    example_name, in_rgb, gt_ml_depth, count_image, d0 = ds_v2_0[int(index)]

    images = []

    divider = np.ones((240, 5, 3))

    images.append((in_rgb / 255).transpose(1, 2, 0))
    images.append(divider)
    images.append(geom2d.apply_colormap(gt_ml_depth[0]))
    images.append(divider)
    images.append(geom2d.apply_colormap(gt_ml_depth[1]))
    images.append(divider)
    images.append(geom2d.apply_colormap(gt_ml_depth[2]))
    images.append(divider)
    images.append(geom2d.apply_colormap(gt_ml_depth[0] - gt_ml_depth[1] - gt_ml_depth[2]))
    images.append(divider)
    images.append(geom2d.apply_colormap(d0))
    images.append(divider)
    images.append(geom2d.apply_colormap(count_image))

    out_image = np.concatenate(images, axis=1)

    str_io = io.BytesIO()
    misc.imsave(str_io, out_image, format='png')
    str_io.seek(0)
    return send_file(str_io, mimetype='image/png')


@app.route('/image/scene3d/<path:path>')
def make_image(path):
    filename = '/data2/scene3d/v1/renderings/{}'.format(path)
    arr = io_utils.read_array_compressed(filename)
    arr[np.isnan(arr)] = arr[~np.isnan(arr)].min()
    arr -= arr.min()
    arr /= arr.max() - arr.min()
    str_io = io.BytesIO()
    misc.imsave(str_io, arr, format='png')
    str_io.seek(0)
    return send_file(str_io, mimetype='image/png')


@app.route('/image/pbrs_depth/<path:path>')
def make_image2(path):
    filename = '/data2/pbrs/depth_v2/{}'.format(path)
    arr = imageio.imread(filename, ignoregamma=True).astype(np.float32)
    arr -= arr.min()
    arr /= arr.max() - arr.min()
    str_io = io.BytesIO()
    misc.imsave(str_io, arr, format='png')
    str_io.seek(0)
    return send_file(str_io, mimetype='image/png')


@app.route('/image/scene3d_diff/<house_id>/<camera_id>')
def make_image3(house_id, camera_id):
    arr1 = io_utils.read_array_compressed('/data2/scene3d/v1/renderings/{}/{:06d}_00.bin'.format(house_id, int(camera_id)))
    arr1[np.isnan(arr1)] = arr1[~np.isnan(arr1)].min()

    arr2 = io_utils.read_array_compressed('/data2/scene3d/v1/renderings/{}/{:06d}_01.bin'.format(house_id, int(camera_id)))
    arr2[np.isnan(arr2)] = arr2[~np.isnan(arr2)].min()

    arr = arr2 - arr1

    arr -= arr.min()
    arr /= arr.max() - arr.min()
    str_io = io.BytesIO()
    misc.imsave(str_io, arr, format='png')
    str_io.seek(0)
    return send_file(str_io, mimetype='image/png')


@app.route('/')
def main():
    return '''
    <a href="/0">0</a><br/>
    <a href="/1">1</a><br/>
    '''


@app.route('/0')
def main_0():
    width = 300

    content = """
     <table style="height:80px">
     <col width="{width}">
     <col width="{width}">
     <col width="{width}">
     <col width="{width}">
     <col width="{width}">
      <tr>
        <th>PBRS RGB</th>
        <th>PBRS Depth</th>
        <th>Depth (first layer)</th>
        <th>Depth (background)</th>
        <th>Diff</th>
      </tr>
    </table> 
    """.format(width=width)

    random.shuffle(filenames)

    for i, filename in enumerate(filenames[:20]):
        m = re.findall(r'/([^/]+)/(\d+)_mlt.png', filename)[0]
        house_id = m[0]
        camera_id = int(m[1])

        pbrs_path = '/static_data/pbrs/mlt_v2/{}/{:06d}_mlt.png'.format(house_id, camera_id)
        pbrs_depth_path = '/image/pbrs_depth/{}/{:06d}_depth.png'.format(house_id, camera_id)

        scene3d_depth0 = '/image/scene3d/{}/{:06d}_00.bin'.format(house_id, camera_id)
        scene3d_depth1 = '/image/scene3d/{}/{:06d}_01.bin'.format(house_id, camera_id)

        scene3d_diff = '/image/scene3d_diff/{}/{:06d}'.format(house_id, camera_id)

        house_mesh_url = "/build_house_mesh/house_id/{}/house.obj".format(house_id)

        content_i = """
    #{i:05d}, house id: {house_id}, camera id: {camera_id}, <a href="{house_mesh}">[house.obj]</a>
    <br/>
    <a href="{a}"><img src="{a}" width="{width}px" /></a>
    <a href="{b}"><img src="{b}" width="{width}px" /></a>
    <a href="{c}"><img src="{c}" width="{width}px" /></a>
    <a href="{d}"><img src="{d}" width="{width}px" /></a>
    <a href="{e}"><img src="{e}" width="{width}px" /></a>
    <br/>
    """.format(a=pbrs_path, b=pbrs_depth_path, c=scene3d_depth0, d=scene3d_depth1, e=scene3d_diff,
               width=width, house_id=house_id, camera_id=camera_id, i=i, house_mesh=house_mesh_url)

        content += content_i

    return content


@app.route('/1')
def main_1():
    width = 321

    content = """
    <style>
    th {{
        display: table-cell;
        vertical-align: inherit;
        font-weight: normal;
        text-align: center;
    }}
    th span {{
        display:block;
        padding-top:5px;
        font-size:80%;
    }}
    .block {{
        display:block;
        width:2500px;
    }}
    </style>
    <div class="block">
    <table style="height:80px">
    <col width="{width}">
    <col width="{width}">
    <col width="{width}">
    <col width="{width}">
    <col width="{width}">
    <col width="{width}">
    <col width="{width}">
     <tr>
       <th>RGB</th>
       <th><b>gt[0]</b><br/><span>i.e. room envelope</span></th>
       <th><b>gt[1]</b><br/><span>i.e. empty space in front of the room envelope</span></th>
       <th><b>gt[2]</b><br/><span>i.e. "filled" space in front of the empty space</span></th>
       <th><b>gt[0]-gt[1]-gt[2]</b><br/><span>i.e. "reconstructed" depth from GT</span></th>
       <th>"traditional depth"</th>
       <th>Total number of ray hits</th>
     </tr>
    </table>
    </div>
    """.format(width=width)

    indices = np.arange(len(ds_v2_0))
    random.shuffle(indices)

    for i, example_index in enumerate(indices[:20]):
        scene3d_image_combined = '/image/scene3d/v2_0/{}'.format(example_index)

        content_i = """
    example index: {example_index:07d}
    <br/>
    <a href="{a}"><img src="{a}" /></a>
    <br/>
    """.format(a=scene3d_image_combined, example_index=example_index)

        content += content_i

    return content


@app.route('/static_data/pbrs/<path:path>')
def serve_pbrs(path):
    if not path.endswith('.png'):
        return

    dirname = os.path.join(scene3d.config.pbrs_root, os.path.dirname(path))
    basename = os.path.basename(path)

    return send_from_directory(dirname, basename)


@app.route('/static_data/scene3d/<path:path>')
def serve_scene3d(path):
    if not path.endswith('.png'):
        return

    dirname = os.path.join(scene3d.config.scene3d_root, os.path.dirname(path))
    basename = os.path.basename(path)

    return send_from_directory(dirname, basename)


@app.route('/build_house_mesh/house_id/<house_id>/house.obj')
def serve_house_mesh(house_id):
    obj_filename = suncg_utils.house_obj_from_json(house_id=house_id)

    dirname = os.path.join(os.path.dirname(obj_filename))
    basename = os.path.basename(obj_filename)

    return send_from_directory(dirname, basename)
