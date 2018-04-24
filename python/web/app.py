import os.path
import re
import glob

from flask import Flask
from flask import send_from_directory

import scene3d.config
from scene3d import suncg_utils

app = Flask(__name__)

filenames = sorted(glob.glob('/data2/scene3d/tmp0/**/*_depth_rescaled.png', recursive=True))


@app.route('/')
def hello_world():
    width = 300

    content = """
     <table style="height:80px">
     <col width="{width}">
     <col width="{width}">
     <col width="{width}">
     <col width="{width}">
      <tr>
        <th>PBRS RGB</th>
        <th>PBRS Depth</th>
        <th>Depth (first layer)</th>
        <th>Depth (background)</th>
      </tr>
    </table> 
    """.format(width=width)

    for i, filename in enumerate(filenames):
        m = re.findall(r'renderings/([^/]+)/(\d+)_depth_rescaled.png', filename)[0]
        house_id = m[0]
        camera_id = int(m[1])

        pbrs_path = '/static_data/pbrs/mlt_v2/{}/{:06d}_mlt.png'.format(house_id, camera_id)
        pbrs_depth_path = '/static_data/scene3d/tmp0/renderings/{}/{:06d}_depth_rescaled.png'.format(house_id, camera_id)

        scene3d_depth0 = '/static_data/scene3d/tmp0/renderings/{}/{:06d}_00_vis.png'.format(house_id, camera_id)
        scene3d_depth1 = '/static_data/scene3d/tmp0/renderings/{}/{:06d}_01_vis.png'.format(house_id, camera_id)

        house_mesh_url = "/build_house_mesh/house_id/{}/house.obj".format(house_id)

        content_i = """
    #{i:05d}, house id: {house_id}, camera id: {camera_id}, <a href="{house_mesh}">[house.obj]</a>
    <br/>
    <a href="{a}"><img src="{a}" width="{width}px" /></a>
    <a href="{b}"><img src="{b}" width="{width}px" /></a>
    <a href="{c}"><img src="{c}" width="{width}px" /></a>
    <a href="{d}"><img src="{d}" width="{width}px" /></a>
    <br/>
    """.format(a=pbrs_path, b=pbrs_depth_path, c=scene3d_depth0, d=scene3d_depth1,
               width=width, house_id=house_id, camera_id=camera_id, i=i, house_mesh=house_mesh_url)

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
