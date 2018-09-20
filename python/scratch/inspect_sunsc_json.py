import collections
import glob
import ujson
import re
import shutil
from os import path

import imageio
import matplotlib.cm
import matplotlib.pyplot as pt
import numpy as np

from scene3d import io_utils
from scene3d import config
from scene3d import render_depth
from scene3d import suncg_utils

# %%


csv_filename = config.category_mapping_csv_filename

import re
import csv

with open(csv_filename) as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    rows = list(spamreader)

class_names = []
model_ids = []
for row in rows[1:]:
    m1 = re.match(r'door|.*?_door|door.|.*?window.*?', row[2])
    m2 = re.match(r'door|.*?_door|door.|.*?window.*?', row[3])
    if m1 or m2:
        class_names.append(row[2])
        class_names.append(row[3])
        model_ids.append(row[1])
model_ids = sorted(model_ids)

print('collected class names: ', set(class_names))
assert len(set(model_ids)) == len(model_ids)
print(len(model_ids))

print('doors_and_windows = {{{}}}'.format(','.join(['"{}"'.format(item) for item in model_ids])))

class_names = []
model_ids = []
for row in rows[1:]:
    m1 = re.match(r'.*?person.*?|.*?plant.*?', row[2])
    m2 = re.match(r'.*?person.*?|.*?plant.*?', row[3])
    if m1 or m2:
        class_names.append(row[2])
        class_names.append(row[3])
        model_ids.append(row[1])
model_ids = sorted(model_ids)

print('collected class names: ', set(class_names))
assert len(set(model_ids)) == len(model_ids)
print(len(model_ids))

print('plants_and_people = {{{}}}'.format(','.join(['"{}"'.format(item) for item in model_ids])))

# %%


# %%

house_id = '0004dd3cb11e50530676f77b55262d38'
house_file = path.join(config.suncg_root, 'house/{}/house.json'.format(house_id))

with open(house_file, 'r') as f:
    house_json = ujson.load(f)

house_obj_filename = suncg_utils.house_obj_from_json(house_id=house_id)

# %%

tmp_render_out_dir = '/tmp/scene3d/renderings'
source_room_camera_file = '/data2/pbrs/camera_v2/{}/room_camera.txt'.format(house_id)

print(source_room_camera_file)
print(tmp_render_out_dir)

output_files = render_depth.run_render(obj_filename=house_obj_filename, camera_filename=source_room_camera_file, out_dir=tmp_render_out_dir)
