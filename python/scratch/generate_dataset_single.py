from scene3d import pbrs_utils
from scene3d import log
from scene3d import render_depth
from scene3d import suncg_utils
from scene3d import io_utils
from scene3d import config
from os import path

# house_id = '16cf44a3f11809f4098d7a306eadcba4'

house_id = '5b82c4db5fcc58525966657f946b00ca'

tmp_out_root = '/mnt/ramdisk/scene3d'
thread_id = 999


def main():
    camera_ids = pbrs_utils.camera_ids(house_id)
    log.info('House is: {}'.format(house_id))
    log.info('Camera ids: {}'.format(camera_ids))

    tmp_house_obj_file = path.join(tmp_out_root, 'house_obj_default/{}/house.obj'.format(thread_id))
    obj_filename, new_house_json_filename = suncg_utils.house_obj_from_json(house_id=house_id, out_file=tmp_house_obj_file, return_house_json_filename=True)
    source_room_camera_file = path.join(config.pbrs_root, 'camera_v2/{}/room_camera.txt'.format(house_id))

    log.info('{}'.format(tmp_house_obj_file))
    log.info('{}'.format(new_house_json_filename))

    # make new camera file. after filtering out ones ignored provided by pbrs.
    out_room_camera_file = path.join(tmp_out_root, '{}/room_camera.txt'.format(thread_id))
    io_utils.ensure_dir_exists(path.dirname(out_room_camera_file))
    with open(source_room_camera_file, 'r') as f:
        lines = f.readlines()

    new_camera_file_content = ''.join([lines[int(cid)] for cid in camera_ids]).strip()
    with open(out_room_camera_file, 'w') as f:
        f.write(new_camera_file_content)
    log.info('{}'.format(out_room_camera_file))

    tmp_render_out_dir = path.join(tmp_out_root, '{}/renderings'.format(thread_id))
    log.info('{}'.format(tmp_render_out_dir))

    output_files = render_depth.run_render(obj_filename=obj_filename, json_filename=new_house_json_filename, camera_filename=out_room_camera_file,
                                           out_dir=tmp_render_out_dir, hw=(480 // 2, 640 // 2))


if __name__ == '__main__':
    main()
