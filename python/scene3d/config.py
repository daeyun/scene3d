from os import path
import socket

"""
pbrs_root:
    Directory containing PBRS data.
    Subdirectories include: camera_v2, mlt_v2, ...
suncg_root:
    Directory containing SUNCG data.
    Subdirectories include: house, object, room, ...
scene3d:
    This project's main data directory. May contain symlinks.
    Subdirectories include: v2, v8/renderings, ...
"""

hostname = socket.gethostname()

if path.isfile(path.expanduser('~/.is_chase_ci')):
    if hostname == 'daeyun-01':
        # suncg_root = '/data/daeyun-data-02/suncg_data'
        # pbrs_root = '/data/daeyun-data-02/data2/pbrs'
        suncg_root = '/data/ucicompvishomes/daeyuns/data2/suncg_data'
        pbrs_root = '/data/ucicompvishomes/daeyuns/data2/pbrs'
        scene3d_root = '/data/daeyun-data-02/scene3d'

    elif hostname == 'daeyun-02':
        suncg_root = '/data/daeyun-data-01/suncg_data'
        pbrs_root = '/data/daeyun-data-01/pbrs'
        scene3d_root = '/data/daeyun-data-01/scene3d'

    elif hostname == 'daeyun-03':
        suncg_root = '/data/daeyun-data-03/suncg_data'
        pbrs_root = '/data/daeyun-data-03/pbrs'
        scene3d_root = '/data/daeyun-data-03/scene3d'

    elif hostname == 'daeyun-04':
        suncg_root = '/data/daeyun-data-04/suncg_data'
        pbrs_root = '/data/daeyun-data-04/pbrs'
        scene3d_root = '/data/daeyun-data-04/scene3d'

    elif hostname == 'daeyun-05':
        suncg_root = '/data/daeyun-data-05/suncg_data'
        pbrs_root = '/data/daeyun-data-05/pbrs'
        scene3d_root = '/data/daeyun-data-05/scene3d'

    elif hostname == 'daeyun-06':
        suncg_root = '/data/daeyun-data-06/suncg_data'
        pbrs_root = '/data/daeyun-data-06/data2/pbrs'
        scene3d_root = '/data/daeyun-data-06/scene3d'

    elif hostname == 'daeyun-07':
        suncg_root = '/data/daeyun-data-07/suncg_data'
        pbrs_root = '/data/daeyun-data-07/pbrs'
        scene3d_root = '/data/daeyun-data-07/scene3d'
    else:
        raise RuntimeError('Unknown hostname: {}'.format(hostname))

elif hostname == 'ren-ubuntu':
    suncg_root = '/media/ren/devRen4T/research/datasets/suncg'
    pbrs_root = '/media/ren/devRen4T/research/datasets/pbrs'
    scene3d_root = '/media/ren/devRen4T/research/datasets/scene3d'

elif hostname == 'daeyun-vision-lab':
    suncg_root = '/data2/suncg_data'
    pbrs_root = '/data2/pbrs'
    scene3d_root = '/data2/scene3d'

elif path.isfile(path.expanduser('~/.is_uci_vision_cluster')):
    suncg_root = '/home/daeyuns/data2/suncg_data'
    pbrs_root = '/home/daeyuns/data2/pbrs'
    scene3d_root = '/home/daeyuns/data2/scene3d'

else:
    raise RuntimeError('Unknown hostname: {}'.format(hostname))

cpp_third_party_root = path.abspath(path.join(path.dirname(__file__), '../../cpp/third_party'))
category_mapping_csv_filename = path.realpath(path.join(path.dirname(__file__), '../../resources/ModelCategoryMapping.csv'))
