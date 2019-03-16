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
    default_out_root = '/home/daeyuns/scene3d_out/out/scene3d'

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
        pbrs_root = '/data/daeyun-data-06/pbrs'
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
    default_out_root = '/media/ren/devRen4T/research/datasets/out/scene3d'  # This should be the directory containing checkpoint files.

elif hostname == 'daeyun-vision-lab':
    # Deprecated.
    suncg_root = '/data2/suncg_data'
    pbrs_root = '/data2/pbrs'
    scene3d_root = '/data2/scene3d'
    default_out_root = '/data3/out/scene3d'
    etn_features_root = '/data3/out/scene3d/overhead_pred'
    nyu_root = '/data3/nyu'

elif hostname == 'daeyun-lab':
    suncg_root = '/data/data/suncg_data'
    pbrs_root = '/data/data/pbrs'
    scene3d_root = '/data/data/scene3d'
    default_out_root = '/data3/out/scene3d'
    etn_features_root = '/data3/out/scene3d/overhead_pred'
    nyu_root = '/data3/nyu'

elif path.isfile(path.expanduser('~/.is_uci_vision_cluster')):
    suncg_root = '/home/daeyuns/data2/suncg_data'
    pbrs_root = '/home/daeyuns/data2/pbrs'
    scene3d_root = '/home/daeyuns/data2/scene3d'
    default_out_root = '/home/daeyuns/data3/out/scene3d'

elif path.isfile(path.expanduser('~/.is_titan_cluster')):
    suncg_root = '/extra/titansc0/daeyun/data/suncg_data'
    pbrs_root = '/extra/titansc0/daeyun/data/pbrs'
    scene3d_root = '/extra/titansc0/daeyun/data/scene3d'
    default_out_root = '/extra/titansc0/daeyun/data/out/scene3d'

# elif hostname == 'aleph0':
#     suncg_root = '/mnt/scratch1/daeyuns/data/suncg_data'
#     pbrs_root = '/mnt/scratch1/daeyuns/data/pbrs'
#     scene3d_root = '/mnt/scratch1/daeyuns/data/scene3d'
#     default_out_root = '/mnt/scratch1/daeyuns/data/out/scene3d'
#     etn_features_root = '/home/daeyuns/overhead_data'

elif hostname == 'aleph1':
    suncg_root = '/mnt/scratch1/daeyuns/data/suncg_data'
    pbrs_root = '/mnt/scratch1/daeyuns/data/pbrs'
    scene3d_root = '/mnt/scratch1/daeyuns/data/scene3d'
    default_out_root = '/home/daeyuns/out_shared/scene3d'
    etn_features_root = '/home/daeyuns/overhead_data'

elif hostname == 'aleph2':
    suncg_root = '/mnt/scratch1/daeyuns/data/suncg_data'
    pbrs_root = '/mnt/scratch1/daeyuns/data/pbrs'
    scene3d_root = '/mnt/scratch1/daeyuns/data/scene3d'
    default_out_root = '/home/daeyuns/out_shared/scene3d'
    etn_features_root = '/home/daeyuns/overhead_data'

else:
    raise RuntimeError('Unknown hostname: {}'.format(hostname))

cpp_third_party_root = path.abspath(path.join(path.dirname(__file__), '../../cpp/third_party'))
category_mapping_csv_filename = path.realpath(path.join(path.dirname(__file__), '../../resources/ModelCategoryMapping.csv'))
