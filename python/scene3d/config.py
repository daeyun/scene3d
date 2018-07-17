from os import path
import socket

cur_hostname = socket.gethostname()
if (cur_hostname=='ren-ubuntu')
    suncg_root = '/media/ren/devRen4T/research/datasets/suncg'

    # Directory containing PBRS data.
    # Subdirectories include: camera_v2, mlt_v2, ...
    pbrs_root = '/media/ren/devRen4T/research/datasets/pbrs'

    scene3d_root = '/media/ren/devRen4T/research/datasets/scene3d'

    cpp_third_party_root = path.abspath(path.join(path.dirname(__file__), '../../cpp/third_party'))

else:
    suncg_root = '/data2/suncg_data'

    # Directory containing PBRS data.
    # Subdirectories include: camera_v2, mlt_v2, ...
    pbrs_root = '/data2/pbrs'

    scene3d_root = '/data2/scene3d'

    cpp_third_party_root = path.abspath(path.join(path.dirname(__file__), '../../cpp/third_party'))
