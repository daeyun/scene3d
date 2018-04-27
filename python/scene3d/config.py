from os import path

suncg_root = '/data2/suncg_data'

# Directory containing PBRS data.
# Subdirectories include: camera_v2, mlt_v2, ...
pbrs_root = '/data2/pbrs'

scene3d_root = '/data2/scene3d'

cpp_third_party_root = path.abspath(path.join(path.dirname(__file__), '../../cpp/third_party'))
