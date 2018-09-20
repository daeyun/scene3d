from scene3d import io_utils
from scene3d import config

sorted_nyu40_names = sorted(list(set(io_utils.get_column_from_csv(config.category_mapping_csv_filename, column_name='nyuv2_40class'))))


def nyu40_name_from_index(index: int):
    return sorted_nyu40_names[index]


def nyu40_index_from_name(name: str):
    return sorted_nyu40_names.index(name)
