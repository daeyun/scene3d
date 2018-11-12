from os import path
from scene3d import config
from scene3d import io_utils
from scene3d.dataset import v8


def main():
    split_filename = path.join(config.scene3d_root, 'v8/all_v2.txt')
    filename_prefixes = io_utils.read_lines_and_strip(split_filename)

    for example_name in filename_prefixes:
        print(example_name, end=' ', flush=True)
        filename = v8.find_etn_filename(example_name)
        assert path.getsize(filename) > 100, (filename, path.getsize(filename))
        print(filename)

    print('OK')


if __name__ == '__main__':
    main()
