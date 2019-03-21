import argparse
from scene3d import shards

parser = argparse.ArgumentParser(description='generate overhead features')
parser.add_argument('--split_file', type=str)
parser.add_argument('--source_dirname', type=str)
parser.add_argument('--target_dirname', type=str)
args = parser.parse_args()


def main():
    shards.move_shard(args.split_file, args.source_dirname, args.target_dirname)


if __name__ == '__main__':
    main()
