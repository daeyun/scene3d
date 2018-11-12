from scene3d import train_eval_pipeline
from os import path
import argparse

parser = argparse.ArgumentParser(description='training')
parser.add_argument('--checkpoint_filename', type=str, default='')
parser.add_argument('--set_experiment_name', type=str, default='')
args = parser.parse_args()


def main():
    checkpoint_filename = args.checkpoint_filename
    assert path.isfile(checkpoint_filename), checkpoint_filename

    pytorch_model, optimizer, metadata_dict, frozen_model = train_eval_pipeline.load_checkpoint(checkpoint_filename, use_cpu=True)

    print(metadata_dict)

    if args.set_experiment_name:
        print('new experiment name is {}'.format(args.set_experiment_name))
        metadata_dict['experiment_name'] = args.set_experiment_name
        print(metadata_dict)
        train_eval_pipeline.save_checkpoint(path.dirname(checkpoint_filename), pytorch_model=pytorch_model, optimizer=optimizer, metadata=metadata_dict)


if __name__ == '__main__':
    main()
