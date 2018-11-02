import multiprocessing

from scene3d import train_eval_pipeline
import argparse

parser = argparse.ArgumentParser(description='training')
parser.add_argument('--batch_size', type=int, default=3)
parser.add_argument('--num_data_workers', type=int, default=5)
parser.add_argument('--save_dir', type=str, default='/data2/out/scene3d/v1/default')
parser.add_argument('--experiment', type=str, default='multi-layer')
parser.add_argument('--max_epochs', type=int, default=500)
parser.add_argument('--save_every', type=int, default=2000)
parser.add_argument('--first_n', type=int, default=0)
parser.add_argument('--model', type=str, default='unet_v0')
parser.add_argument('--load_checkpoint', type=str, default='')
parser.add_argument('--use_cpu', type=bool, default=False)
args = parser.parse_args()


def main():
    trainer = train_eval_pipeline.Trainer(args)
    trainer.train()


if __name__ == '__main__':
    # multiprocessing.set_start_method('spawn')
    main()
