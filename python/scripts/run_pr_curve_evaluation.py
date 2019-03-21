from scene3d import train_eval_pipeline_v9  # TODO
from os import path
from scene3d.dataset import v9
from scene3d import config

skipped = [
    'e5e5e9fb2f46af947a72644a9c3fff51/000008',
    'cb1ab3737e580172555b6318c5c3fc62/000037',
    '3d7a0d0b5f35afaf8fa9f1e70cd5a0db/000015',
]


def main():
    dataset = v9.MultiLayerDepth(
        split=[
            # path.join(config.scene3d_root, 'v9/test_subset_factored3d.txt')
            path.join(config.scene3d_root, 'v9/test_subset_factored3d__shuffled_0001_of_0009.txt'),  # sharded for running on multiple machines
        ],
        subtract_mean=True, image_hw=(240, 320), first_n=None, rgb_scale=1.0 / 255, fields=[])

    pkl_filename = path.join(config.default_out_root, 'v9_pr_curves/pr_curve_v9__1.pkl')
    # TODO: do not commit
    pr_eval = train_eval_pipeline_v9.PRCurveEvaluation(save_filename=pkl_filename)

    for i in range(len(dataset)):
        example = dataset[i]
        print(i, example['name'])
        if example['name'] in skipped:
            print('skipped')
            continue
        pr_eval.run_evaluation(example)
        if i % 2 == 0:
            pr_eval.save()
    pr_eval.save()


if __name__ == '__main__':
    main()
