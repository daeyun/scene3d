from scene3d import train_eval_pipeline_v9  #TODO
from os import path
from scene3d.dataset import v9
from scene3d import config


def main():
    dataset = v9.MultiLayerDepth(
        split=path.join(config.scene3d_root, 'v9/test_v2_subset_factored3d.txt'),
        subtract_mean=True, image_hw=(240, 320), first_n=200, rgb_scale=1.0 / 255, fields=[])

    pkl_filename = path.join(config.default_out_root, 'pr_curve_v9.pkl')
    # TODO: do not commit
    pr_eval = train_eval_pipeline_v9.PRCurveEvaluation(save_filename=pkl_filename)

    for i in range(len(dataset)):
        example = dataset[i]
        print(i, example['name'])
        pr_eval.run_evaluation(example)
        if i % 60 == 0:
            pr_eval.save()
    pr_eval.save()


if __name__ == '__main__':
    main()
