from scene3d import train_eval_pipeline
from scene3d.dataset import v8


def main():
    dataset = v8.MultiLayerDepth(
        split='/data2/scene3d/v8/test_v2_subset_factored3d.txt',
        subtract_mean=True, image_hw=(240, 320), first_n=None, rgb_scale=1.0 / 255, fields=[])

    pkl_filename = '/data3/out/scene3d/pr_curve.pkl'
    pr_eval = train_eval_pipeline.PRCurveEvaluation(save_filename=pkl_filename)

    for i in range(500):
        example = dataset[i]
        print(i, example['name'])
        pr_eval.run_evaluation(example)
        if i % 5 == 0:
            pr_eval.save()


if __name__ == '__main__':
    main()
