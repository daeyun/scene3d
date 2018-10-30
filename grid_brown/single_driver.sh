#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -r y
#$ -o ../../log/$JOB_ID.$TASK_ID.out
#$ -e ../../log/$JOB_ID.$TASK_ID.err

echo "JOB ID: " $JOB_ID
echo "SGE TASK ID: "$SGE_TASK_ID

# run
cd ../python
conda activate scene3d
echo "activated"
nvidia-smi
python scripts/train.py --batch_size=24 --save_dir=/nbu/liv/ren/git/scene3d/results/single/ --experiment=v8-single_layer_depth  --model=unet_v2
deactivate
echo "all done"
exit


