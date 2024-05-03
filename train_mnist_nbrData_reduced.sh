#!/bin/bash
#SBATCH --array=0-29
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --job-name=nbrData_mnist_reduced
#SBATCH --mem=10GB
#SBATCH -t 48:00:00
#SBATCH -D log_mnist/log_nbrData_reduced/
#SBATCH --partition=normal
#SBATCH --gres=gpu:1

hostname
echo $CUDA_VISIBLE_DEVICES
echo $CUDA_DEVICE_ORDER

for number in {0..240}
do
    python3 main_mnist.py --experiment_id=$((30*$number+${SLURM_ARRAY_TASK_ID})) --task_to_run='NbrData_Reduced' --gpu_id=0
done