#!/bin/bash
#SBATCH --array=0-23
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --job-name=nbrData_mnist
#SBATCH --mem=10GB
#SBATCH -t 02:00:00
#SBATCH -D log_mnist/log_nbrData/
#SBATCH --partition=normal
#SBATCH --gres=gpu:1

hostname
echo $CUDA_VISIBLE_DEVICES
echo $CUDA_DEVICE_ORDER

for number in {0..1}
do
    python3 main_mnist.py --experiment_id=$((24*$number+${SLURM_ARRAY_TASK_ID})) --task_to_run='NbrData' --gpu_id=0
done
