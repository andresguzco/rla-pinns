#!/bin/bash
#SBATCH --partition=rtx6000
#SBATCH --qos=m4

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --array=1-20%17

echo "[DEBUG] Host name: " `hostname`

source  ~/miniforge3/etc/profile.d/conda.sh
conda activate rla_pinns

wandb agent --count 1 rla-pinns/exp12_poisson5d_large_fixedlr/x9u551s6