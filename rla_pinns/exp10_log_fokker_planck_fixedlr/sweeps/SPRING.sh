#!/bin/bash
#SBATCH --partition=rtx6000
#SBATCH --qos=m2

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --array=1-50%17

echo "[DEBUG] Host name: " `hostname`

source  ~/miniforge3/etc/profile.d/conda.sh
conda activate rla_pinns

wandb agent --count 1 rla-pinns/exp10_log_fokker_planck_fixedlr/iya1zu96