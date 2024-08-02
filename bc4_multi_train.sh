#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=5-00:00:00
#SBATCH --mem=15GB
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --job-name=open_swin_train_l1
#SBATCH --account=coms031144
#SBATCH --output=slurm_jobs/swin_opening_train_l1.out

. ~/initConda.sh

conda activate strike

python multi_param_training.py --network "swin" --imageSize 512 --lossFunction "L1" --strikeWeight 1.0 --openingWeight 1.0 --learningParam "strike" "opening"
