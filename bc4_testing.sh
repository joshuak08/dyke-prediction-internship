#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=6:00:00
#SBATCH --mem=15GB
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_veryshort
#SBATCH --job-name=swin_test
#SBATCH --account=coms031144
#SBATCH --output=slurm_jobs/swin_opening_test_l1.out

. ~/initConda.sh

conda activate strike

python testing.py --network "swin" --lossFunction "L1" --imageSize 512 --learningParam "opening"
