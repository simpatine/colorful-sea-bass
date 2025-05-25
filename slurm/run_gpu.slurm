#!/bin/bash

#SBATCH --job-name train-subsample-xgboost-gpu
#SBATCH --output logs/train_gpu.log
#SBATCH --error logs/train_gpu.err
#SBATCH --mail-user giovanni.gaio.1@studenti.unipd.it
##SBATCH --mail-user simone.moretti.1@studenti.unipd.it
#SBATCH --mail-type ALL
#SBATCH
#SBATCH --time 0-2:00:00
# use many CPU if not using CUDA
#SBATCH --cpus-per-task 4
#SBATCH --partition allgroups
#SBATCH --mem 64G
# uncomment to use GPU
#SBATCH --gres=gpu:rtx

source /nfsd/bcb/bcbg/rossigno/.miniconda3/bin/activate
cd ..

# srun python dna-classifier.py
srun python python/xgbclassifier_random.py --use-gpu=1 --subsample_ratio=0.1 --dataset "/nfsd/bcb/bcbg/rossigno/PNRR/variant-classifier/datasets/exclude-chr3/features-sample10.csv" --target "/nfsd/bcb/bcbg/rossigno/PNRR/variant-classifier/datasets/exclude-chr3/mortality-sample10.csv"

