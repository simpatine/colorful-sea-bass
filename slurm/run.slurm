#!/bin/bash

#SBATCH --job-name train-subsample-xgboost
#SBATCH --output logs/train.log
#SBATCH --error logs/train.err
#SBATCH --mail-user giovanni.gaio.1@studenti.unipd.it
#SBATCH --mail-type ALL
#SBATCH
#SBATCH --time 3-00:00:00
# use many CPU if not using CUDA
#SBATCH --cpus-per-task 32
#SBATCH --partition allgroups
#SBATCH --mem 64G
# uncomment to use GPU
##SBATCH --gres=gpu:rtx

source /nfsd/bcb/bcbg/rossigno/.miniconda3/bin/activate
cd ..

# srun python dna-classifier.py
srun python python/xgbclassifier_random.py --subsample_ratio=0.5 --dataset "/nfsd/bcb/bcbg/rossigno/PNRR/variant-classifier/datasets/include-chr3/features.csv" --target "/nfsd/bcb/bcbg/rossigno/PNRR/variant-classifier/datasets/include-chr3/mortality.csv"

