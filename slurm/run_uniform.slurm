#!/bin/bash

#SBATCH --job-name train-subsample-xgboost
#SBATCH --output logs/train.log
#SBATCH --error logs/train.err
#SBATCH --mail-user giovanni.gaio.1@studenti.unipd.it
##SBATCH --mail-user simone.moretti.1@studenti.unipd.it
#SBATCH --mail-type ALL
#SBATCH
#SBATCH --time 2-00:00:00
# use many CPU if not using CUDA
#SBATCH --cpus-per-task 4
#SBATCH --partition allgroups
#SBATCH --mem 30G
# uncomment to use GPU
# #SBATCH --gres=gpu:rtx

source /nfsd/bcb/bcbg/rossigno/.miniconda3/bin/activate
cd ..

# srun python dna-classifier.py
srun python python/xgbclassifier_random.py --subsample_ratios="[0.1]" --uniform_over_chromosomes=True --dataset "/nfsd/bcb/bcbg/rossigno/PNRR/variant-classifier/datasets/include-chr3/features.csv" --target "/nfsd/bcb/bcbg/rossigno/PNRR/variant-classifier/datasets/include-chr3/mortality.csv" --method="hist"

