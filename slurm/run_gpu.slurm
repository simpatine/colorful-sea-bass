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
srun python python/xgbclassifier_random.py --use-gpu=1 --subsample_ratios="[0.1]" --method="hist" \
    --dataset "/nfsd/bcb/bcbg/rossigno/PNRR/variant-classifier/datasets/include-chr3/features.csv" \
    --target "/nfsd/bcb/bcbg/rossigno/PNRR/variant-classifier/datasets/include-chr3/mortality.csv"

