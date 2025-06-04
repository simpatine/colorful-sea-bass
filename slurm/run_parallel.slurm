#!/bin/bash

#SBATCH --job-name subsample-xgboost-parallel
#SBATCH --output logs/train_parallel.log
#SBATCH --error logs/train_parallel.err
#SBATCH --mail-user giovanni.gaio.1@studenti.unipd.it
##SBATCH --mail-user simone.moretti.1@studenti.unipd.it
#SBATCH --mail-type ALL
#SBATCH
#SBATCH --time 0-05:00:00
# use many CPU if not using CUDA
#SBATCH --cpus-per-task 32
#SBATCH --partition allgroups
#SBATCH --mem 512G
# uncomment to use GPU
# #SBATCH --gres=gpu:rtx

source /nfsd/bcb/bcbg/rossigno/.miniconda3/bin/activate
outdir="$(pwd)/parallel"
mkdir -p $outdir
cd ..

file_id=0

touch $outdir/out_$file_id.txt
srun python python/xgbclassifier_random.py --subsample_ratios="[0.00001,0.00005,0.0001,0.0005,0.001,0.005,0.01,0.05]" --uniform_over_chromosomes=True --method="hist" --iterations 32 --stdout-values 1 \
        --dataset "/nfsd/bcb/bcbg/rossigno/PNRR/variant-classifier/datasets/include-chr3/features.csv" \
        --target "/nfsd/bcb/bcbg/rossigno/PNRR/variant-classifier/datasets/include-chr3/mortality.csv" >> $outdir/out_$file_id.txt
