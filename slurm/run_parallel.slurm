#!/bin/bash

#SBATCH --job-name subsample-xgboost-parallel
#SBATCH --output logs/train_parallel.log
#SBATCH --error logs/train_parallel.err
#SBATCH --mail-user giovanni.gaio.1@studenti.unipd.it
##SBATCH --mail-user simone.moretti.1@studenti.unipd.it
#SBATCH --mail-type ALL
#SBATCH
#SBATCH --time 0-04:00:00
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

touch $outdir/out_unifsub.txt
srun python python/xgbclassifier_random.py --subsample_ratios="[0.00003162, 0.0001, 0.00031623, 0.001, 0.00316228, 0.01, 0.03162278, 0.1, 0.31622777, 1.]" --uniform_over_chromosomes=True --method="hist" --iterations 64 --stdout-values 1 \
        --dataset "/nfsd/bcb/bcbg/rossigno/PNRR/variant-classifier/datasets/include-chr3/features.csv" \
        --target "/nfsd/bcb/bcbg/rossigno/PNRR/variant-classifier/datasets/include-chr3/mortality.csv" >> $outdir/out_unifsub.txt
touch $outdir/out_subsample.txt
srun python python/xgbclassifier_random.py --subsample_ratios="[0.00003162, 0.0001, 0.00031623, 0.001, 0.00316228, 0.01, 0.03162278, 0.1, 0.31622777, 1.]" --method="hist" --iterations 64 --stdout-values 1 \
        --dataset "/nfsd/bcb/bcbg/rossigno/PNRR/variant-classifier/datasets/include-chr3/features.csv" \
        --target "/nfsd/bcb/bcbg/rossigno/PNRR/variant-classifier/datasets/include-chr3/mortality.csv" >> $outdir/out_subsample.txt
