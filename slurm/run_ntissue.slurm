#!/bin/bash

#SBATCH --job-name ntissue-subsample
#SBATCH --output logs/train_ntissue_int.log
#SBATCH --error logs/train_ntissue_int.err
#SBATCH --mail-user giovanni.gaio.1@studenti.unipd.it
##SBATCH --mail-user simone.moretti.1@studenti.unipd.it
#SBATCH --mail-type ALL
#SBATCH
#SBATCH --time 0-05:00:00
# use many CPU if not using CUDA
#SBATCH --cpus-per-task 12
#SBATCH --partition allgroups
#SBATCH --mem 256G
# uncomment to use GPU
# #SBATCH --gres=gpu:rtx

source /nfsd/bcb/bcbg/rossigno/.miniconda3/bin/activate
outdir="$(pwd)/annotated"
mkdir -p $outdir
cd ..

for ntissue in 0 10 20
do
        touch $outdir/out_$ntissue.txt
        srun python python/xgbclassifier_random.py --subsample_ratios="[0.00031623, 0.001, 0.00316228, 0.01, 0.03162278, 0.1, 0.31622777, 1.]"  --method="hist" --iterations 16 --stdout-values 1 \
                --dataset "/nfsd/bcb/bcbg/rossigno/PNRR/variant-classifier/datasets/include-chr3/features.csv" \
                --target "/nfsd/bcb/bcbg/rossigno/PNRR/variant-classifier/datasets/include-chr3/mortality.csv" \
                --annotations "/nfsd/bcb/bcbg/rossigno/PNRR/variant-classifier/datasets/include-chr3/features-sets/data_ensembl.csv" --n_tissue $ntissue >> $outdir/out_$ntissue.txt
done
