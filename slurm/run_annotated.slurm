#!/bin/bash

#SBATCH --job-name subsample-xgboost-parallel
#SBATCH --output logs/train_annotated.log
#SBATCH --error logs/train_annotated.err
#SBATCH --mail-user giovanni.gaio.1@studenti.unipd.it
##SBATCH --mail-user simone.moretti.1@studenti.unipd.it
#SBATCH --mail-type ALL
#SBATCH
#SBATCH --time 0-04:30:00
# use many CPU if not using CUDA
#SBATCH --cpus-per-task 8
#SBATCH --partition allgroups
#SBATCH --mem 512G
# uncomment to use GPU
# #SBATCH --gres=gpu:rtx

source /nfsd/bcb/bcbg/rossigno/.miniconda3/bin/activate
outdir="$(pwd)/annotated"
mkdir -p $outdir
cd ..

for file_id in "Open_chromatin" "Enhancer" "Promoter"
do
        touch $outdir/out_$file_id.txt
        srun python python/xgbclassifier_random.py --subsample_ratios="[1,0.5,0.1,0.05]"  --method="hist" --iterations 32 --stdout-values 1 \
                --dataset "/nfsd/bcb/bcbg/rossigno/PNRR/variant-classifier/datasets/include-chr3/features.csv" \
                --target "/nfsd/bcb/bcbg/rossigno/PNRR/variant-classifier/datasets/include-chr3/mortality.csv" \
                --annotations "/nfsd/bcb/bcbg/rossigno/PNRR/variant-classifier/datasets/include-chr3/features-sets/data_ensembl.csv" --annotation_type $file_id >> $outdir/out_$file_id.txt
done