#!/bin/bash

#SBATCH --job-name subsample-xgboost-sweep
#SBATCH --output logs/train_sweep.log
#SBATCH --error logs/train_sweep.err
#SBATCH --mail-user giovanni.gaio.1@studenti.unipd.it
##SBATCH --mail-user simone.moretti.1@studenti.unipd.it
#SBATCH --mail-type ALL
#SBATCH
#SBATCH --time 1-00:00:00
# use many CPU if not using CUDA
#SBATCH --cpus-per-task 8
#SBATCH --partition allgroups
#SBATCH --mem 256G
# uncomment to use GPU
# #SBATCH --gres=gpu:rtx

source /nfsd/bcb/bcbg/rossigno/.miniconda3/bin/activate
outdir="$(pwd)/sweep"
mkdir -p $outdir
cd ..

    # echo "--------------- Running subsample_ratio=$i ---------------"
for s in $(seq 0 4)
do
    for i in 1 0.5 0.1 0.05 0.01 0.005 0.001 0.0005 0.0001 0.00005
    do
    touch $outdir/out_subsample_$i.txt
        srun python python/xgbclassifier_random.py --subsample_ratio $i --method="hist" --stdout-values 1 \
            --random-state $s \
            --dataset "/nfsd/bcb/bcbg/rossigno/PNRR/variant-classifier/datasets/exclude-chr3/features.csv" \
            --target "/nfsd/bcb/bcbg/rossigno/PNRR/variant-classifier/datasets/exclude-chr3/mortality.csv" >> $outdir/out_subsample_$i.txt &
        # sleep 10
    done
    wait
done
