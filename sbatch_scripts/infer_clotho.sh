#!/bin/bash

exp_folder="abl-output/enclap-Clotho-SW-kernel-L10/checkpoints"
epoch=(13 13 13 13 13)

for e in "${epoch[@]}"; do

    sbatch sbatch_infer_clotho.sh $exp_folder $e
    # sh sbatch_infer.sh $exp_folder $e
done
# e=11
# sbatch --job-name=$exp_folder-epoch_$e sbatch_infer.sh $exp_folder $e