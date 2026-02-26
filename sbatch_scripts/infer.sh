#!/bin/bash

exp_folder="abl-output/enclap-audiocaps-CL-baseline/checkpoints"
epoch=(12 12 12)

for e in "${epoch[@]}"; do

    sbatch sbatch_infer.sh $exp_folder $e
    # sh sbatch_infer.sh $exp_folder $e
done
# e=11
# sbatch --job-name=$exp_folder-epoch_$e sbatch_infer.sh $exp_folder $e