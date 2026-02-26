#!/bin/bash
#SBATCH --job-name=enclap-clotho-SW-L1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=5000
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=tien.luong@monash.edu
#SBATCH --output=logs/rebuttal/%x.out
#SBATCH --error=logs/rebuttal/%x.err
#SBATCH --partition=A100
#SBATCH --qos=gpua100


# srun doesnot inherit cpus-per-task from sbatch
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}
# so processes know who to talk to
MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
# Allow communication over InfiniBand cells.
MASTER_ADDR="${MASTER_ADDR}i"
# Get IP for hostname.
export MASTER_ADDR="$(nslookup "$MASTER_ADDR" | grep -oP '(?<=Address: ).*')"
export MASTER_PORT=7010
export GPUS_PER_NODE=2
export NNODES=$SLURM_JOB_NUM_NODES

# hide duplicated errors using this hack - will be properly fixed in pt-1.12
export TORCHELASTIC_ERROR_FILE=/tmp/torch-elastic-error.json

# force crashing on nccl issues like hanging broadcast
export NCCL_ASYNC_ERROR_HANDLING=1

# handle timeouts
export NCCL_IB_TIMEOUT=20

# Make sure we are on the right directory
# cd $MYPROJECT/src

# # This loads modules and python packages
# source sc_venv_template/activate.sh

export LOGLEVEL=INFO
export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=12802
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`

# Run the demo
# time srun bash -c 'accelerate launch \
#     --main_process_ip $MASTER_ADDR \
#     --main_process_port $MASTER_PORT \
#     --multi_gpu \
#     --mixed_precision=no \
#     --dynamo_backend=no \
#     --num_machines=$NNODES  \
#     --machine_rank=$SLURM_PROCID \
#     --rdzv_conf "rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT rdzv_backend=c10d" \
#     train.py cfg/audiocaps/large.yaml'

# time srun bash -c 'accelerate launch \
#     --main_process_ip $MASTER_ADDR \
#     --main_process_port $MASTER_PORT \
#     --multi_gpu \
#     --mixed_precision=bf16 \
#     --dynamo_backend=no \
#     --num_machines=$NNODES  \
#     --machine_rank=$SLURM_PROCID \
#     --rdzv_conf "rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT rdzv_backend=c10d" \
#     train_ot_align.py cfg/clotho/large.yaml'
# time srun bash -c 'accelerate launch \
#     --main_process_ip $MASTER_ADDR \
#     --main_process_port $MASTER_PORT \
#     --multi_gpu \
#     --mixed_precision=bf16 \
#     --dynamo_backend=no \
#     --num_machines=$NNODES  \
#     --machine_rank=$SLURM_PROCID \
#     --rdzv_conf "rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT rdzv_backend=c10d" \
#     train_ot_align.py cfg/clotho/large.yaml'
conda init bash
source ~/.bashrc
conda activate enclap

NGPUS=2
NNODES=1
MASTER_ADDR="127.0.0.1"
MASTER_PORT=29500
NODE_RANK=0

python -m torch.distributed.run --nproc_per_node="${NGPUS}" --nnodes="${NNODES}" --node_rank="${NODE_RANK}" --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" \
        train_ot_align.py cfg/clotho/large.yaml