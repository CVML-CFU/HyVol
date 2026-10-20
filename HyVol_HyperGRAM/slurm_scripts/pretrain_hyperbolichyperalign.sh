#!/bin/bash
#SBATCH -A YOUR_SLURM_ACCOUNT
#SBATCH -p boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --job-name=hha_pretrain
#SBATCH -o /PATH/TO/HyperbolicHyperAlign/code/slurm_scripts/logs/%x_%j.out
#SBATCH -e /PATH/TO/HyperbolicHyperAlign/code/slurm_scripts/logs/%x_%j.out
# HyperbolicHyperAlign = HyperGRAM (hybrid geometry) + our hypergraph, on OUR VAST-150k,
# 1 epoch, recipe 2e-5/bs256/frames2, VAST init. save_best keeps the best-val checkpoint.
echo "START=$(date +%T) [HyperbolicHyperAlign pretrain, 1 epoch]"
source /PATH/TO/miniconda3/etc/profile.d/conda.sh
conda activate Multimodal_hypergraph
export WANDB_MODE=offline GRAM_MP_CTX=forkserver PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
cd /PATH/TO/HyperbolicHyperAlign/code
mkdir -p slurm_scripts/logs
RESUME=""; ls ./workdir_pretrain/ckpt/optimizer_step_*.pt >/dev/null 2>&1 && RESUME="--resume true"
srun python3 -m torch.distributed.launch --nnodes 1 --node_rank 0 --nproc_per_node 4 --master_port 9992 \
  ./run.py --config ./config/gram/pretrain_cfg/hyperbolic_hyperalign.json \
  --output_dir ./workdir_pretrain --save_best true --checkpointing true $RESUME 2>&1
echo "DONE $(date +%T)"
