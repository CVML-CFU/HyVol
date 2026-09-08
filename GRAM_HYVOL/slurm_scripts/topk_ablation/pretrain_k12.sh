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
#SBATCH --job-name=ha_topk_k12
#SBATCH -o /PATH/TO/HyperAlign/slurm_scripts/logs/%x_%j.out
#SBATCH -e /PATH/TO/HyperAlign/slurm_scripts/logs/%x_%j.out
# TOP-K ABLATION: HyperAlign (GRAM+hypergraph), knn_k=12, epoch 1, recipe 2e-5/bs256/frames2.
# Only knn_k differs across the sweep; everything else identical -> isolates the effect of k.
echo "START=$(date +%T) [HyperAlign top-k ablation knn_k=12 -> ./workdir_topk_ablation/k12]"
source /PATH/TO/miniconda3/etc/profile.d/conda.sh
conda activate Multimodal_hypergraph
export WANDB_MODE=offline
export GRAM_MP_CTX=forkserver
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
cd /PATH/TO/HyperAlign
RESUME=""
ls ./workdir_topk_ablation/k12/ckpt/optimizer_step_*.pt >/dev/null 2>&1 && RESUME="--resume true" && echo "RESUME found"
srun python3 -m torch.distributed.launch --nnodes 1 --node_rank 0 --nproc_per_node 4 --master_port 9905 \
  ./run.py --config ./config/gram/pretrain_cfg/topk_ablation/hyperalign_k12.json \
  --output_dir ./workdir_topk_ablation/k12 --save_best true --checkpointing true $RESUME 2>&1
echo "DONE $(date +%T)"
