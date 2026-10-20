#!/bin/bash
#SBATCH -A YOUR_SLURM_ACCOUNT
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --job-name=hh_smoke
#SBATCH -o /PATH/TO/HyperbolicHyperAlign/code/slurm_scripts/logs/%x_%j.out
#SBATCH -e /PATH/TO/HyperbolicHyperAlign/code/slurm_scripts/logs/%x_%j.out
# SMOKE test: HyperbolicHyperAlign = HyperGRAM (hybrid geometry) + our hypergraph.
# 20 train steps only, eval skipped (valid_freq high). Verifies the model+losses run
# end-to-end on our VAST-150k data. Recipe: lr2e-5 / bs256 / frames2 / epoch1 / new_lr5e-4.
echo "START=$(date +%T) [HyperbolicHyperAlign SMOKE -> ./workdir_smoke]"
source /PATH/TO/miniconda3/etc/profile.d/conda.sh
conda activate Multimodal_hypergraph
export WANDB_MODE=offline
export GRAM_MP_CTX=forkserver
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
cd /PATH/TO/HyperbolicHyperAlign/code
srun python3 -m torch.distributed.launch --nnodes 1 --node_rank 0 --nproc_per_node 4 --master_port 9977 \
  ./run.py --config ./config/gram/pretrain_cfg/hyperbolic_hyperalign_smoke.json \
  --output_dir ./workdir_smoke --num_train_steps 12 --valid_freq 1 --checkpointing true 2>&1
echo "DONE $(date +%T)"
