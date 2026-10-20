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
#SBATCH --job-name=HyperAlign_pretrain_bs256_lr2e-5_hglr5e-4
#SBATCH -o /PATH/TO/HyperAlign/slurm_scripts/logs/%x_%j.out
#SBATCH -e /PATH/TO/HyperAlign/slurm_scripts/logs/%x_%j.err
# VAST-150k pretraining launcher — GRAM recipe (epoch5/bs256/lr2e-5/frames2) with hypergraph
# alignment (gate initialized to 1.0, signed refinement, w_doc 1.0).
# Writes to workdir_pretrain_bs256_lr2e-5_hglr5e-4/4model. Resume only triggers off that directory, so a first run
# always starts from the VAST pretrained weights.
echo "START=$(date +%T) [HyperAlign pretrain, GRAM recipe + hypergraph -> ./workdir_pretrain_bs256_lr2e-5_hglr5e-4/4model]"
source /PATH/TO/miniconda3/etc/profile.d/conda.sh
conda activate Multimodal_hypergraph
export WANDB_MODE=offline
export GRAM_MP_CTX=forkserver
cd /PATH/TO/HyperAlign
# auto-resume: continue from the latest optimizer checkpoint after a crash
RESUME=""
if ls ./workdir_pretrain_bs256_lr2e-5_hglr5e-4/4model/ckpt/optimizer_step_*.pt >/dev/null 2>&1; then
  RESUME="--resume true"; echo "RESUME: checkpoint found -> continue"
else
  echo "FRESH"
fi
srun python3 -m torch.distributed.launch --nnodes 1 --node_rank 0 --nproc_per_node 4 --master_port 9893 \
  ./run.py --config ./config/gram/pretrain_cfg/hyperalign.json \
  --output_dir ./workdir_pretrain_bs256_lr2e-5_hglr5e-4/4model --checkpointing true $RESUME 2>&1
echo "DONE $(date +%T)"
