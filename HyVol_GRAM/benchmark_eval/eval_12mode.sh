#!/bin/bash
#SBATCH -A YOUR_SLURM_ACCOUNT
#SBATCH -p boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=02:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --job-name=HHR_GRAM_eval
#SBATCH -o /PATH/TO/HYVOL/HHR_GRAM/benchmark_eval/logs/%x_%j.out
#SBATCH -e /PATH/TO/HYVOL/HHR_GRAM/benchmark_eval/logs/%x_%j.out
# HHR-GRAM zero-shot eval, all 12 benchmark-modes. run_eval.py applies test-time score re-ranking by
# default; SIZE_ADAPTIVE scales the temp by gallery size so large galleries
# (e.g. ActivityNet) behave. Set RERANK_OFF=1 for the plain baseline. configs point to the k12 model.
set -uo pipefail
E2E=/PATH/TO/HYVOL/HHR_GRAM
EVAL=$E2E/benchmark_eval; CFG=$EVAL/configs; RES=$EVAL/eval_results
source /PATH/TO/miniconda3/etc/profile.d/conda.sh
conda activate Multimodal_hypergraph
export WANDB_MODE=offline GRAM_MP_CTX=forkserver PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export TEMP="0.01" SIZE_ADAPTIVE=1
cd $E2E; mkdir -p $RES $EVAL/logs
for cfg in $CFG/zs_*.json; do
  name=$(basename $cfg .json | sed 's/^zs_//')
  echo "===== EVAL $name $(date +%T) ====="
  srun python3 -m torch.distributed.launch --nnodes 1 --node_rank 0 --nproc_per_node 4 --master_port 9996 \
       $EVAL/run_eval.py --config $cfg --output_dir $RES/out_$name 2>&1 | tee $RES/$name.log
done
echo "ALL DONE $(date +%T)"
