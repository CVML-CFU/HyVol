#!/bin/bash
#SBATCH -A YOUR_SLURM_ACCOUNT
#SBATCH -p boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --job-name=Hyper_eval
#SBATCH -o /PATH/TO/HyperbolicHyperAlign/code/benchmark_eval/logs/%x_%j.out
#SBATCH -e /PATH/TO/HyperbolicHyperAlign/code/benchmark_eval/logs/%x_%j.out
set -uo pipefail
E2E=/PATH/TO/HyperbolicHyperAlign/code
EVAL=$E2E/benchmark_eval; CFG=$EVAL/configs; RES=$EVAL/eval_results
source /PATH/TO/miniconda3/etc/profile.d/conda.sh
conda activate Multimodal_hypergraph
export WANDB_MODE=offline GRAM_MP_CTX=forkserver PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export TEMPS="0.005,0.01,0.02,0.05,0.1" TEMP="0.01"
cd $E2E; mkdir -p $RES $EVAL/logs
for cfg in $CFG/zs_*.json; do
  name=$(basename $cfg .json|sed 's/^zs_//')
  echo "===== EVAL $name $(date +%T) ====="
  srun python3 -m torch.distributed.launch --nnodes 1 --node_rank 0 --nproc_per_node 4 --master_port 9995 \
       $EVAL/run_eval.py --config $cfg --output_dir $RES/out_$name 2>&1 | tee $RES/$name.log
done
echo "ALL DONE $(date +%T)"
