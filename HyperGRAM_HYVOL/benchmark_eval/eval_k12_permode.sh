#!/bin/bash
#SBATCH -A YOUR_SLURM_ACCOUNT
#SBATCH -p boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --array=0-11
#SBATCH --job-name=k12eval
#SBATCH -o /PATH/TO/HYVOL/HHR_HyperGRAM/code/benchmark_eval/logs/%x_%A_%a.out
#SBATCH -e /PATH/TO/HYVOL/HHR_HyperGRAM/code/benchmark_eval/logs/%x_%A_%a.out
# k12 HyperGRAM eval — ONE mode per array task (small jobs, not one 12-mode run).
# TAG + RERANK_OFF passed at submit:  DSL -> TAG=dsl RERANK_OFF=0 ; non-DSL -> TAG=nodsl RERANK_OFF=1
set -uo pipefail
E2E=/PATH/TO/HyperbolicHyperAlign/code
EVAL=/PATH/TO/HYVOL/HHR_HyperGRAM/code/benchmark_eval
CFG=$EVAL/configs_k12
source /PATH/TO/miniconda3/etc/profile.d/conda.sh
conda activate Multimodal_hypergraph
export WANDB_MODE=offline GRAM_MP_CTX=forkserver PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export TEMP="0.01" SIZE_ADAPTIVE=1
export TEMPS="0.005,0.01,0.02,0.05,0.1"
export RERANK_OFF="${RERANK_OFF:-0}"
TAG="${TAG:-dsl}"

MODES=(msrvtt_tv msrvtt_tva msrvtt_tvas didemo_tv didemo_tva \
       activitynet_tv activitynet_tva vatex_tv vatex_tva vatex_tvas \
       audiocaps_tva vggsound_tav)
name=${MODES[$SLURM_ARRAY_TASK_ID]}
RES=$EVAL/eval_results_k12_${TAG}
mkdir -p "$RES" "$EVAL/logs"
# per-(tag,mode) unique port so nothing clashes
off=0; [ "$TAG" = "nodsl" ] && off=50
PORT=$((9700 + off + SLURM_ARRAY_TASK_ID))

echo "===== k12 EVAL  mode=$name  TAG=$TAG  RERANK_OFF=$RERANK_OFF  port=$PORT  $(date +%T) ====="
cd "$E2E"
srun python3 -m torch.distributed.launch --nnodes 1 --node_rank 0 --nproc_per_node 4 --master_port $PORT \
     "$EVAL/run_eval.py" --config "$CFG/zs_${name}.json" --output_dir "$RES/out_${name}" 2>&1 | tee "$RES/${name}.log"
echo "DONE $name $(date +%T)"
