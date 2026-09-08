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
#SBATCH --job-name=HyperAlign_zeroshot_eval_bs256_lr2e-5_hglr5e-4
#SBATCH -o /PATH/TO/HyperAlign/benchmark_eval/logs/%x_%j.out
#SBATCH -e /PATH/TO/HyperAlign/benchmark_eval/logs/%x_%j.err
#
# Zero-shot retrieval evaluation with the hypergraph model (gate 1.0, signed refinement, graph ON).
# Regenerates the 12 zs_*.json configs (checkpoint baked into workdir_pretrain_bs256_lr2e-5_hglr5e-4), loops run_eval.py,
# then eval_summary.py -> paper-format table.
#
# Usage:  sbatch eval_zeroshot.sh            # all 12 benchmark/mode
#         sbatch eval_zeroshot.sh didemo     # one benchmark first
set -uo pipefail
E2E=/PATH/TO/HyperAlign
EVAL=$E2E/benchmark_eval
CFG=$EVAL/configs
RES=$EVAL/eval_results_bs256_lr2e-5_hglr5e-4
export EVAL_RES_DIR="$RES"                 # eval_summary.py reads results from this bs+lr dir
mkdir -p "$RES" "$EVAL/logs"

# 5-min safety pause after training ends (checkpoint fully flushed) before make_configs picks best ckpt
echo "eval starts, pausing 5 min for checkpoint flush... $(date +%T)"; sleep 300

source /PATH/TO/miniconda3/etc/profile.d/conda.sh
conda activate Multimodal_hypergraph
export WANDB_MODE=offline
export GRAM_MP_CTX=forkserver

# regenerate the 12 zero-shot configs against the best-val checkpoint (workdir_pretrain_bs256_lr2e-5_hglr5e-4)
python3 "$EVAL/make_configs.py"

cd "$E2E"                              # relative paths (datasets/..., ./config/...) resolve from repo root

FILTER="${*:-}"
echo "START $(date +%T)   filter='${FILTER:-ALL}'   [hypergraph]"

for cfg in "$CFG"/zs_*.json; do
  name=$(basename "$cfg" .json | sed 's/^zs_//')
  bench=${name%_*}
  if [ -n "$FILTER" ] && ! { echo "$FILTER" | grep -qw "$bench" || echo "$FILTER" | grep -qw "$name"; }; then continue; fi
  echo "===================== EVAL $name ====================="
  srun python3 -m torch.distributed.launch --nnodes 1 --node_rank 0 --nproc_per_node 4 \
       --master_port 9899 \
       "$EVAL/run_eval.py" --config "$cfg" --output_dir "$RES/out_$name" 2>&1 | tee "$RES/$name.log"
  echo "----- done $name $(date +%T) -----"
done

echo "===================== SUMMARY ====================="
python3 "$EVAL/eval_summary.py" 2>&1 | tee "$RES/RESULTS_zeroshot.txt"
echo "DONE $(date +%T)"
