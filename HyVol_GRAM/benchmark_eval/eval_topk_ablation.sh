#!/bin/bash
#SBATCH -A YOUR_SLURM_ACCOUNT
#SBATCH -p boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --job-name=topk_eval12
#SBATCH -o /PATH/TO/HyperAlign/benchmark_eval/logs/%x_%j.out
#SBATCH -e /PATH/TO/HyperAlign/benchmark_eval/logs/%x_%j.out
# 12-mode benchmark eval for the top-k ablation checkpoints (k=4,12,16).
# SEQUENTIAL over k (make_configs.py writes shared configs, so no parallel race).
# Each k's results go to a SEPARATE folder: benchmark_eval/topk_eval/k<k>/
set -uo pipefail
E2E=/PATH/TO/HyperAlign
EVAL=$E2E/benchmark_eval
CFG=$EVAL/configs
source /PATH/TO/miniconda3/etc/profile.d/conda.sh
conda activate Multimodal_hypergraph
export WANDB_MODE=offline GRAM_MP_CTX=forkserver PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
cd "$E2E"
mkdir -p "$EVAL/logs"

for k in 4 12 16; do
  CKPT=$(ls -t "$E2E"/workdir_topk_ablation/k${k}/ckpt/best_*.pt 2>/dev/null | head -1)
  if [ -z "$CKPT" ]; then echo "SKIP k=$k: no best checkpoint"; continue; fi
  # GUARD 1: the checkpoint path MUST live in this k's own folder (no cross-k / no main k=8 mix-up)
  case "$CKPT" in
    *"/workdir_topk_ablation/k${k}/"*) : ;;
    *) echo "ABORT k=$k: checkpoint path not in k${k} folder -> $CKPT"; continue ;;
  esac
  RES="$EVAL/topk_eval/k${k}"        # <-- separate folder per k (no overwrite)
  mkdir -p "$RES"
  echo "================= EVAL k=$k  ckpt=$CKPT =================" | tee "$RES/_ckpt.txt"
  export GRAM_CKPT="$CKPT"            # make_configs bakes THIS checkpoint into the 12 configs
  python3 "$EVAL/make_configs.py"
  # GUARD 2: verify make_configs actually baked THIS k's checkpoint (not the fallback main k=8 ckpt)
  BAKED=$(python3 -c "import json,glob;print(json.load(open(sorted(glob.glob('$CFG/zs_*.json'))[0]))['run_cfg'].get('checkpoint',''))" 2>/dev/null)
  echo "  baked checkpoint = $BAKED" | tee -a "$RES/_ckpt.txt"
  case "$BAKED" in
    *"/workdir_topk_ablation/k${k}/"*) echo "  ✅ GUARD OK: k${k} checkpoint baked" ;;
    *) echo "  ❌ ABORT k=$k: baked checkpoint is NOT k${k} ($BAKED) — skipping to avoid wrong eval"; continue ;;
  esac
  for cfg in "$CFG"/zs_*.json; do
    name=$(basename "$cfg" .json | sed 's/^zs_//')
    echo "----- k=$k mode $name $(date +%T) -----"
    srun python3 -m torch.distributed.launch --nnodes 1 --node_rank 0 --nproc_per_node 4 --master_port $((9950+k)) \
         "$EVAL/run_eval.py" --config "$cfg" --output_dir "$RES/out_$name" 2>&1 | tee "$RES/$name.log"
  done
  python3 "$EVAL/eval_summary.py" 2>&1 | tee "$RES/RESULTS_k${k}.txt" || true
  echo "===== k=$k DONE $(date +%T) -> $RES ====="
done
echo "ALL TOP-K EVAL DONE $(date +%T)"
