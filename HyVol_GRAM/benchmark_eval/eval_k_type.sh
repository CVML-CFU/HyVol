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
#SBATCH --job-name=evalKT
#SBATCH -o /PATH/TO/HYVOL/HHR_GRAM/benchmark_eval/logs/evalKT_%j.out
#SBATCH -e /PATH/TO/HYVOL/HHR_GRAM/benchmark_eval/logs/evalKT_%j.out
# ONE checkpoint's 12-mode eval, for a given topk-ablation k (K from --export=K=..).
# run_eval.py applies test-time score re-ranking by default; SIZE_ADAPTIVE
# scales the temp by gallery size so large galleries behave). Set RERANK_OFF=1 for the plain baseline.
# Config-isolated (own copy of configs, checkpoint re-pointed) -> parallel-safe, no make_configs race.
set -uo pipefail
: "${K:?set K (e.g. 2|4|12|16)}"
HA=/PATH/TO/HYVOL/HHR_GRAM
SRC=$HA/benchmark_eval/configs
source /PATH/TO/miniconda3/etc/profile.d/conda.sh
conda activate Multimodal_hypergraph
export WANDB_MODE=offline GRAM_MP_CTX=forkserver PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export TEMP="0.01" SIZE_ADAPTIVE=1

CKPT=$(ls -t "$HA"/workdir_topk_ablation/k${K}/ckpt/best_*.pt 2>/dev/null | head -1)
case "$CKPT" in *"/workdir_topk_ablation/k${K}/"*) : ;; *) echo "ABORT: bad ckpt $CKPT"; exit 1 ;; esac

RUNNER="$HA/benchmark_eval/run_eval.py"
OUT="$HA/benchmark_eval/topk_eval/k${K}"
CFGDIR="$HA/benchmark_eval/configs_k${K}"      # private per-k config dir (no race)
mkdir -p "$OUT" "$CFGDIR" "$HA/benchmark_eval/logs"

# copy the 12 configs and re-point run_cfg.checkpoint -> THIS k's best ckpt
for cfg in "$SRC"/zs_*.json; do
  bn=$(basename "$cfg")
  python3 -c "import json;c=json.load(open('$cfg'));c['run_cfg']['checkpoint']='$CKPT';json.dump(c,open('$CFGDIR/$bn','w'),indent=2)"
done
B=$(python3 -c "import json,glob;print(json.load(open(sorted(glob.glob('$CFGDIR/zs_*.json'))[0]))['run_cfg']['checkpoint'])")
case "$B" in *"/workdir_topk_ablation/k${K}/"*) echo "OK ckpt=$B" ;; *) echo "ABORT: baked $B"; exit 1 ;; esac
echo "$CKPT" > "$OUT/_ckpt.txt"

PORT=$((9800 + K))
echo "START k=$K (port $PORT) $(date +%T)"
for cfg in "$CFGDIR"/zs_*.json; do
  name=$(basename "$cfg" .json | sed 's/^zs_//')
  echo "----- k=$K $name $(date +%T) -----"
  srun python3 -m torch.distributed.launch --nnodes 1 --node_rank 0 --nproc_per_node 4 --master_port $PORT \
       "$RUNNER" --config "$cfg" --output_dir "$OUT/out_$name" 2>&1 | tee "$OUT/$name.log"
done
echo "DONE k=$K $(date +%T) -> $OUT"
