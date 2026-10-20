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
#SBATCH --job-name=vggHHA
#SBATCH -o /PATH/TO/HyperbolicHyperAlign/code/benchmark_eval/logs/vggfix_%j.out
#SBATCH -e /PATH/TO/HyperbolicHyperAlign/code/benchmark_eval/logs/vggfix_%j.out
set -uo pipefail
E2E=/PATH/TO/HyperbolicHyperAlign/code
source /PATH/TO/miniconda3/etc/profile.d/conda.sh
conda activate Multimodal_hypergraph
export WANDB_MODE=offline GRAM_MP_CTX=forkserver PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
cd "$E2E"; mkdir -p benchmark_eval/logs benchmark_eval/vggfix_out
echo "START vgg-cls fix test $(date +%T)  [HyperbolicHyperAlign]"
srun python3 -m torch.distributed.launch --nnodes 1 --node_rank 0 --nproc_per_node 4 --master_port 9944 \
   ./run.py --config $E2E/benchmark_eval/configs/zs_vggsound_tav.json --output_dir $E2E/benchmark_eval/vggfix_out 2>&1 | tee benchmark_eval/vggfix.log
echo "DONE $(date +%T)"; echo "=== VGG-CLS (expect ~40) ==="
grep -aE 'VGG-CLS|volume_ITM_T2D_r1|AssertionError' benchmark_eval/vggfix.log | tail -5
