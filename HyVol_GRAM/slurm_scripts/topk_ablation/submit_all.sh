#!/bin/bash
# top-k ablation sweep {4,12,16}. k=8 = main HyperAlign run (not duplicated here);
# k=2/3 dropped (isolate ~17%/6% of nodes); k>16 clamps to 16 at per-branch B=64.
cd /PATH/TO/HyperAlign
sbatch slurm_scripts/topk_ablation/pretrain_k4.sh
sbatch slurm_scripts/topk_ablation/pretrain_k12.sh
sbatch slurm_scripts/topk_ablation/pretrain_k16.sh
