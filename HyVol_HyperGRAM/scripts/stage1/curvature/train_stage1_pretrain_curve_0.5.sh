#!/bin/bash
# Stage 1: Pretrain Hybrid GRAM (Curvature=0.5, Learnable Weights) on VAST150K (0.5 epoch)
set -e

mkdir -p output2/logs

uv run python -m torch.distributed.launch \
  --nproc_per_node 8 \
  --master_port 9855 \
  ./run.py \
  --config ./configs/pretrain/pretrain_hybrid_curve_0.5_vast150k.json \
  --pretrain_dir ./pretrained_models/pretrain_vast \
  --output_dir ./output2/hybrid_curve_0.5_vast150k \
  2>&1 | tee output2/logs/stage1_pretrain_curve_0.5.log
