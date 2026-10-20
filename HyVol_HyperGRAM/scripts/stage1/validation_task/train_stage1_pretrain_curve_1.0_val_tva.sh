#!/bin/bash
# Stage 1: Pretrain Hybrid GRAM (Curvature=1.0, Learnable Weights, Val=TVA) on VAST150K (0.5 epoch)
set -e

mkdir -p output2/logs

uv run python -m torch.distributed.launch \
  --nproc_per_node 8 \
  --master_port 9862 \
  ./run.py \
  --config ./configs/pretrain/pretrain_hybrid_curve_1.0_val_tva_vast150k.json \
  --pretrain_dir ./pretrained_models/pretrain_vast \
  --output_dir ./output2/hybrid_curve_1.0_val_tva_vast150k \
  2>&1 | tee output2/logs/stage1_pretrain_curve_1.0_val_tva.log
