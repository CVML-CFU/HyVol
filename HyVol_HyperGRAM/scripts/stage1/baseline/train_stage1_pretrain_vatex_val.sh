#!/bin/bash
# Stage 1: Pretrain Hybrid GRAM on VAST150K with VATEX Validation (1 epoch)
set -e

mkdir -p output2/logs

uv run python -m torch.distributed.launch \
  --nproc_per_node 8 \
  --master_port 9839 \
  ./run.py \
  --config ./configs/pretrain/pretrain_hybrid_vast150k_vatex_val_paper.json \
  --pretrain_dir ./pretrained_models/pretrain_vast \
  --output_dir ./output2/hybrid_vast150k_vatex_val \
  2>&1 | tee output2/logs/stage1_pretrain_vatex_val.log
