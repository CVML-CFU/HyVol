#!/bin/bash
# Stage 1: Pretrain Hybrid GRAM (Fixed 0.7:0.3) on VAST150K (0.5 epoch)
set -e

mkdir -p output2/logs

uv run python -m torch.distributed.launch \
  --nproc_per_node 8 \
  --master_port 9852 \
  ./run.py \
  --config ./configs/pretrain/pretrain_hybrid_fixed_0.7_vast150k.json \
  --pretrain_dir ./pretrained_models/pretrain_vast \
  --output_dir ./output2/hybrid_fixed_0.7_vast150k \
  2>&1 | tee output2/logs/stage1_pretrain_fixed_0.7.log
