#!/bin/bash
# Stage 2: Finetune Hybrid GRAM on MSR-VTT (PRE version - official hyperparameters)
set -e

mkdir -p output2/logs

uv run python -m torch.distributed.launch \
  --nproc_per_node 8 \
  --master_port 9846 \
  ./run.py \
  --config ./configs/finetune/finetune_hybrid_msrvtt_pre_paper.json \
  --pretrain_dir ./pretrained_models/GRAM_finetuned_MSRVTT \
  --output_dir ./output2/hybrid_msrvtt_pre \
  2>&1 | tee output2/logs/stage2_finetune_msrvtt_pre.log
