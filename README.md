# HyVoL: Hypergraph-Regularized Gramian Volumes for Multimodal Retrieval

Code release for **HyVoL**, a **training-time** hypergraph regularizer for
volume-based multimodal alignment.
During training it couples each clip with a *document* hyperedge (over its own
modalities) and *semantic* mutual-top-$k$ hyperedges (over related clips), then
gates a refinement back onto the modality embeddings. **At inference the
hypergraph is removed** and the plain volume backbone is scored, so HyVoL adds no
test-time cost.

We release HyVoL on two volume backbones:

| Folder | Backbone | Volume |
| --- | --- | --- |
| `GRAM_HYVOL/` | GRAM | Euclidean Gramian |
| `HyperGRAM_HYVOL/` | HyperGRAM | hyperbolic / hybrid Gramian |

The two folders share the same layout and interface; pick the backbone you want.

## 1. Environment

```bash
conda create -n hyvol python=3.9 -y
conda activate hyvol
pip install torch==2.0.1 torchvision --index-url https://download.pytorch.org/whl/cu117
pip install torch_geometric
pip install dhg --no-deps
pip install -r requirements.txt
```
Multi-GPU launch uses `torch.distributed`; commands below assume 4 GPUs.

## 2. Model weights (frozen encoders + init)

HyVoL builds on the **GRAM** backbone (`github.com/ispamm/GRAM`) and follows its
frozen-encoder and initialisation setup: three frozen encoders and one
initialisation checkpoint, with continued pretraining starting from the public
VAST foundation model. Download each into `${WORK_ROOT}/pretrained_weights/` —
none is redistributed here.

| Component | File | Source |
| --- | --- | --- |
| Vision — EVA-CLIP ViT-g/14 | `EVA01_CLIP_g_14_psz14_s11B.pt` | `huggingface.co/QuanSun/EVA-CLIP` |
| Audio — BEATs | `BEATs_iter3_plus_AS2M.pt` | `github.com/microsoft/unilm` (`beats/`) |
| Text — BERT-base | `bert-base-uncased` | `huggingface.co/google-bert/bert-base-uncased` |
| Initialisation — VAST foundation | `pretrain_vast/ckpt/model_step_*.pt` | `github.com/ispamm/GRAM` (GRAM / VAST setup) |

## 3. Data

Six benchmarks — **five retrieval and one audio–visual classification** — plus the
continued-pretraining corpus. Obtain each from its original source; none is
redistributed here. *Test* is the retrieval gallery size used in evaluation.

| Dataset | Modalities | Train | Test | Task |
| --- | --- | --- | --- | --- |
| MSR-VTT | video, audio, subtitles | 9,000 | 1,000 | Retrieval |
| DiDeMo | video, audio | 8,394 | 1,003 | Retrieval |
| ActivityNet Captions | video, audio | 10,009 | 4,917 | Retrieval |
| VATEX | video, audio, subtitles | 14,060 | 431 | Retrieval |
| AudioCaps | video, audio | — | 700 | Retrieval |
| VGGSound-5K | video, audio | — | 5,000 | Classification |
| VAST-27M subset | video, audio, subtitles | 150k (nominal) | — | Continued pretraining |

MSR-VTT and VATEX carry a subtitle track; DiDeMo, ActivityNet, and AudioCaps use
the {video, audio} set. VGGSound-5K is scored as classification-as-retrieval over
its class labels and is not directly comparable to the retrieval tables.

## 4. Environment variables

Set these (or edit the config JSONs, which currently carry a `/PATH/TO`
placeholder) so the code finds your local data and weights:

```bash
export DATA_ROOT=/your/datasets       # benchmark clips + annotations
export WORK_ROOT=/your/workdir        # pretrained_weights/, output checkpoints
```

## 5. Training (continued pretraining with the hypergraph regularizer)

```bash
cd GRAM_HYVOL        # or HyperGRAM_HYVOL
python -m torch.distributed.launch --nproc_per_node 4 run.py \
    --config config/<train_config>.json
```
Key settings (defaults in the configs):

| Setting | Value |
| --- | --- |
| neighbours per clip, $k$ | 12 (adaptive cap $\min(k,\lfloor B_{\text{shard}}/4\rfloor)$) |
| document-hyperedge weight $w_{\text{doc}}$ | 1.0 |
| regularizer weight $w_{\text{reg}}$ | 0.1 |
| gate initialization | 1.0 |
| temperature $\tau$ (learned) | init 0.07 |
| label smoothing | 0.1 |
| learning rate / batch size / frames | 2e-5 / 256 / 2 |

The hypergraph is built **only** during training.

## 6. Evaluation

Pretrained HyVoL checkpoints are available on Hugging Face:
[huggingface.co/anindyanag58/HyVol](https://huggingface.co/anindyanag58/HyVol)

| Folder | Checkpoint |
| --- | --- |
| `GRAM_HYVOL/` | `GRAM_HYVOL/HyVol_GRAM.pt` |
| `HyperGRAM_HYVOL/` | `HyperGRAM_HYVOL/HyVol_HyperGRAM.pt` |

Set `GRAM_CKPT` below to the downloaded file to evaluate without training.

```bash
cd GRAM_HYVOL        # or HyperGRAM_HYVOL
export GRAM_CKPT=${WORK_ROOT}/ckpt/<checkpoint>.pt
python benchmark_eval/make_configs.py          # bakes the checkpoint into configs/zs_*.json
for cfg in benchmark_eval/configs/zs_*.json; do
  python -m torch.distributed.launch --nproc_per_node 4 \
      benchmark_eval/run_eval.py --config "$cfg" --output_dir out/$(basename "$cfg" .json)
done
```
Reported metric: two-stage `volume_ITM_T2D` / `volume_ITM_D2T` (R@1/R@5/R@10).

**Missing-modality robustness** is controlled by environment variables (the
observed-modality volume is recovered with the identity-padded Gramian in
`utils/volume.py`):

```bash
export MISS_RATE=25 MISS_MODS=vas MISS_SEED=1234   # rate in {0,25,50,75,90}
python -m torch.distributed.launch --nproc_per_node 4 \
    benchmark_eval/run_eval.py --config benchmark_eval/configs/zs_msrvtt_tvas.json \
    --output_dir out/msrvtt_r25
```
`MISS_RATE=0` reproduces the no-mask numbers exactly.

## 7. Config → paper mapping

| Config / setting | Paper |
| --- | --- |
| `zs_msrvtt_{tv,tva,tvas}` | Table 1 (MSR-VTT retrieval) |
| `zs_didemo_{tv,tva}`, `zs_activitynet_{tv,tva}`, `zs_vatex_{tv,tva,tvas}` | Table 2 (DiDeMo / ActivityNet / VATEX) |
| `zs_audiocaps_tva`, `zs_vggsound_tav` | Table 3 (AudioCaps retrieval; VGGSound-5K classification) |
| `MISS_RATE`/`MISS_MODS` sweep (on GRAM) | Table 4 (missing modality; MSR-VTT/VATEX T–VAS, DiDeMo/ActivityNet T–VA) |
| `workdir_topk_ablation/k{4,8,12,16}` (GRAM) | Figure 2 (semantic-neighbourhood size $k$) |

## 8. Layout

```
GRAM_HYVOL/ , HyperGRAM_HYVOL/
  model/          backbone forward + hypergraph refinement (stage B)
  utils/          Gramian volume (+ masked, missing-modality) and helpers
  evaluation/     retrieval (evaluation_mm.py) and classification eval
  benchmark_eval/ make_configs.py, run_eval.py, configs/
  config/         default model/run configs
  data/           dataset / dataloader code
  run.py          training / eval entry point
```

