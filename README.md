# HyperAlign
Hypergraph-Regularized Gramian Volumes for Multimodal Retrieval

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

HyVoL keeps the backbone's frozen encoders and continues pretraining from the
public VAST foundation checkpoint. Download each into `${WORK_ROOT}/pretrained_weights/`:

- **EVA-CLIP** vision encoder (`EVA01_CLIP_g_14`)
- **BEATs** audio encoder (`BEATs_iter3_plus_AS2M`)
- **BERT** text encoder (`bert-base-uncased`)
- **VAST** foundation checkpoint (`pretrain_vast/ckpt/model_step_*.pt`) as initialization

None of these weights are redistributed here.

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
