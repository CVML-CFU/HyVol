"""Zero-shot eval runner for the trained model.
Reuses the HyperAlign folder's own build_model / dataloaders / test(), but:
  - skips wandb.init (run.py's wandb block reads data_cfg.train[0], which eval configs omit),
  - for VGGSound (val name == 'vgg_ret') swaps in the classification evaluate_mm at RUNTIME
    (mutates the in-memory registry dict only).

Launched exactly like run.py:
  python -m torch.distributed.launch ... run_eval.py --config <zs_cfg.json> --output_dir <dir>
"""
import os, sys
# fully neuter wandb BEFORE it is imported: disabled mode + no stdout console-capture
# (the capture recursed -> RecursionError on configs that print a lot, e.g. audio-only loaders)
os.environ['WANDB_MODE'] = 'disabled'
os.environ['WANDB_CONSOLE'] = 'off'
os.environ['WANDB_SILENT'] = 'true'
E2E = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # the HyperAlign folder (self-locating)
sys.path.insert(0, E2E)
os.chdir(E2E)                                    # relative paths in configs resolve from here

from utils.args import get_args, logging_cfgs
from utils.initialize import initialize
from utils.build_model import build_model
from utils.build_dataloader import create_val_dataloaders
from utils.pipeline import test
import evaluation
import evaluation.evaluation_mm as emm
import wandb
import torch.nn.functional as F

# the eval fn calls wandb.log(val_log); mock it to a plain no-op so we never touch wandb.init /
# its stdout console-capture (which recursed -> RecursionError on configs with heavy log output).
wandb.log = lambda *a, **k: None
wandb.init = lambda *a, **k: None

# --- test-time score re-ranking applied to the similarity matrix before ranking (score re-weighting,
# temperature TEMP). forward (T2V, sort over dim=-1) -> prior softmax over
# dim=0; backward (V2T) -> dim=1. SIZE_ADAPTIVE scales the temp by gallery size N/TEMP_REF_N so it
# behaves on large galleries (e.g. ActivityNet N=4917). Set RERANK_OFF=1 for the plain baseline.
TEMP          = float(os.environ.get('TEMP', '0.01'))
RERANK_OFF    = os.environ.get('RERANK_OFF', '0') == '1'
TEMP_REF_N    = float(os.environ.get('TEMP_REF_N', '1000'))
SIZE_ADAPTIVE = os.environ.get('SIZE_ADAPTIVE', '1') == '1'
_orig_metric  = emm.compute_metric_ret

def _apply(sm0, temp, direction):
    pdim = 0 if direction == 'forward' else 1
    if SIZE_ADAPTIVE:
        temp = temp * (sm0.shape[pdim] / TEMP_REF_N)
    return sm0 * F.softmax(sm0 / temp, dim=pdim)

def rerank_metric(score_matrix, ids, ids_txt, direction='forward'):
    if RERANK_OFF:
        return _orig_metric(score_matrix, ids, ids_txt, direction)
    return _orig_metric(_apply(score_matrix.float(), TEMP, direction), ids, ids_txt, direction)

emm.compute_metric_ret = rerank_metric


def main():
    args = get_args()
    initialize(args)
    logging_cfgs(args)

    # VGGSound = classification (Acc@1/@10), not paired retrieval. The vgg_ret logic already
    # lives in evaluation_classification.evaluate_ret; route to it by name, keeping E2E pristine.
    is_vgg = any(v.get('name', '') == 'vgg_ret' for v in args.data_cfg.val)
    if is_vgg:
        from evaluation.evaluation_classification import evaluate_mm as evaluate_classification
        evaluation.evaluation_registry['evaluation_mm'] = evaluate_classification
        print("VGGSound classification mode: swapped in evaluation_classification.evaluate_mm")

    model, _, _ = build_model(args)
    val_loaders = create_val_dataloaders(args)
    print("EVAL (testing) MODE")
    test(model, val_loaders, args.run_cfg)


if __name__ == '__main__':
    main()
