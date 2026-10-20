
import os, sys
os.environ['WANDB_MODE'] = 'disabled'; os.environ['WANDB_CONSOLE'] = 'off'; os.environ['WANDB_SILENT'] = 'true'


E2E = '/PATH/TO/HyperbolicHyperAlign/code'
sys.path.insert(0, E2E)
os.chdir(E2E)

from utils.args import get_args, logging_cfgs
from utils.initialize import initialize
from utils.build_model import build_model
from utils.build_dataloader import create_val_dataloaders
from utils.pipeline import test
import evaluation
import evaluation.evaluation_mm as emm
import wandb, torch
import torch.nn.functional as F
wandb.log = lambda *a, **k: None
wandb.init = lambda *a, **k: None

TEMP  = float(os.environ.get('TEMP', '0.01'))         
TEMPS = [float(x) for x in os.environ.get('TEMPS', '0.005,0.01,0.02,0.05,0.1').split(',')]  # explored
RERANK_OFF   = os.environ.get('RERANK_OFF', '0') == '1'
_orig_metric = emm.compute_metric_ret

TEMP_REF_N = float(os.environ.get('TEMP_REF_N', '1000'))   # reference gallery size for size-adaptive
SIZE_ADAPTIVE = os.environ.get('SIZE_ADAPTIVE', '1') == '1'

def _apply(sm0, temp, direction, adaptive=False):
    pdim = 0 if direction == 'forward' else 1     # forward T2V->prior over texts(0); backward V2T->videos(1)
    if adaptive:
        N = sm0.shape[pdim]                       # size of the softmax'd (prior) dimension
        temp = temp * (N / TEMP_REF_N)            # big gallery -> larger temp -> weaker correction
    return sm0 * F.softmax(sm0 / temp, dim=pdim)

def rerank_metric(score_matrix, ids, ids_txt, direction='forward'):
    base = _orig_metric(score_matrix, ids, ids_txt, direction)
    if RERANK_OFF:
        return base
    sm0 = score_matrix.float()
    rk = 'forward_r1' if direction == 'forward' else 'backward_r1'   # fix: backward returns backward_r1
    pdim = 0 if direction == 'forward' else 1
    N = sm0.shape[pdim]
    # fixed-temp landscape
    land = [(t, _orig_metric(_apply(sm0, t, direction), ids, ids_txt, direction).get(rk)) for t in TEMPS]
    print(f"[rerank  {direction} N={N}] base={base.get(rk)} | " +
          "  ".join(f"t{t}={r1}" for t, r1 in land), flush=True)
    # size-adaptive landscape (temp scaled by N/TEMP_REF_N)
    landa = [(t, _orig_metric(_apply(sm0, t, direction, adaptive=True), ids, ids_txt, direction).get(rk)) for t in TEMPS]
    print(f"[rerank-adpt {direction} N={N}] base={base.get(rk)} | " +
          "  ".join(f"r{t}(t{round(t*N/TEMP_REF_N,4)})={r1}" for t, r1 in landa), flush=True)
    return _orig_metric(_apply(sm0, TEMP, direction, adaptive=SIZE_ADAPTIVE), ids, ids_txt, direction)

emm.compute_metric_ret = rerank_metric   


def main():
    args = get_args()
    initialize(args); logging_cfgs(args)
    is_vgg = any(v.get('name', '') == 'vgg_ret' for v in args.data_cfg.val)
    if is_vgg:                                    
        from evaluation.evaluation_classification import evaluate_mm as evc
        evaluation.evaluation_registry['evaluation_mm'] = evc
    model, _, _ = build_model(args)
    val_loaders = create_val_dataloaders(args)
    print(f"EVAL (temp={TEMP}, off={RERANK_OFF}) MODE", flush=True)
    test(model, val_loaders, args.run_cfg)


if __name__ == '__main__':
    main()
