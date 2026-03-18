import sys
from pathlib import Path as _Path
sys.path.append(str(_Path(__file__).resolve().parent / 'src'))
sys.path.append(str(_Path(__file__).resolve().parent))

import torch

from src.main import build_parser
from src.algorithms.FedDF import MMFL as FedDF
from src.algorithms.FedMD import MMFL as FedMD


class DummyWandb:
    def log(self, *args, **kwargs):
        return None


def nan_params(model, label):
    bad = []
    for name, param in model.named_parameters():
        if torch.isnan(param.detach()).any() or torch.isinf(param.detach()).any():
            bad.append(name)
    print(label, 'nan_param_count', len(bad))
    for name in bad[:30]:
        print(label, 'bad_param', name)


def check_eval_features(evaluator, loader, label):
    feats = evaluator.extract_features(loader)
    for key in ['image_features', 'caption_features']:
        tensor = feats[key]
        print(label, key, 'has_nan', torch.isnan(tensor).any().item(), 'has_inf', torch.isinf(tensor).any().item(), 'mean_abs', tensor.float().abs().mean().item())

args = build_parser().parse_args([])
args.dataset = 'iapr'
args.data_root = '/home/bd/data/zs/FedMMDP/preprocessed_iapr/domain_datasets'
args.lr = 1e-5
args.local_epochs = 1
args.comm_rounds = 1
args.model = 'clip'
args.batch_size = 256
args.pub_data_num = 5000
args.device = 0

args.FL_algorithm = 'FedDF'
algo = FedDF(args, wandb=DummyWandb())
algo.load_dataset(args)
algo.create_model(args)
print('feddf before')
nan_params(algo.engine.model, 'feddf_server_before')
check_eval_features(algo.engine.evaluator, algo.val_dataloader[0], 'feddf_server_before')
algo.train(0)
print('feddf after')
nan_params(algo.engine.model, 'feddf_server_after')
check_eval_features(algo.engine.evaluator, algo.val_dataloader[0], 'feddf_server_after')

args2 = build_parser().parse_args([])
args2.FL_algorithm = 'FedMD'
args2.dataset = 'iapr'
args2.data_root = '/home/bd/data/zs/FedMMDP/preprocessed_iapr/domain_datasets'
args2.lr = 1e-5
args2.local_epochs = 1
args2.comm_rounds = 1
args2.model = 'clip'
args2.batch_size = 256
args2.pub_data_num = 5000
args2.device = 0
algo2 = FedMD(args2, wandb=DummyWandb())
algo2.load_dataset(args2)
algo2.create_model(args2)
mm = algo2.total_local_trainers[-1]
print('fedmd before')
nan_params(mm.model, 'fedmd_mm_before')
check_eval_features(mm.evaluator, algo2.val_dataloader[4], 'fedmd_mm_before')
algo2.train(0)
print('fedmd after')
nan_params(mm.model, 'fedmd_mm_after')
check_eval_features(mm.evaluator, algo2.val_dataloader[4], 'fedmd_mm_after')
