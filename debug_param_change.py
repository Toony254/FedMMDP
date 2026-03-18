import sys
from pathlib import Path as _Path
sys.path.append(str(_Path(__file__).resolve().parent / 'src'))
sys.path.append(str(_Path(__file__).resolve().parent))
import hashlib
import numpy as np
import torch

from src.main import build_parser
from src.algorithms.FedDF import MMFL as FedDF
from src.algorithms.FedMD import MMFL as FedMD


class DummyWandb:
    def log(self, *args, **kwargs):
        return None


def digest_state_dict(state_dict):
    hasher = hashlib.sha1()
    total_norm = 0.0
    for k in sorted(state_dict.keys()):
        v = state_dict[k]
        if torch.is_tensor(v):
            arr = v.detach().float().cpu().numpy()
            hasher.update(k.encode())
            hasher.update(arr.tobytes())
            total_norm += float(np.abs(arr).sum())
    return hasher.hexdigest(), total_norm


def model_digest(model):
    return digest_state_dict(model.state_dict())

args = build_parser().parse_args([])
args.FL_algorithm = 'FedDF'
args.dataset = 'iapr'
args.data_root = '/home/bd/data/zs/FedMMDP/preprocessed_iapr/domain_datasets'
args.lr = 1e-5
args.local_epochs = 1
args.comm_rounds = 1
args.model = 'clip'
args.batch_size = 256
args.pub_data_num = 5000
args.device = 0
args.name = 'debug-param'

algo = FedDF(args, wandb=DummyWandb())
algo.load_dataset(args)
algo.create_model(args)
pre = model_digest(algo.engine.model)
print('feddf_server_before', pre)
first_mm = model_digest(algo.total_local_trainers[-1].model)
print('feddf_mm_before', first_mm)
algo.train(0)
post = model_digest(algo.engine.model)
print('feddf_server_after', post)
first_mm_after = model_digest(algo.total_local_trainers[-1].model)
print('feddf_mm_after', first_mm_after)

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
args2.name = 'debug-param2'

algo2 = FedMD(args2, wandb=DummyWandb())
algo2.load_dataset(args2)
algo2.create_model(args2)
pre2 = model_digest(algo2.total_local_trainers[-1].model)
print('fedmd_mm_before', pre2)
algo2.train(0)
post2 = model_digest(algo2.total_local_trainers[-1].model)
print('fedmd_mm_after', post2)
