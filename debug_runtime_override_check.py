import sys
from pathlib import Path as _Path
sys.path.append(str(_Path(__file__).resolve().parent / 'src'))
sys.path.append(str(_Path(__file__).resolve().parent))

from src.main import build_parser
from src.algorithms.FedMD import MMFL

class DummyWandb:
    def log(self, *args, **kwargs):
        return None

args = build_parser().parse_args([
    '--FL_algorithm', 'FedMD',
    '--dataset', 'iapr',
    '--feature_dim', '768',
    '--batch_size', '64',
    '--alpha', '0.3',
    '--partition', 'homo',
    '--use_pretrained_proj', '0',
])
algo = MMFL(args, DummyWandb())
print('model_name', algo.config.model.name)
print('embed_dim', algo.config.model.embed_dim)
print('use_pretrained_proj', algo.config.model.use_pretrained_proj)
print('batch_size', algo.config.dataloader.batch_size)
print('eval_batch_size', getattr(algo.config.dataloader, 'eval_batch_size', None))
print('alpha', algo.config.dataloader.alpha)
print('partition', algo.config.dataloader.partition)
