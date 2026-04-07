import importlib
import random
import argparse
import os
from pathlib import Path

from utils.helper import Helper as helper
from utils.model_utils import MODEL_CHOICES


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_COCO_ROOT = os.environ.get(
    'FEDMMDP_COCO_ROOT',
    str((PROJECT_ROOT / 'data' / 'MSCOCO' / '2014').resolve()),
)


ALGORITHM_MODULES = {
    'FedMMDP': 'algorithms.FedMMDP',
    'MASA': 'algorithms.MASA',
    'FedAvg': 'algorithms.FedAvg',
    'FedProx': 'algorithms.FedProx',
    'FedMD': 'algorithms.FedMD',
    'MOON': 'algorithms.MOON',
    'FedDF': 'algorithms.FedDF',
    'Harmony': 'algorithms.Harmony',
    'Cream': 'algorithms.Cream',
    'FedMobile': 'algorithms.FedMobile',
    'FedMEKT': 'algorithms.FedMEKT',
    'FedMEMA': 'algorithms.FedMEMA',
    'RawCLIP': 'algorithms.RawCLIP',
    'CenterTraining': 'algorithms.CenterTraining',
}


def init_wandb(args):
    import wandb

    wandb.init(
        project='FedMMDP',
        name=str(args.name),
        resume=None,
        config=args,
        mode='offline',
    )
    return wandb


def build_parser():
    parser = argparse.ArgumentParser(description='Federated Learning')
    parser.add_argument('--name', type=str, default='FedMMDP', help='The name for different experimental runs.')
    parser.add_argument('--exp_dir', type=str, default='./experiments/', help='Locations to save different experimental runs.')
    parser.add_argument('--FL_algorithm', type=str, default='FedMMDP', choices=list(ALGORITHM_MODULES.keys()), help='Federated learning algorithm to use.')
    parser.add_argument('--local_epochs', type=int, default=1)
    parser.add_argument('--comm_rounds', type=int, default=20)

    parser.add_argument('--model', type=str, default='clip', choices=list(MODEL_CHOICES))
    parser.add_argument('--pretrained', type=int, default=0)
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=random.randint(0, 100000), metavar='S', help='random seed')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--aggregate', action='store_true', default=False)

    parser.add_argument('--num_img_clients', type=int, default=5)
    parser.add_argument('--num_txt_clients', type=int, default=5)
    parser.add_argument('--num_mm_clients', type=int, default=5)
    parser.add_argument('--num_domains', type=int, default=5)
    parser.add_argument('--client_num_per_round', type=int, default=15)

    parser.add_argument('--mu', type=float, default=0.01, help='coefficient of mu')
    parser.add_argument('--con', type=float, default=1.0, help='coefficient of con')
    parser.add_argument('--temperature', type=float, default=0.5, help='contrastive temperature')

    parser.add_argument('--dataset', type=str, default='imagenet', choices=['imagenet', 'fashion', 'food', 'iapr'])
    parser.add_argument('--data_root', type=str, default='preprocessed_imagenet/domain_datasets/')
    parser.add_argument('--coco_root', type=str, default=DEFAULT_COCO_ROOT,
                        help='root directory for the public MSCOCO dataset used by distillation baselines')
    parser.add_argument('--batch_size', type=int, default=256, metavar='N', help='input batch size for training')
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--partition', type=str, default='hetero', help='data partition mode for all clients')
    parser.add_argument('--pub_data_num', type=int, default=5000, help='public dataset size for distillation-based baselines')

    parser.add_argument('--server_lr', type=float, default=0.0002)
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR', help='student learning rate')
    parser.add_argument('--loss', type=str, default='l1', choices=['l1', 'kl', 'l1softmax'])
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['multistep', 'cosine', 'exponential', 'none'])
    parser.add_argument('--steps', nargs='+', default=[0.05, 0.15, 0.3, 0.5, 0.75], type=float, help='percentage epochs at which to take next step')
    parser.add_argument('--scale', type=float, default=0.1, help='fractional decrease in lr')
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum')
    parser.add_argument('--disable_distill', action='store_true', default=False)

    parser.add_argument('--agg_method', type=str, default='con_w', help='representation aggregation method')
    parser.add_argument('--contrast_local_intra', action='store_true', default=True)
    parser.add_argument('--contrast_local_inter', action='store_true', default=True)
    parser.add_argument('--mlp_local', action='store_true', default=False)
    parser.add_argument('--cluster_weight', type=float, default=1.0)
    parser.add_argument('--rmg_weight', type=float, default=1.0)
    parser.add_argument('--loss_scale', action='store_true', default=False)
    parser.add_argument('--save_client', action='store_true', default=False)
    parser.add_argument('--kd_weight', type=float, default=0.3, help='coefficient of kd')
    parser.add_argument('--interintra_weight', type=float, default=0.5, help='coefficient of inter+intra')

    parser.add_argument('--num_relays', type=int, default=5)
    parser.add_argument('--num_clusters', type=int, default=5)
    parser.add_argument('--max_ssm_components', type=int, default=5)

    parser.add_argument('--data_local', action='store_true', default=False, help='change data directory to ~/data_local')
    parser.add_argument('--feature_dim', type=int, default=1024)
    parser.add_argument('--use_pretrained_proj', type=int, default=1, choices=[0, 1],
                        help='1: load pretrained projector weights, 0: use random projector initialization')
    parser.add_argument('--pretrained_proj_variant', type=str, default='',
                        help='optional pretrained projector variant tag, e.g. clonly, rmgonly, maxmargin')
    parser.add_argument('--pretrained_proj_path', type=str, default='',
                        help='optional explicit pretrained projector checkpoint path')
    parser.add_argument('--cluster_method', type=str, default='finch', choices=['finch', 'spectral', 'kmeans', 'dbscan'])
    parser.add_argument('--partition_level', type=int, default=1, help='partition level for FINCH clustering')
    parser.add_argument('--n_clusters', type=int, default=5, help='number of clusters for kmeans or spectral clustering')
    parser.add_argument('--eps', type=float, default=0.5, help='DBSCAN epsilon')
    parser.add_argument('--min_samples', type=int, default=10, help='DBSCAN min_samples')
    parser.add_argument('--disable_tsne', action='store_true', default=False, help='skip t-SNE visualization during training')
    parser.add_argument('--tau', type=float, default=0.5, help='temperature for FedMMDP domain contrastive loss')
    parser.add_argument('--centroid_init', type=str, default='random_unit', choices=['random_unit'], help='FedMMDP centroid initialization strategy')
    parser.add_argument('--secure_agg_mode', type=str, default='plaintext', choices=['plaintext'], help='FedMMDP secure aggregation backend')
    parser.add_argument('--fedmmdp_disable_cluster_loss', type=int, default=0, choices=[0, 1],
                        help='FedMMDP ablation flag: disable cluster-based domain supervision while keeping the rest unchanged')
    parser.add_argument('--fedmmdp_disable_rmg_loss', type=int, default=0, choices=[0, 1],
                        help='FedMMDP ablation flag: disable RMG loss on multimodal clients')
    parser.add_argument('--fedmmdp_cluster_inner_steps', type=int, default=1,
                        help='number of secure-aggregation Lloyd updates per communication round for FedMMDP')
    parser.add_argument('--fedmmdp_log_aux_losses', type=int, default=0, choices=[0, 1],
                        help='when enabled, FedMMDP stores auxiliary loss traces to a dedicated CSV file')
    parser.add_argument('--fedmmdp_ablation_tag', type=str, default='',
                        help='optional tag appended only to FedMMDP auxiliary artifacts for ablation bookkeeping')
    parser.add_argument('--fedmmdp_adaptive_aux_norm', type=int, default=1, choices=[0, 1],
                        help='when enabled, rescale FedMMDP auxiliary losses to the current base-loss magnitude on each client')
    parser.add_argument('--fedmmdp_aux_norm_eps', type=float, default=1e-6,
                        help='numerical stabilizer used by FedMMDP adaptive auxiliary-loss normalization')
    parser.add_argument('--fedmobile_gen_lr', type=float, default=1e-4)
    parser.add_argument('--fedmobile_gen_weight', type=float, default=0.5)
    parser.add_argument('--fedmobile_align_weight', type=float, default=0.1)
    parser.add_argument('--fedmobile_aux_ce_weight', type=float, default=0.5)
    parser.add_argument('--fedmobile_num_clusters', type=int, default=3)
    parser.add_argument('--fedmobile_shapley_samples', type=int, default=24)
    parser.add_argument('--fedmobile_eval_batches', type=int, default=1)
    parser.add_argument('--fedmobile_probe_labels', type=int, default=16)
    parser.add_argument('--fedmobile_noise_dim', type=int, default=128)
    parser.add_argument('--fedmobile_hidden_dim', type=int, default=2048)
    parser.add_argument('--fedmobile_local_lr', type=float, default=1e-4)
    parser.add_argument('--fedmobile_mm_lr', type=float, default=1e-4)
    parser.add_argument('--fedmekt_local_lr', type=float, default=1e-4)
    parser.add_argument('--fedmekt_mm_lr', type=float, default=1e-4)
    parser.add_argument('--fedmekt_local_align_weight', type=float, default=0.5)
    parser.add_argument('--fedmekt_server_align_weight', type=float, default=1.0)
    parser.add_argument('--fedmekt_server_retrieval_weight', type=float, default=1.0)
    parser.add_argument('--fedmekt_proxy_steps', type=int, default=16)
    parser.add_argument('--fedmekt_server_proxy_epochs', type=int, default=1)
    parser.add_argument('--harmony_stage1_rounds', type=int, default=0,
                        help='rounds used for Harmony modality-wise warmup; 0 selects an automatic split')
    parser.add_argument('--harmony_clusters', type=int, default=2,
                        help='number of multimodal clusters during Harmony fusion')
    parser.add_argument('--harmony_cluster_mode', type=str, default='fixed', choices=['fixed', 'svd'],
                        help='cluster count strategy for Harmony multimodal fusion')
    parser.add_argument('--harmony_svd_threshold', type=float, default=0.1,
                        help='relative singular value threshold when Harmony uses SVD-based cluster selection')
    parser.add_argument('--harmony_bias_metric', type=str, default='cosine', choices=['cosine'],
                        help='discrepancy metric used by Harmony for benchmark encoder comparison')
    return parser


def resolve_algorithm(algorithm_name):
    module = importlib.import_module(ALGORITHM_MODULES[algorithm_name])
    return module.MMFL


parser = build_parser()
args = parser.parse_args()

if __name__ == '__main__':
    wandb = init_wandb(args)
    algorithm_cls = resolve_algorithm(args.FL_algorithm)
    algo = algorithm_cls(args, wandb)

    args.save_dirs = helper.get_save_dirs(args.exp_dir, args.name)
    args.log_dir = args.save_dirs['logs']
    helper.set_seed(args.seed)

    algo.create_model(args)
    algo.load_dataset(args)

    for round_n in range(args.comm_rounds):
        algo.train(round_n)
