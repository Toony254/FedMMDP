import gc
import random

import os
import sys
from sklearn.metrics.pairwise import cosine_similarity
from datasets import load_from_disk
from collections import OrderedDict

import operator
import torch.nn as nn
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm


sys.path.append("./")
sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../")

from src.datasets.transform import collate_fn
from src.datasets.load_FL_datasets import get_FL_trainloader, get_class_size
from src.algorithms.ClientTrainer import ClientTrainer
from src.algorithms.MMClientTrainer import MMClientTrainer

from src.algorithms.eval_coco import COCOEvaluator
from src.algorithms.retrieval_trainer import TrainerEngine
from src.algorithms.mm_eval import MMEvaluator
from src.utils.config import parse_config
from src.utils.load_datasets import prepare_coco_dataloaders
from src.utils.logger import PythonLogger

try:
    from apex import amp
except ImportError:
    print('failed to import apex')


is_test = False


class MMFL(object):
    def __init__(self, args, wandb=None):
        self.args = args
        self.wandb = wandb

        self.device = None
        self.img_local_trainers = None
        self.txt_local_trainers = None
        self.mm_local_trainers = None
        # self.engine = None
        self.class_size = get_class_size('/home/bd/data/zs/FedMMDP/data/processed_datasets/domain_dataset_0') * 5
        self.engine = None
        self.best_score = 0
        self.cur_epoch = 0
        self.best_metadata = None

        # img & txt local dataloaders
        self.img_train_loaders, self.txt_train_loaders = None, None

        # coco global dataloaders
        self.dataloaders_global = None
        # universal test dataloader
        self.test_loader = None

        self.config = None
        self.set_config()

        self.logger = PythonLogger(output_file=self.config.train.output_file)
        self.img_vec, self.txt_vec = None, None
        self.global_img_feature = None
        self.global_txt_feature = None
        self.distill_index = None


    def set_config(self, img='image', txt='text'):
        self.config = parse_config("./src/imageNet_cap.yaml", strict_cast=False)
        self.config.train.model_save_path = 'model_last_no_prob'
        self.config.train.best_model_save_path = 'model_best_no_prob'
        self.config.train.output_file = 'model_noprob'
        self.config.model.name = self.args.model
        self.config.model.img_client = img
        self.config.model.txt_client = txt
        self.config.train.model_save_path = self.config.train.model_save_path + '.pth'
        self.config.train.best_model_save_path = self.config.train.best_model_save_path + '.pth'
        self.config.train.output_file = self.config.train.output_file + '.log'

        self.config.model.embed_dim = self.args.feature_dim  # set global model dim
    
    def load_dataset(self, args):
        dataset_root = '/home/bd/data/zs' + '/data/mmdata/MSCOCO/2014'
        vocab_path = './src/datasets/vocabs/coco_vocab.pkl'
        self.dataloaders_global, self.vocab = prepare_coco_dataloaders(self.config.dataloader, dataset_root, vocab_path, subset_num=self.args.pub_data_num)

        self.engine = TrainerEngine()
        self.engine.set_logger(self.logger)

        self.config.optimizer.learning_rate = self.args.server_lr

        self._dataloaders = self.dataloaders_global.copy()
        self.evaluator = MMEvaluator(model_name=self.args.model,
                                       eval_method='matmul',
                                       verbose=False,
                                       eval_device='cuda',
                                       n_crossfolds=5, 
                                       class_size=self.class_size)
        self.engine.create(self.config, self.evaluator, self.args.mlp_local)

        self.train_eval_dataloader = self._dataloaders.pop(
            'train_subset_eval' + f'_{self.args.pub_data_num}') if self._dataloaders is not None else None

        self.engine.model_to_device()
        torch.backends.cudnn.enabled = True
        if self.config.train.get('use_fp16'):
            self.engine.logger.log('Train with half precision')
            self.engine.to_half()
            
        self.val_dataloader = {}
        for i in range(args.num_img_clients):
            val_dataset = load_from_disk(f'/home/bd/data/zs/FedMMDP/data/processed_datasets/domain_dataset_{i}_test')
            self.val_dataloader[i] = torch.utils.data.DataLoader(val_dataset, 
                                                            batch_size=self.args.batch_size, 
                                                            shuffle=False, 
                                                            num_workers=0,
                                                            collate_fn=collate_fn,
                                                            # timeout=300
                                                            )

    def create_model(self, args):
        self.logger.log('start creating model and partition datasets')
        self.device = torch.device("cuda:%d" % args.device)

        os.makedirs('/home/bd/data/zs' + f'/data/yClient', exist_ok=True)

        # Create Client Models
        self.img_local_trainers, self.txt_local_trainers, self.mm_local_trainers = [], [], []
        # img clients
        if args.num_img_clients > 0:
            dataset = 'image'
            self.img_trainloaders, test_loaders = get_FL_trainloader(dataset, '/home/bd/data/zs/FedMMDP/data/processed_datasets/',
                                                                 args.num_img_clients, "hetero", self.args.alpha, self.args.batch_size)
            self.img_local_trainers = []
            for i in range(args.num_img_clients):
                self.img_local_trainers.append(
                    ClientTrainer(args, dataset, self.class_size, self.logger,
                                  inter_distance=4, client_id=i, wandb=self.wandb))
                self.img_local_trainers[i].train_loader = self.img_trainloaders[i]
                self.img_local_trainers[i].test_loader = test_loaders[i]
                if is_test and i == 0:
                    break
        # txt clients
        if args.num_txt_clients > 0:
            dataset = 'text'
            self.txt_trainloaders, test_loaders = get_FL_trainloader(dataset, '/home/bd/data/zs/FedMMDP/data/processed_datasets/',
                                                                 args.num_txt_clients, "hetero", self.args.alpha, self.args.batch_size)
            self.txt_local_trainers = []
            for i in range(args.num_txt_clients):
                self.txt_local_trainers.append(
                    ClientTrainer(args, dataset, self.class_size, self.logger,
                                  inter_distance=4, client_id=i, wandb=self.wandb))
                self.txt_local_trainers[i].train_loader = self.txt_trainloaders[i]
                self.txt_local_trainers[i].test_loader = test_loaders[i]
                if is_test and i == 0:
                    break
        # mm clients
        if args.num_mm_clients > 0:
            # mm img models
            config = parse_config("./src/imageNet_cap.yaml", strict_cast=False)
            config.model.cache_dir = config.model.cache_dir + '-' + config.train.server_dataset
            config.train.output_file = os.path.join(config.model.cache_dir, config.train.output_file)
            config.train.best_model_save_path = os.path.join(config.model.cache_dir, config.train.best_model_save_path)
            config.train.model_save_path = os.path.join(config.model.cache_dir, config.train.model_save_path)
            config.model.embed_dim = self.args.feature_dim
            config.model.name = self.args.model
            self.mm_local_trainers = []
            for client_id in range(args.num_mm_clients):
                self.mm_local_trainers.append(
                    MMClientTrainer(args, config, self.class_size, self.logger, client=client_id, dset_name="mm",
                                    device='cuda', mlp_local=self.args.mlp_local, wandb=self.wandb))
                if is_test and client_id == 0:
                    break
            print(f"Samples Num: {[len(i.train_loader.dataset) for i in self.mm_local_trainers]}")

        self.total_local_trainers = self.img_local_trainers + self.txt_local_trainers + self.mm_local_trainers

        for i in range(len(self.total_local_trainers)):
            self.total_local_trainers[i].client_idx = i

    def train(self, round_n):
        self.cur_epoch = round_n
        self.cur_trainers = self.total_local_trainers

        if not is_test:
            self.logger.log(f"Round {round_n + 1}!")
            if len(self.total_local_trainers) != 0:
                self.cur_trainers = random.sample(self.total_local_trainers, self.args.client_num_per_round)
        

        alignment_loader = self.train_eval_dataloader
        img_logits = []
        txt_logits = []
        for trainer in self.cur_trainers:
            if trainer.dset_name == 'image':
                logits = trainer.predict_logits(alignment_loader)
                # print(logits.shape)
                img_logits.append(logits)
            elif trainer.dset_name == 'text':
                logits = trainer.predict_logits(alignment_loader)
                # print(logits.shape)
                txt_logits.append(logits)
            elif trainer.dset_name == 'mm':
                img, txt = trainer.predict_logits(alignment_loader)
                # print(img.shape, txt.shape)
                img_logits.append(img)
                txt_logits.append(txt)

        avg_img_logits = np.mean(np.stack(img_logits), axis=0)
        avg_txt_logits = np.mean(np.stack(txt_logits), axis=0)

        if round_n > 0:
            for trainer in self.cur_trainers:
                trainer.distill_with_logits(alignment_loader, avg_img_logits, avg_txt_logits)

        for trainer in self.cur_trainers:
            trainer.run()

        def get_lr(optimizer):
            for param_group in optimizer.param_groups:
                return param_group['lr']
        
        # test in own domain
        import csv

        if not hasattr(self, 'img_txt_results'):
            self.img_txt_results = []
        if not hasattr(self, 'mm_results'):
            self.mm_results = []
        if not hasattr(self, 'rsum_history'):
            self.rsum_history = []

        img_txt_rows = []
        mm_rows = []
        
        print("Testing...")
        rsum = 0
        for idx, trainer in enumerate(self.total_local_trainers):
            if trainer.dset_name == "image":
                domain_idx = idx
                trainer.test_loader = self.val_dataloader[idx]
                print(f"Client {trainer.dset_name} {idx} tests in domain {idx}:")
                losses, test_top1, test_top5 = trainer.test()
                img_txt_rows.append([
                    round_n, trainer.client_idx, domain_idx,
                    losses, test_top1, test_top5
                ])
            elif trainer.dset_name == "text":
                domain_idx = idx - 5
                trainer.test_loader = self.val_dataloader[domain_idx]
                print(f"Client {trainer.dset_name} {idx} tests in domain {domain_idx}:")
                losses, test_top1, test_top5 = trainer.test()
                img_txt_rows.append([
                    round_n, trainer.client_idx, domain_idx,
                    losses, test_top1, test_top5
                ])
            else:
                for domain_idx in range(self.args.num_domains):
                    print(f"Client {trainer.dset_name} {idx} tests in domain {domain_idx}:")
                    test_scores = trainer.evaluate({'test': self.val_dataloader[domain_idx]})
                    metadata = trainer.metadata.copy()
                    metadata['cur_epoch'] = round_n + 1
                    metadata['lr'] = get_lr(trainer.optimizer)
                    trainer.report_scores(step=round_n + 1,
                                            scores=test_scores,
                                            metadata=metadata)
                    rsum_i = test_scores['test']['i2t']['recall_1'] + test_scores['test']['t2i']['recall_1'] + \
                        test_scores['test']['i2t']['recall_5'] + test_scores['test']['t2i']['recall_5']
                    if domain_idx == idx - 10:
                        rsum += rsum_i
                    mm_rows.append([
                        round_n, trainer.client_idx, domain_idx, rsum_i,
                        test_scores['test']['n_fold']['i2t']['recall_1'],
                        test_scores['test']['n_fold']['t2i']['recall_1'],
                        test_scores['test']['i2t']['recall_1'],
                        test_scores['test']['t2i']['recall_1'],
                        test_scores['test']['n_fold']['i2t']['recall_5'],
                        test_scores['test']['n_fold']['t2i']['recall_5'],
                        test_scores['test']['i2t']['recall_5'],
                        test_scores['test']['t2i']['recall_5'],
                    ])
                    self.wandb.log({f"Multimodal_{idx-10} rsum_r1": rsum_i}, step=self.cur_epoch)
                    self.wandb.log({f"Multimodal_{idx-10} n_fold_i2t_r1": test_scores['test']['n_fold']['i2t']['recall_1']}, step=self.cur_epoch)
                    self.wandb.log({f"Multimodal_{idx-10} n_fold_t2i_r1": test_scores['test']['n_fold']['t2i']['recall_1']}, step=self.cur_epoch)
                    self.wandb.log({f"Multimodal_{idx-10} i2t_r1": test_scores['test']['i2t']['recall_1']}, step=self.cur_epoch)
                    self.wandb.log({f"Multimodal_{idx-10} t2i_r1": test_scores['test']['t2i']['recall_1']}, step=self.cur_epoch)
        
        self.rsum_history.append(rsum)
        if self.best_score < rsum:
            best_score = rsum
            metadata['best_score'] = best_score
            metadata['best_epoch'] = round_n + 1
            self.best_metadata, self.best_score = metadata, best_score
            print(f"Best score updated: {best_score} at epoch {round_n + 1}")
            # torch.save({'net': trainer.model.state_dict()}, self.args.name + '-best_model.pt')

        if round_n == self.args.comm_rounds - 1:
            print(f"Final best score: {self.best_score} at epoch {self.best_metadata['best_epoch']}")
        #     torch.save({'net': trainer.model.state_dict()}, self.args.name + '-last_model.pt')
        
        os.makedirs('results', exist_ok=True)
        mm_csv = f'results/mm_{self.args.FL_algorithm}.csv'

        if mm_rows:
            write_header = not os.path.exists(mm_csv)
            with open(mm_csv, 'a', newline='') as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow([
                        'round', 'client_id', 'domain_idx', 'rsum_i',
                        'n_fold_i2t_r1', 'n_fold_t2i_r1', 'i2t_r1', 't2i_r1',
                        'n_fold_i2t_r5', 'n_fold_t2i_r5', 'i2t_r5', 't2i_r5'
                    ])
                writer.writerows(mm_rows)
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(range(1, len(self.rsum_history)+1), self.rsum_history, marker='o')
        plt.xlabel('Round')
        plt.ylabel('rsum')
        plt.title(f'rsum Curve (Best: {self.best_score} at epoch {self.best_metadata["best_epoch"]})')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'results/rsum_{self.args.FL_algorithm}_{self.args.lr}_{self.args.alpha}_{self.args.local_epochs}x{self.args.comm_rounds}.png')
        plt.close()
        print("Rsum at round {} is {}".format(round_n, self.rsum_history[-1]))
        gc.collect()
