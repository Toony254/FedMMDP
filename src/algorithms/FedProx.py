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
import copy


sys.path.append("./")
sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../")

from src.datasets.transform import collate_fn
from src.datasets.load_FL_datasets import get_FL_trainloader
from src.algorithms.ClientTrainer import ClientTrainer
from src.algorithms.MMClientTrainer import MMClientTrainer

from src.algorithms.eval_coco import COCOEvaluator
from src.algorithms.retrieval_trainer import TrainerEngine
from src.algorithms.mm_eval import MMEvaluator
from src.utils.config import parse_config, apply_runtime_overrides
from src.utils.load_datasets import prepare_coco_dataloaders
from src.utils.logger import PythonLogger
from src.utils.experiment_naming import projector_tag
from src.utils.model_utils import is_embedding_model

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
        if self.args.dataset == 'imagenet':
            self.class_size = 50
        elif self.args.dataset == 'fashion':
            self.class_size = 48
        elif self.args.dataset == 'food':
            self.class_size = 101
        elif self.args.dataset == 'iapr':
            self.class_size = 30  # 5 domains x 6 classes
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
        if self.args.dataset == 'imagenet':
            yaml_name = 'imageNet_cap.yaml'
        elif self.args.dataset == 'fashion':
            yaml_name = 'fashion_gen.yaml'
        elif self.args.dataset == 'food':
            yaml_name = 'umpc_food.yaml'
        elif self.args.dataset == 'iapr':
            yaml_name = 'iapr.yaml'
        self.config = parse_config("./src/" + yaml_name, strict_cast=False)
        self.config = apply_runtime_overrides(self.args, self.config)
        self.config.train.model_save_path = 'model_last_no_prob'
        self.config.train.best_model_save_path = 'model_best_no_prob'
        self.config.train.output_file = f'{self.args.name}_{self.args.dataset}_{self.args.model}_{self.args.lr}_{self.args.alpha}_{self.args.local_epochs}x{self.args.comm_rounds}_{projector_tag(self.args)}_model_noprob'
        self.config.model.name = self.args.model
        self.config.model.img_client = img
        self.config.model.txt_client = txt
        self.config.train.model_save_path = self.config.train.model_save_path + '.pth'
        self.config.train.best_model_save_path = self.config.train.best_model_save_path + '.pth'
        self.config.train.output_file = self.config.train.output_file + '.log'

        self.config.model.embed_dim = self.args.feature_dim  # set global model dim
    
    def load_dataset(self, args):
        self.engine = TrainerEngine()
        self.engine.set_logger(self.logger)

        self.config.optimizer.learning_rate = self.args.server_lr

        self.evaluator = MMEvaluator(model_name=self.args.model,
                                       dataset=self.args.dataset,
                                       eval_method='matmul',
                                       verbose=False,
                                       eval_device='cuda',
                                       n_crossfolds=1, 
                                       class_size=self.class_size,
                                        feature_dim=self.args.feature_dim,
                                        data_root=self.args.data_root)
        self.engine.create(self.config, self.evaluator, self.args.mlp_local)

        self.engine.model_to_device()
        torch.backends.cudnn.enabled = True
        if self.config.train.get('use_fp16'):
            self.engine.logger.log('Train with half precision')
            self.engine.to_half()
            
        self.val_dataloader = {}
        for i in range(args.num_img_clients):
            val_dataset = load_from_disk(os.path.join(self.args.data_root, f'domain_dataset_{i}', 'test'))
            self.val_dataloader[i] = torch.utils.data.DataLoader(val_dataset, 
                                                            batch_size=self.args.batch_size, 
                                                            shuffle=False, 
                                                            num_workers=4,
                                                            collate_fn=collate_fn
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
            self.img_trainloaders, test_loaders = get_FL_trainloader(dataset, self.args.data_root,
                                                                 args.num_img_clients, self.args.partition, self.args.alpha, self.args.batch_size)
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
            self.txt_trainloaders, test_loaders = get_FL_trainloader(dataset, self.args.data_root,
                                                                 args.num_txt_clients, self.args.partition, self.args.alpha, self.args.batch_size)
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
            if self.args.dataset == 'imagenet':
                yaml_name = 'imageNet_cap.yaml'
            elif self.args.dataset == 'fashion':
                yaml_name = 'fashion_gen.yaml'
            elif self.args.dataset == 'food':
                yaml_name = 'umpc_food.yaml'
            elif self.args.dataset == 'iapr':
                yaml_name = 'iapr.yaml'
            config = parse_config("./src/" + yaml_name, strict_cast=False)
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
    
    
    def aggregate_clip_models(self, local_image_models, local_text_models, local_mm_models):
        server_model = self.engine.model
        
        image_encoder_params = OrderedDict()
        
        all_image_encoders = []
        for model in local_image_models:
            all_image_encoders.append({
                'visual_projector': model.visual_projector.state_dict()
            })
        
        for model in local_mm_models:
            all_image_encoders.append({
                'visual_projector': model.img_enc.visual_projector.state_dict()
            })
        
        for key in all_image_encoders[0]['visual_projector'].keys():
            param_name = f'visual_projector.{key}'
            params = [enc['visual_projector'][key] for enc in all_image_encoders]
            orig_dtype = params[0].dtype
            avg_param = torch.mean(torch.stack([p.float() for p in params]), dim=0)
            image_encoder_params[param_name] = avg_param.to(orig_dtype)
        
        server_model.img_enc.load_state_dict(image_encoder_params)
        
        text_encoder_params = OrderedDict()
        
        all_text_encoders = []
        for model in local_text_models:
            all_text_encoders.append({
                'text_projector': model.text_projector.state_dict()
            })
        
        for model in local_mm_models:
            all_text_encoders.append({
                'text_projector': model.txt_enc.text_projector.state_dict()
            })
        
        for key in all_text_encoders[0]['text_projector'].keys():
            param_name = f'text_projector.{key}'
            params = [enc['text_projector'][key] for enc in all_text_encoders]
            orig_dtype = params[0].dtype
            avg_param = torch.mean(torch.stack([p.float() for p in params]), dim=0)
            text_encoder_params[param_name] = avg_param.to(orig_dtype)
        
        server_model.txt_enc.load_state_dict(text_encoder_params)
        
        return server_model
    
    def aggregate_resnet_models(self, local_image_models, local_text_models, local_mm_models):
        server_model = self.engine.model
        
        image_encoder_params = OrderedDict()
        
        all_image_encoders = []
        for model in local_image_models:
            model.to(self.device)
            all_image_encoders.append({
                'resnet': model.state_dict()
            })
        
        for model in local_mm_models:
            model.to(self.device)
            all_image_encoders.append({
                'resnet': model.img_enc.cnn.state_dict()
            })
        for key in all_image_encoders[0]['resnet'].keys():
            param_name = f'{key}'
            if all(key in enc['resnet'] for enc in all_image_encoders):
                params = [enc['resnet'][key] for enc in all_image_encoders]
                orig_dtype = params[0].dtype
                avg_param = torch.mean(torch.stack([p.float() for p in params]), dim=0)
                image_encoder_params[param_name] = avg_param.to(orig_dtype)
        
        server_model.img_enc.cnn.load_state_dict(image_encoder_params)
        
        text_encoder_params = OrderedDict()
        
        all_text_encoders = []
        for model in local_text_models:
            model.to(self.device)
            all_text_encoders.append({
                'text': model.state_dict()
            })
        
        for model in local_mm_models:
            all_text_encoders.append({
                'text': model.txt_enc.state_dict()
            })
            
        for key in all_text_encoders[0]['text'].keys():
            param_name = f'{key}'
            if all(key in enc['text'] for enc in all_text_encoders):
                params = [enc['text'][key] for enc in all_text_encoders]
                orig_dtype = params[0].dtype
                avg_param = torch.mean(torch.stack([p.float() for p in params]), dim=0)
                text_encoder_params[param_name] = avg_param.to(orig_dtype)
        
        server_model.txt_enc.load_state_dict(text_encoder_params)
        
        return server_model
    def train(self, round_n):
        self.cur_epoch = round_n
        self.cur_trainers = self.total_local_trainers

        if not is_test:
            self.logger.log(f"Round {round_n + 1}!")
            if len(self.total_local_trainers) != 0:
                self.cur_trainers = random.sample(self.total_local_trainers, self.args.client_num_per_round)

        # local training
        local_image_model = []
        local_text_model = []
        local_mm_model = []
        for idx, trainer in enumerate(self.cur_trainers):
            self.logger.log(f"Training Client {trainer.client_idx}!")
            trainer.cur_epoch = round_n
            trainer.global_model = copy.deepcopy(self.engine.model)
            trainer.run_with_prox()
            if trainer.dset_name == 'image':
                local_image_model.append(trainer.model)
            elif trainer.dset_name == 'text':
                local_text_model.append(trainer.model)
            elif trainer.dset_name == 'mm':
                local_mm_model.append(trainer.model)
        
        # aggregate local models
        if is_embedding_model(self.args.model):
            server_model = self.aggregate_clip_models(local_image_model, local_text_model, local_mm_model)
            self.engine.model = server_model
            for trainer in self.cur_trainers:
                if hasattr(trainer.model, "img_enc") and hasattr(trainer.model, "txt_enc"):
                    trainer.model.load_state_dict(server_model.state_dict())
                elif hasattr(trainer.model, "visual_projector") and hasattr(server_model.img_enc, "visual_projector"):
                    for name, param in server_model.img_enc.visual_projector.state_dict().items():
                        if name in trainer.model.visual_projector.state_dict():
                            trainer.model.visual_projector.state_dict()[name].copy_(param)
                elif hasattr(trainer.model, "text_projector") and hasattr(server_model.txt_enc, "text_projector"):
                    for name, param in server_model.txt_enc.text_projector.state_dict().items():
                        if name in trainer.model.text_projector.state_dict():
                            trainer.model.text_projector.state_dict()[name].copy_(param)
        elif self.args.model == 'resnet':
            server_model = self.aggregate_resnet_models(local_image_model, local_text_model, local_mm_model)
            self.engine.model = server_model
            for trainer in self.cur_trainers:
                if hasattr(trainer.model, "img_enc") and hasattr(trainer.model, "txt_enc"):
                    trainer.model.load_state_dict(server_model.state_dict())
                elif hasattr(trainer.model, "ResNet"):
                    for name, param in server_model.img_enc.cnn.state_dict().items():
                        if name in trainer.model.state_dict():
                            trainer.model.state_dict()[name].copy_(param)
                elif hasattr(trainer.model, "EncoderText"):
                    for name, param in server_model.txt_enc.cnn.state_dict().items():
                        if name in trainer.model.state_dict():
                            trainer.model.state_dict()[name].copy_(param)

        def get_lr(optimizer):
            for param_group in optimizer.param_groups:
                return param_group['lr']
        
        # test in own domain
        import csv
        
        if not hasattr(self, 'mm_results'):
            self.mm_results = []
        if not hasattr(self, 'rsum_history'):
            self.rsum_history = []
            
        mm_rows = []
        img_txt_rows = []
        
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
                domain_idx = idx - self.args.num_img_clients
                trainer.test_loader = self.val_dataloader[domain_idx]
                print(f"Client {trainer.dset_name} {idx} tests in domain {domain_idx}:")
                losses, test_top1, test_top5 = trainer.test()
                img_txt_rows.append([
                    round_n, trainer.client_idx, domain_idx,
                    losses, test_top1, test_top5
                ])
                
        for domain_idx in range(self.args.num_domains):
            print(f"Server tests in domain {domain_idx}:")
            test_scores = self.engine.evaluate({'test': self.val_dataloader[domain_idx]})
            metadata = self.engine.metadata.copy()
            metadata['cur_epoch'] = round_n + 1
            metadata['lr'] = get_lr(self.engine.optimizer)
            
            self.engine.report_scores(step=round_n + 1,
                                    scores=test_scores,
                                    metadata=metadata)
            rsum_i = test_scores['test']['i2t']['recall_1'] + test_scores['test']['t2i']['recall_1'] + \
                test_scores['test']['i2t']['recall_5'] + test_scores['test']['t2i']['recall_5']
            rsum += rsum_i
            mm_rows.append([
                round_n, domain_idx, rsum_i,
                test_scores['test']['n_fold']['i2t']['recall_1'],
                test_scores['test']['n_fold']['t2i']['recall_1'],
                test_scores['test']['i2t']['recall_1'],
                test_scores['test']['t2i']['recall_1'],
                test_scores['test']['n_fold']['i2t']['recall_5'],
                test_scores['test']['n_fold']['t2i']['recall_5'],
                test_scores['test']['i2t']['recall_5'],
                test_scores['test']['t2i']['recall_5'],
            ])
            self.wandb.log({f"Multimodal rsum_r1": rsum_i}, step=self.cur_epoch)
            self.wandb.log({f"Multimodal n_fold_i2t_r1": test_scores['test']['n_fold']['i2t']['recall_1']}, step=self.cur_epoch)
            self.wandb.log({f"Multimodal n_fold_t2i_r1": test_scores['test']['n_fold']['t2i']['recall_1']}, step=self.cur_epoch)
            self.wandb.log({f"Multimodal i2t_r1": test_scores['test']['i2t']['recall_1']}, step=self.cur_epoch)
            self.wandb.log({f"Multimodal t2i_r1": test_scores['test']['t2i']['recall_1']}, step=self.cur_epoch)
        
        self.rsum_history.append(rsum)
        if self.best_score < rsum:
            best_score = rsum
            metadata['best_score'] = best_score
            metadata['best_epoch'] = round_n + 1
            self.best_metadata, self.best_score = metadata, best_score
            print(f"Best score updated: {best_score} at epoch {round_n + 1}")
            # torch.save({'net': self.engine.model.state_dict()}, self.args.name + '-best_model.pt')

        if round_n == self.args.comm_rounds - 1:
            print(f"Final best score: {self.best_score} at epoch {self.best_metadata['best_epoch']}")
            # torch.save({'net': self.engine.model.state_dict()}, self.args.name + '-last_model.pt')
        
        os.makedirs('results', exist_ok=True)
                
        mm_csv = f'results/{self.args.name}_{self.args.dataset}_{self.args.model}_{self.args.lr}_{self.args.alpha}_{self.args.local_epochs}x{self.args.comm_rounds}_server_{self.args.FL_algorithm}_{projector_tag(self.args)}.csv'
        if mm_rows:
            write_header = not os.path.exists(mm_csv)
            with open(mm_csv, 'a', newline='') as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow([
                        'round', 'domain_idx', 'rsum_i',
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
        plt.savefig(f'results/rsum_{self.args.name}_{self.args.dataset}_{self.args.model}_{self.args.lr}_{self.args.alpha}_{self.args.local_epochs}x{self.args.comm_rounds}_{projector_tag(self.args)}.png')
        plt.close()
        print("Rsum at round {} is {}".format(round_n, self.rsum_history[-1]))
        gc.collect()
