import gc
import os
import numpy as np
import torch
import pickle
from datasets import load_from_disk
from src.datasets.transform import collate_fn
def get_FL_trainloader(dataset_name, data_root, num_clients, partition, alpha, batch_size):
    net_dataset_map_list = []
    test_loaders = []
    for domain in range(num_clients):
        dataset_path = os.path.join(data_root, f"domain_dataset_{domain}", "train")
        dataset = load_from_disk(dataset_path)
        train_test_split = dataset.train_test_split(test_size=0.1)
        train_set = train_test_split["train"]
        test_set = train_test_split["test"]

        if dataset_name == 'image':
            train_set = train_set.remove_columns(["cap_tokens"])
            
        elif dataset_name == 'text':
            train_set = train_set.remove_columns(["processed_img"])
    
        targets = train_set['class_id']
        num_samples = train_set.num_rows
        
        # data_root = "/home/bd/data/zs/FedMMDP/preprocessed_imagenet/domain_datasets/"
        # data_root = "/home/bd/data/zs/FedMMDP/preprocessed_fashion/domain_datasets/"
        # data_root = "/home/bd/data/zs/FedMMDP/preprocessed_food/domain_datasets/"
        dataset_name_from_path = data_root.split('preprocessed_')[-1].split('/')[0]
        check_dir = os.path.join('./data_partition/', dataset_name_from_path)
        if "ALIGN" in data_root:
            check_dir = './data_partition/imagenet_align_'
        elif "siglip" in data_root:
            check_dir = './data_partition/imagenet_siglip_'
        net_dataidx_map = data_partitioner(domain, num_samples, 3, partition=partition,
                                        check_dir=check_dir, alpha=alpha,
                                        y_train=np.array(targets))
        print(f"Samples Num: {[len(i) for i in net_dataidx_map.values()]}")
        
        net_dataset_map = {i: torch.utils.data.Subset(train_set, net_dataidx_map[i]) for i in net_dataidx_map.keys()}
        net_dataset_map_list.append(net_dataset_map)
        test_loaders.append(torch.utils.data.DataLoader(
            test_set,
            batch_size=batch_size * 2,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            collate_fn=collate_fn
        ))
    generator = torch.Generator()
    if dataset_name == 'image':
        loader_map = {
            i: torch.utils.data.DataLoader(
                net_dataset_map_list[i][0], 
                batch_size=batch_size, 
                shuffle=True, 
                num_workers=0,
                generator=generator,
                pin_memory=False,
                collate_fn=collate_fn,
                drop_last=True
            ) for i in range(num_clients)
        }
    elif dataset_name == 'text':
        loader_map = {
            i: torch.utils.data.DataLoader(
                net_dataset_map_list[i][1], 
                batch_size=batch_size, 
                shuffle=True, 
                num_workers=0,
                generator=generator,
                pin_memory=False,
                collate_fn=collate_fn,
                drop_last=True
            ) for i in range(num_clients)
        }
    elif dataset_name == 'mm':
        loader_map = {
            i: torch.utils.data.DataLoader(
                net_dataset_map_list[i][2], 
                batch_size=batch_size, 
                shuffle=True, 
                num_workers=0,
                generator=generator,
                pin_memory=False,
                collate_fn=collate_fn,
                drop_last=True
            ) for i in range(num_clients)
        }
    
    return loader_map, test_loaders


def data_partitioner(domain, num_samples, num_clients, partition='homo', check_dir=None, alpha=0.5, y_train=None):
    check_dir = check_dir + f'domain_{domain}'

    if partition == "homo":
        check_dir = check_dir + "_iid.pkl"
        if os.path.isfile(check_dir):
            net_dataidx_map = pickle.load(open(check_dir, 'rb'))
        else:
            idxs = np.random.permutation(num_samples)
            batch_idxs = np.array_split(idxs, num_clients)
            net_dataidx_map = {i: batch_idxs[i] for i in range(num_clients)}
            pickle.dump(net_dataidx_map, open(check_dir, 'wb'))

    elif partition == "hetero":
        check_dir = check_dir + "_noniid.pkl"
        if os.path.isfile(check_dir):
            net_dataidx_map = pickle.load(open(check_dir, 'rb'))
        else:
            min_size = 0
            unique_labels = np.unique(y_train)
            net_dataidx_map = {}
            print('Hetero partition')
            while min_size < 50: # min_size of single modal dataset per clients
                idx_batch = [[] for _ in range(num_clients)]
                # for each class in the dataset
                for k in unique_labels:
                    idx_k = np.where(y_train == k)[0]
                    np.random.shuffle(idx_k)
                    proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
                    ## Balance
                    proportions = np.array(
                        [p * (len(idx_j) < num_samples / num_clients) for p, idx_j in zip(proportions, idx_batch)])
                    proportions = proportions / proportions.sum()
                    proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                    idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                    min_size = min([len(idx_j) for idx_j in idx_batch])

            for j in range(num_clients):
                np.random.shuffle(idx_batch[j])
                net_dataidx_map[j] = idx_batch[j]

            pickle.dump(net_dataidx_map, open(check_dir, 'wb'))

    return net_dataidx_map
