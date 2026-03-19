import os
import random
import copy
import sys
import time

import numpy as np
import pickle
from torch.utils.data import DataLoader

sys.path.append("./")
sys.path.append("../")
sys.path.append("../..")
sys.path.append("../../..")

from src.datasets._dataloader import image_to_caption_collate_fn
from src.datasets.coco import CocoCaptionsCap
from src.datasets.coco_preprocess import load_clip_cache, precompute_coco_embeddings
from src.utils.model_utils import normalize_model_name


#  COCO
def _get_coco_cache_path(dataset_root, model_name, feature_dim):
    model_name = normalize_model_name(model_name)
    return os.path.join(dataset_root, f'coco_emb_{model_name}_{feature_dim}.pt')


def _validate_coco_cache(cache_path, feature_dim, model_name):
    cache = load_clip_cache(cache_path, map_location='cpu')
    normalized_model_name = normalize_model_name(model_name)
    cache_model_name = normalize_model_name(cache.get('model_name', normalized_model_name))
    image_dim = cache['image_features'].shape[-1]
    caption_dim = cache['caption_features'].shape[-1]
    if cache_model_name != normalized_model_name:
        raise ValueError(
            f"COCO cache model mismatch for {cache_path}: cache_model={cache_model_name}, expected={normalized_model_name}"
        )
    if image_dim != feature_dim or caption_dim != feature_dim:
        raise ValueError(
            f"COCO cache dimension mismatch for {cache_path}: "
            f"image_dim={image_dim}, caption_dim={caption_dim}, expected={feature_dim}"
        )


def ensure_coco_embedding_cache(dataset_root,
                                image_root,
                                train_ann,
                                val_ann,
                                feature_dim,
                                clip_model_name='RN50',
                                model_name='clip',
                                device=None,
                                batch_size=256,
                                wait_seconds=10):
    cache_path = _get_coco_cache_path(dataset_root, model_name, feature_dim)
    if os.path.exists(cache_path):
        _validate_coco_cache(cache_path, feature_dim, model_name)
        return cache_path

    os.makedirs(dataset_root, exist_ok=True)
    lock_path = cache_path + '.lock'

    while os.path.exists(lock_path):
        if os.path.exists(cache_path):
            _validate_coco_cache(cache_path, feature_dim, model_name)
            return cache_path
        print(f'Waiting for COCO cache generation: {lock_path}')
        time.sleep(wait_seconds)

    try:
        with open(lock_path, 'w', encoding='utf-8') as fout:
            fout.write(str(os.getpid()))

        if not os.path.exists(cache_path):
            print(f'COCO cache not found, generating {cache_path}')
            precompute_coco_embeddings(
                root=image_root,
                ann_file=train_ann,
                extra_ann_file=val_ann,
                output_path=cache_path,
                model_name=model_name,
                clip_model_name=clip_model_name,
                batch_size=batch_size,
                device=device,
                feature_dim=feature_dim,
            )

        _validate_coco_cache(cache_path, feature_dim, model_name)
    finally:
        if os.path.exists(lock_path):
            os.remove(lock_path)

    return cache_path


def prepare_coco_dataloaders(dataloader_config,
                             dataset_root,
                             vocab_path='./vocabs/coco_vocab.pkl',
                             num_workers=0,
                             tsne=False,
                             client=-1,
                             pub_data_num=50000,
                             feature_dim=1024,
                             clip_model_name='RN50',
                             model_name='clip',
                             cache_device=None):
    """Prepare MS-COCO Caption train / val / test dataloaders
    Args:
        dataloader_config (dict): configuration file which should contain "batch_size"
        dataset_root (str): root of your MS-COCO dataset (see README.md for detailed dataset hierarchy)
        vocab_path (str, optional): path for vocab pickle file (default: ./vocabs/coco_vocab.pkl).
        num_workers (int, optional): num_workers for the dataloaders (default: 6)
    Returns:
        dataloaders (dict): keys = ["train", "val", "te"], values are the corresponding dataloaders.
        vocab (Vocabulary object): vocab object
    """
    batch_size = dataloader_config['batch_size']
    tr_cutout_prob = dataloader_config.get('random_erasing_prob', 0.0)
    tr_caption_drop_prob = dataloader_config.get('caption_drop_prob', 0.0)
    eval_batch_size = dataloader_config.get('eval_batch_size', batch_size)

    vocab = load_vocab(vocab_path)
    train_ids, train_extra_ids, val_ids, te_ids, image_root, train_ann, val_ann = _get_coco_file_paths(dataset_root)

    cache_file = ensure_coco_embedding_cache(
        dataset_root=dataset_root,
        image_root=image_root,
        train_ann=train_ann,
        val_ann=val_ann,
        feature_dim=feature_dim,
        clip_model_name=clip_model_name,
        model_name=model_name,
        device=cache_device,
        batch_size=max(batch_size, eval_batch_size),
    )

    dataloaders = {}

    # dataloaders['train'] = _get_coco_loader(
    #     image_root, train_ann, train_ids, vocab,
    #     num_workers=num_workers, batch_size=batch_size,
    #     train=True,
    #     extra_annotation_path=val_ann,
    #     extra_ids=train_extra_ids,
    #     cutout_prob=tr_cutout_prob,
    #     caption_drop_prob=tr_caption_drop_prob,
    # )

    if tsne:
        pass
    elif client > -1:
        dataloaders['train_client'] = _get_coco_loader(
            image_root, train_ann, train_ids, vocab,
            num_workers=num_workers, batch_size=batch_size,
            train=True,
            extra_annotation_path=val_ann,
            extra_ids=train_extra_ids,
            cutout_prob=tr_cutout_prob,
            caption_drop_prob=tr_caption_drop_prob,
            subset=False,
            client=client,
            cache_file=cache_file
        )
    else:
        dataloaders['train_subset' + f'_{pub_data_num}'] = _get_coco_loader(
            image_root, train_ann, train_ids, vocab,
            num_workers=num_workers, batch_size=batch_size,
            train=True,
            extra_annotation_path=val_ann,
            extra_ids=train_extra_ids,
            cutout_prob=tr_cutout_prob,
            caption_drop_prob=tr_caption_drop_prob,
            subset=True,
            pub_data_num=pub_data_num,
            cache_file=cache_file
        )

        dataloaders['train_subset_eval' + f'_{pub_data_num}'] = _get_coco_loader(
            image_root, train_ann, train_ids, vocab,
            num_workers=num_workers, batch_size=batch_size * 2,
            train=False,
            extra_annotation_path=val_ann,
            extra_ids=train_extra_ids,
            cutout_prob=tr_cutout_prob,
            caption_drop_prob=tr_caption_drop_prob,
            subset=True,
            pub_data_num=pub_data_num,
            cache_file=cache_file
        )

    dataloaders['val'] = _get_coco_loader(
        image_root, val_ann, val_ids, vocab,
        num_workers=num_workers, batch_size=eval_batch_size,
        train=False,
        cache_file=cache_file,
    )

    dataloaders['test'] = _get_coco_loader(
        image_root, val_ann, te_ids, vocab,
        num_workers=num_workers, batch_size=eval_batch_size if not tsne else 200,
        train=False,
        cache_file=cache_file,
    )

    return dataloaders, vocab


def _get_coco_file_paths(dataset_root):
    """Select proper train / val classes and omit id files.
    """
    train_ids = np.load('./src/datasets/annotations/coco_train_ids.npy')
    train_extra_ids = np.load('./src/datasets/annotations/coco_restval_ids.npy')
    val_ids = np.load('./src/datasets/annotations/coco_dev_ids.npy')[:5000]
    te_ids = np.load('./src/datasets/annotations/coco_test_ids.npy')

    image_root = os.path.join(dataset_root, 'allimages')
    train_ann = os.path.join(dataset_root, 'annotations/captions_train2014.json')
    val_ann = os.path.join(dataset_root, 'annotations/captions_val2014.json')

    return train_ids, train_extra_ids, val_ids, te_ids, image_root, train_ann, val_ann


def _get_coco_loader(image_root,
                     annotation_path,
                     ids, vocab,
                     num_workers,
                     batch_size=64,
                     train=False,
                     extra_ids=None,
                     extra_annotation_path=None,
                     cutout_prob=0.0,
                     caption_drop_prob=0.0,
                     subset=False,
                     pub_data_num=50000,
                     client=-1,
                     cache_file=None):
    _image_transform = imagenet_transform(
        random_resize_crop=train,
        random_erasing_prob=cutout_prob,
    )
    _caption_transform = caption_transform(vocab,
                                           caption_drop_prob)

    coco_dataset = CocoCaptionsCap(image_root, annotation_path,
                                   extra_annFile=extra_annotation_path,
                                   ids=ids, cache_file=cache_file,
                                   extra_ids=extra_ids, client=client)

    if subset:
        subset_cache_dir = os.path.join(dataset_root, '.cache')
        os.makedirs(subset_cache_dir, exist_ok=True)
        subset_idx_file = os.path.join(subset_cache_dir, f'coco_subset_idx_{pub_data_num}.pkl')
        subset_lock_path = subset_idx_file + '.lock'
        while os.path.exists(subset_lock_path):
            if os.path.exists(subset_idx_file):
                break
            time.sleep(1)
        if not os.path.exists(subset_idx_file):
            try:
                with open(subset_lock_path, 'w', encoding='utf-8') as fout:
                    fout.write(str(os.getpid()))
                if not os.path.exists(subset_idx_file):
                    full_idx = [i for i in range(566435)]
                    random.shuffle(full_idx)
                    idx = full_idx[0: pub_data_num]
                    idx.sort()
                    with open(subset_idx_file, 'wb') as f:
                        pickle.dump(idx, f)
            finally:
                if os.path.exists(subset_lock_path):
                    os.remove(subset_lock_path)

        with open(subset_idx_file, 'rb') as f:
            idx = pickle.load(f)

        coco_dataset = torch.utils.data.Subset(coco_dataset, idx)

    elif client > -1:
        idx = [i for i in range(100000+client*10000, 110000+client*10000)]
        coco_dataset = torch.utils.data.Subset(coco_dataset, idx)

    dataloader = DataLoader(coco_dataset,
                            batch_size=batch_size,
                            shuffle=train,
                            num_workers=num_workers,
                            collate_fn=image_to_caption_collate_fn,
                            pin_memory=True)
    if subset or client > -1:
        print(f'Loading COCO Caption: n_captions {len(coco_dataset)}...')
    else:
        print(f'Loading COCO Caption: n_images {coco_dataset.n_images} n_captions {len(coco_dataset)}...')
    return dataloader


def load_vocab(vocab_path):
    if isinstance(vocab_path, str):
        vocab = Vocabulary()
        vocab.load_from_pickle(vocab_path)
    else:
        vocab = vocab_path
    return vocab


class Vocabulary(object):
    """Simple vocabulary wrapper."""

    def __init__(self):
        self.idx = 0
        self.word2idx = {}
        self.idx2word = {}

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def load_from_pickle(self, data_path):
        with open(data_path, 'rb') as fin:
            data = pickle.load(fin)
        self.idx = data['idx']
        self.word2idx = data['word2idx']
        self.idx2word = data['idx2word']

    def __call__(self, word):
        if word not in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

from functools import partial

from clip import tokenize as word_tokenize

import random
import math
import torch
from torchvision import transforms


def imagenet_normalize():
    """Standard ImageNet normalize transform
    """
    return transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])


def imagenet_transform(resize_size=256,
                       crop_size=224,
                       random_resize_crop=False,
                       random_erasing_prob=0.0,
                       custom_transforms=None):
    """Standard ImageNet transform with resize/crop/normalize.

    Args:
        resize_size (int, Default: 256): resize for validation
            (only used when random_resize_crop is False).
        crop_size (int, Default: 224): final crop size.
        random_resize_crop (bool, Default: False): if True, use random transform (for training),
            if False, use center crop (for validation).
        custom_transforms (list of transform, Default: None): additional transforms.
    """
    if custom_transforms is not None:
        if not isinstance(custom_transforms, list):
            raise TypeError(f'custom_transforms should be list, not {type(custom_transforms)}')
    transform = []
    if random_resize_crop:
        transform.append(transforms.RandomResizedCrop(crop_size))
        transform.append(transforms.RandomHorizontalFlip())
    else:
        transform.append(transforms.Resize(resize_size))
        transform.append(transforms.CenterCrop(crop_size))
    transform.append(transforms.ToTensor())
    transform.append(imagenet_normalize())

    if custom_transforms:
        transform.extend(custom_transforms)

    if random_erasing_prob > 0:
        print(f'adding cutout {random_erasing_prob}')
        transform.append(RandomErasing(random_erasing_prob,
                                       mode='const',
                                       max_count=1, num_splits=0, device='cpu'))

    transform = transforms.Compose(transform)
    return transform


def tokenize(sentence, vocab, caption_drop_prob):
    """nltk word_tokenize for caption transform.
    """
    tokens = word_tokenize(sentence, truncate=True)[0]
    return torch.Tensor(tokens)


def caption_transform(vocab, caption_drop_prob=0):
    """Transform for captions.
    "caption drop augmentation" randomly alters the given input tokens as <unk>
    """
    transform = []
    if caption_drop_prob < 0 or caption_drop_prob is None:
        print('warning: wrong caption drop prob', caption_drop_prob, 'set to zero')
        caption_drop_prob = 0
    elif caption_drop_prob > 0:
        print('adding caption drop prob', caption_drop_prob)
    transform.append(partial(tokenize, vocab=vocab, caption_drop_prob=caption_drop_prob))
    transform = transforms.Compose(transform)
    return transform


def _get_pixels(per_pixel, rand_color, patch_size, dtype=torch.float32, device='cuda'):
    # NOTE I've seen CUDA illegal memory access errors being caused by the normal_()
    # paths, flip the order so normal is run on CPU if this becomes a problem
    # Issue has been fixed in master https://github.com/pytorch/pytorch/issues/19508
    if per_pixel:
        return torch.empty(patch_size, dtype=dtype, device=device).normal_()
    elif rand_color:
        return torch.empty((patch_size[0], 1, 1), dtype=dtype, device=device).normal_()
    else:
        return torch.zeros((patch_size[0], 1, 1), dtype=dtype, device=device)


class RandomErasing:
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf

        This variant of RandomErasing is intended to be applied to either a batch
        or single image tensor after it has been normalized by dataset mean and std.
    Args:
         probability: Probability that the Random Erasing operation will be performed.
         min_area: Minimum percentage of erased area wrt input image area.
         max_area: Maximum percentage of erased area wrt input image area.
         min_aspect: Minimum aspect ratio of erased area.
         mode: pixel color mode, one of 'const', 'rand', or 'pixel'
            'const' - erase block is constant color of 0 for all channels
            'rand'  - erase block is same per-channel random (normal) color
            'pixel' - erase block is per-pixel random (normal) color
        max_count: maximum number of erasing blocks per image, area per box is scaled by count.
            per-image count is randomly chosen between 1 and this value.
    """

    def __init__(
            self,
            probability=0.5, min_area=0.02, max_area=1 / 3, min_aspect=0.3, max_aspect=None,
            mode='const', min_count=1, max_count=None, num_splits=0, device='cuda'):
        self.probability = probability
        self.min_area = min_area
        self.max_area = max_area
        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))
        self.min_count = min_count
        self.max_count = max_count or min_count
        self.num_splits = num_splits
        mode = mode.lower()
        self.rand_color = False
        self.per_pixel = False
        if mode == 'rand':
            self.rand_color = True  # per block random normal
        elif mode == 'pixel':
            self.per_pixel = True  # per pixel random normal
        else:
            assert not mode or mode == 'const'
        self.device = device

    def _erase(self, img, chan, img_h, img_w, dtype):
        if random.random() > self.probability:
            return
        area = img_h * img_w
        count = self.min_count if self.min_count == self.max_count else \
            random.randint(self.min_count, self.max_count)
        for _ in range(count):
            for attempt in range(10):
                target_area = random.uniform(self.min_area, self.max_area) * area / count
                aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))
                if w < img_w and h < img_h:
                    top = random.randint(0, img_h - h)
                    left = random.randint(0, img_w - w)
                    img[:, top:top + h, left:left + w] = _get_pixels(
                        self.per_pixel, self.rand_color, (chan, h, w),
                        dtype=dtype, device=self.device)
                    break

    def __call__(self, input):
        if len(input.size()) == 3:
            self._erase(input, *input.size(), input.dtype)
        else:
            batch_size, chan, img_h, img_w = input.size()
            # skip first slice of batch if num_splits is set (for clean portion of samples)
            batch_start = batch_size // self.num_splits if self.num_splits > 1 else 0
            for i in range(batch_start, batch_size):
                self._erase(input[i], chan, img_h, img_w, input.dtype)
        return input
