"""MS-COCO image-to-caption retrieval dataset code

reference codes:
https://github.com/pytorch/vision/blob/v0.2.2_branch/torchvision/datasets/coco.py
https://github.com/yalesong/pvse/blob/master/data.py
"""

import os
from glob import glob

import torch

try:
    import ujson as json
except ImportError:
    import json

from pycocotools.coco import COCO
from torch.utils.data import Dataset

from src.datasets.coco_preprocess import build_cache_path, load_clip_cache


class CocoCaptionsCap(Dataset):
    """`MS Coco Captions <http://mscoco.org/dataset/#captions-challenge2015>`_ Dataset.
    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        ids (list, optional): list of target caption ids
        extra_annFile (string, optional): Path to extra json annotation file (for training)
        extra_ids (list, optional): list of extra target caption ids (for training)
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        instance_annFile (str, optional): Path to instance annotation json (for PMRP computation)

    Example:
        .. code:: python
            import torchvision.datasets as dset
            import torchvision.transforms as transforms
            cap = dset.CocoCaptions(root='dir where images are',
                                    annFile='json annotation file',
                                    transform=transforms.ToTensor())
            print('Number of samples: ', len(cap))
            img, target = cap[3] # load 4th sample
            print("Image Size: ", img.size())
            print(target)
        Output: ::
            Number of samples: 82783
            Image Size: (3L, 427L, 640L)
            [u'A plane emitting smoke stream flying over a mountain.',
            u'A plane darts across a bright blue sky behind a mountain covered in snow',
            u'A plane leaves a contrail above the snowy mountain top.',
            u'A mountain that has a plane flying overheard in the distance.',
            u'A mountain view with a plume of smoke in the background']
    """

    def __init__(self, root, annFile, ids=None,
                 extra_annFile=None, extra_ids=None,
                 instance_annFile=None, client=-1,
                 cache_file=None, cache_map_location="cpu"):
        self.root = os.path.expanduser(root)
        if extra_annFile:
            self.coco = COCO()
            with open(annFile, 'r') as fin1, open(extra_annFile, 'r') as fin2:
                dataset = json.load(fin1)
                extra_dataset = json.load(fin2)
                if not isinstance(dataset, dict) or not isinstance(extra_dataset, dict):
                    raise TypeError('invalid type {} {}'.format(type(dataset),
                                                                type(extra_dataset)))
                if set(dataset.keys()) != set(extra_dataset.keys()):
                    raise KeyError('key mismatch {} != {}'.format(list(dataset.keys()),
                                                                  list(extra_dataset.keys())))
                for key in ['images', 'annotations']:
                    dataset[key].extend(extra_dataset[key])
            self.coco.dataset = dataset
            self.coco.createIndex()
        else:
            self.coco = COCO(annFile)
        self.ids = list(self.coco.anns.keys()) if ids is None else list(ids)
        if extra_ids is not None:
            self.ids += list(extra_ids)
        self.ids = [int(id_) for id_ in self.ids]

        self.all_image_ids = set([self.coco.loadAnns(annotation_id)[0]['image_id'] for annotation_id in self.ids])

        iid_to_cls = {}
        if instance_annFile:
            for ins_file in glob(instance_annFile + '/instances_*'):
                with open(ins_file) as fin:
                    instance_ann = json.load(fin)
                for ann in instance_ann['annotations']:
                    image_id = int(ann['image_id'])
                    code = iid_to_cls.get(image_id, [0] * 90)
                    code[int(ann['category_id']) - 1] = 1
                    iid_to_cls[image_id] = code

                seen_classes = {}
                new_iid_to_cls = {}
                idx = 0
                for k, v in iid_to_cls.items():
                    v = ''.join([str(s) for s in v])
                    if v in seen_classes:
                        new_iid_to_cls[k] = seen_classes[v]
                    else:
                        new_iid_to_cls[k] = idx
                        seen_classes[v] = idx
                        idx += 1
                iid_to_cls = new_iid_to_cls

                if self.all_image_ids - set(iid_to_cls.keys()):
                    # print(f'Found mismatched! {self.all_image_ids - set(iid_to_cls.keys())}')
                    print(f'Found mismatched! {len(self.all_image_ids - set(iid_to_cls.keys()))}')

        self.iid_to_cls = iid_to_cls
        self.n_images = len(self.all_image_ids)

        cache_path = cache_file or build_cache_path(annFile, extra_annFile)
        cache = load_clip_cache(cache_path, map_location=cache_map_location)

        self.image_features = cache["image_features"].float()
        self.caption_features = cache["caption_features"].float()
        self.cache_image_ids = cache["image_ids"]
        self.cache_ann_ids = cache["ann_ids"]
        self.cache_captions = cache.get("captions", [])

        self.image_id_to_idx = {iid: idx for idx, iid in enumerate(self.cache_image_ids)}
        self.ann_id_to_idx = {ann_id: idx for idx, ann_id in enumerate(self.cache_ann_ids)}

        missing_ann = [ann_id for ann_id in self.ids if ann_id not in self.ann_id_to_idx]
        if missing_ann:
            sample = missing_ann[:5]
            raise KeyError(f"Missing {len(missing_ann)} annotation embeddings in cache {cache_path}, e.g. {sample}")

        missing_image_ids = self.all_image_ids - set(self.image_id_to_idx)
        if missing_image_ids:
            sample = list(missing_image_ids)[:5]
            raise KeyError(f"Missing {len(missing_image_ids)} image embeddings in cache {cache_path}, e.g. {sample}")

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is a caption for the annotation.
        """
        coco = self.coco
        annotation_id = self.ids[index]
        annotation = coco.loadAnns(annotation_id)[0]
        image_id = annotation['image_id']
        caption = annotation['caption']  # language caption

        img_idx = self.image_id_to_idx[image_id]
        caption_idx = self.ann_id_to_idx[annotation_id]

        img = self.image_features[img_idx]
        target = self.caption_features[caption_idx]
        if self.cache_captions:
            caption = self.cache_captions[caption_idx]

        return img, target, caption, annotation_id, image_id, index

    def __len__(self):
        return len(self.ids)
