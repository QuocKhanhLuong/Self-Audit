import os
import json
import torch
import torch.utils.data as data
import torch.nn.functional as F
import numpy as np
from data.custom_transforms import (
    RandomHorizontalTensorFlip, RandomVerticalFlip, RandomResizedCrop
)


class TrainACDC(data.Dataset):
    def __init__(self, root, labeldir, mode, split='train', res1=224, res2=224,
                 inv_list=[], eqv_list=[], scale=(0.5, 1), split_manifest=None):
        self.root = root
        self.split = split
        self.res1 = res1
        self.res2 = res2
        self.mode = mode
        self.scale = scale
        self.view = -1

        self.inv_list = inv_list
        self.eqv_list = eqv_list
        self.labeldir = labeldir

        self.split_manifest = split_manifest
        self._volumes_cache = {}

        self.slices = self._build_slice_index()
        self.reshuffle()

    def _build_slice_index(self):
        with open(self.split_manifest) as f:
            manifest = json.load(f)
        vol_names = manifest['splits'][self.split]['volumes']

        slices = []
        for vol_name in sorted(vol_names):
            vol_path = os.path.join(self.root, 'volumes', vol_name + '.npy')
            vol = np.load(vol_path, mmap_mode='r')
            n_slices = vol.shape[2]
            for z in range(n_slices):
                slices.append((vol_name, z, n_slices))
        return slices

    def _load_volume(self, vol_name):
        if vol_name not in self._volumes_cache:
            vol_path = os.path.join(self.root, 'volumes', vol_name + '.npy')
            self._volumes_cache[vol_name] = np.load(vol_path).astype(np.float32)
        return self._volumes_cache[vol_name]

    def _get_25d_slice(self, vol_name, z, n_slices):
        vol = self._load_volume(vol_name)
        z_prev = max(0, z - 1)
        z_next = min(n_slices - 1, z + 1)
        slice_25d = np.stack([vol[:, :, z_prev],
                              vol[:, :, z],
                              vol[:, :, z_next]], axis=0)
        return torch.from_numpy(slice_25d.copy())

    def __getitem__(self, index):
        index = self.shuffled_indices[index]
        vol_name, z, n_slices = self.slices[index]

        image = self._get_25d_slice(vol_name, z, n_slices)
        image = self._resize(image, self.res2)

        image = self.transform_image(index, image)
        label = self.transform_label(index)

        return (index,) + image + label

    def _resize(self, tensor, target_res):
        _, h, w = tensor.shape
        if h == target_res and w == target_res:
            return tensor
        return F.interpolate(tensor.unsqueeze(0), size=target_res,
                             mode='bilinear', align_corners=False).squeeze(0)

    def transform_image(self, index, image):
        if self.mode == 'compute':
            if self.view == 1:
                return (image,)
            elif self.view == 2:
                image = self._resize(image, self.res1)
                return (image,)
            else:
                raise ValueError('View [{}] is an invalid option.'.format(self.view))
        elif 'train' in self.mode:
            image1 = image
            if self.mode == 'baseline_train':
                return (image1,)
            image2 = self._resize(image, self.res1)
            return (image1, image2)
        else:
            raise ValueError('Mode [{}] is an invalid option.'.format(self.mode))

    def transform_label(self, index):
        if self.mode == 'train':
            label1 = torch.load(os.path.join(self.labeldir, 'label_1', '{}.pkl'.format(index)), weights_only=False)
            label2 = torch.load(os.path.join(self.labeldir, 'label_2', '{}.pkl'.format(index)), weights_only=False)
            label1 = torch.LongTensor(label1)
            label2 = torch.LongTensor(label2)
            X1 = int(np.sqrt(label1.shape[0]))
            X2 = int(np.sqrt(label2.shape[0]))
            label1 = label1.view(X1, X1)
            label2 = label2.view(X2, X2)
            return label1, label2
        elif self.mode == 'baseline_train':
            label1 = torch.load(os.path.join(self.labeldir, 'label_1', '{}.pkl'.format(index)), weights_only=False)
            label1 = torch.LongTensor(label1)
            X1 = int(np.sqrt(label1.shape[0]))
            label1 = label1.view(X1, X1)
            return (label1,)
        return (None,)

    def reshuffle(self):
        self.shuffled_indices = np.arange(len(self.slices))
        np.random.shuffle(self.shuffled_indices)
        self.init_transforms()

    def init_transforms(self):
        N = len(self.slices)
        self.random_horizontal_flip = RandomHorizontalTensorFlip(N=N)
        self.random_vertical_flip = RandomVerticalFlip(N=N)
        self.random_resized_crop = RandomResizedCrop(N=N, res=self.res1, scale=self.scale)

    def transform_eqv(self, indice, image):
        if 'random_crop' in self.eqv_list:
            image = self.random_resized_crop(indice, image)
        if 'h_flip' in self.eqv_list:
            image = self.random_horizontal_flip(indice, image)
        if 'v_flip' in self.eqv_list:
            image = self.random_vertical_flip(indice, image)
        return image

    def __len__(self):
        return len(self.slices)
