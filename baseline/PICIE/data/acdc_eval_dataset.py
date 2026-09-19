import os
import json
import torch
import torch.utils.data as data
import torch.nn.functional as F
import numpy as np


class EvalACDC(data.Dataset):
    def __init__(self, root, split, mode, res=224, split_manifest=None, label=True):
        self.root = root
        self.split = split
        self.mode = mode
        self.res = res
        self.label = label
        self.view = -1

        self.split_manifest = split_manifest
        self._volumes_cache = {}
        self._masks_cache = {}

        with open(os.path.join(root, 'metadata.json')) as f:
            self._metadata = json.load(f)
        self._vol_info = self._metadata.get('volume_info', {})

        self.slices = self._build_slice_index()

    def _build_slice_index(self):
        with open(self.split_manifest) as f:
            manifest = json.load(f)
        vol_names = manifest['splits'][self.split]['volumes']

        slices = []
        for vol_name in sorted(vol_names):
            info = self._vol_info.get(vol_name, {})
            n_slices = info.get('num_slices', None)
            if n_slices is None:
                vol = np.load(os.path.join(self.root, 'volumes', vol_name + '.npy'),
                              mmap_mode='r')
                n_slices = vol.shape[2]
            spacing = tuple(info.get('effective_spacing', [1.0, 1.0, 1.0]))
            for z in range(n_slices):
                slices.append((vol_name, z, n_slices, spacing))
        return slices

    def _load_volume(self, vol_name):
        if vol_name not in self._volumes_cache:
            path = os.path.join(self.root, 'volumes', vol_name + '.npy')
            self._volumes_cache[vol_name] = np.load(path).astype(np.float32)
        return self._volumes_cache[vol_name]

    def _load_mask(self, vol_name):
        if vol_name not in self._masks_cache:
            path = os.path.join(self.root, 'masks', vol_name + '.npy')
            self._masks_cache[vol_name] = np.load(path).astype(np.int64)
        return self._masks_cache[vol_name]

    def _get_25d_slice(self, vol_name, z, n_slices):
        vol = self._load_volume(vol_name)
        z_prev = max(0, z - 1)
        z_next = min(n_slices - 1, z + 1)
        slice_25d = np.stack([vol[:, :, z_prev],
                              vol[:, :, z],
                              vol[:, :, z_next]], axis=0)
        return torch.from_numpy(slice_25d.copy())

    def __getitem__(self, index):
        vol_name, z, n_slices, spacing = self.slices[index]

        image = self._get_25d_slice(vol_name, z, n_slices)
        _, h, w = image.shape
        if h != self.res or w != self.res:
            image = F.interpolate(image.unsqueeze(0), size=self.res,
                                  mode='bilinear', align_corners=False).squeeze(0)

        if not self.label:
            return index, image, None

        mask_vol = self._load_mask(vol_name)
        label_2d = mask_vol[:, :, z]
        label = torch.from_numpy(label_2d.copy())

        meta = {
            'vol_name': vol_name,
            'slice_idx': z,
            'n_slices': n_slices,
            'spacing': spacing,
            'orig_h': h,
            'orig_w': w,
        }

        return index, image, label, meta

    def __len__(self):
        return len(self.slices)
