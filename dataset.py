__all__ = ['MultiScaleDataset',
           'ImageDataset',
           'MultiScalePatchDataset',
           'MultiScalePatchProgressivePairedDataset'
           ]

from io import BytesIO
import math

import lmdb
from PIL import Image
from torch.utils.data import Dataset
import torch
import numpy as np

import tensor_transforms as tt


class MultiScaleDataset(Dataset):
    def __init__(self, path, transform, resolution=256, to_crop=False, crop_size=64, integer_values=False):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        self.crop_size = crop_size
        self.integer_values = integer_values
        self.n = resolution // crop_size
        self.log_size = int(math.log(self.n, 2))
        self.crop = tt.RandomCrop(crop_size)
        self.crop_resolution = tt.RandomCrop(resolution)
        self.to_crop = to_crop
        self.coords = tt.convert_to_coord_format(1, resolution, resolution, integer_values=self.integer_values)

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

        self.resolution = resolution
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        data = {}

        with self.env.begin(write=False) as txn:
            key = f'{str(index).zfill(7)}'.encode('utf-8')
            img_bytes = txn.get(key)

        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        img = self.transform(img).unsqueeze(0)

        if self.to_crop:
            img = self.crop_resolution(img)

        stack = torch.cat([img, self.coords], 1)
        del img

        data[0] = self.crop(stack).squeeze(0)
        stack = stack.squeeze(0)

        stack_strided = None
        for ls in range(self.log_size, 0, -1):
            n = 2 ** ls
            bias = self.resolution - n*self.crop_size + n
            bw = np.random.randint(bias)
            bh = np.random.randint(bias)
            stack_strided = stack[:, bw::n, bh::n]
            if stack_strided.size(2) != self.crop or stack_strided.size(1) != self.crop:
                data[ls] = self.crop(stack_strided.unsqueeze(0)).squeeze(0)
            else:
                data[ls] = stack_strided

        del stack
        del stack_strided

        return data


class ImageDataset(Dataset):
    def __init__(self, path, transform, resolution=256, to_crop=False):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        self.crop = tt.RandomCrop(resolution)
        self.to_crop = to_crop

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

        self.resolution = resolution
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = f'{str(index).zfill(7)}'.encode('utf-8')
            img_bytes = txn.get(key)

        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        img = self.transform(img)
        if self.to_crop:
            img = self.crop(img.unsqueeze(0)).squeeze(0)

        return img


class MultiScalePatchDataset(Dataset):
    def __init__(self, path, transform, resolution=256, to_crop=False, crop_size=64, integer_values=False):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        self.crop_size = crop_size
        self.integer_values = integer_values
        self.n = resolution // crop_size
        self.log_size = int(math.log(self.n, 2))
        self.crop = tt.RandomCrop(crop_size)
        self.crop_resolution = tt.RandomCropWithReproducible(crop_size)
        self.to_crop = to_crop
        self.coords = tt.convert_to_coord_format(1, resolution, resolution, integer_values=self.integer_values)

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

        self.resolution = resolution
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        data = {}

        with self.env.begin(write=False) as txn:
            key = f'{str(index).zfill(7)}'.encode('utf-8')
            img_bytes = txn.get(key)

        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        img = self.transform(img).unsqueeze(0)

        if self.to_crop:
            img, (h, w, h_start, w_start, size) = self.crop_resolution(img)
            coords = self.coords[:, :, h_start: h_start + size, w_start: w_start + size]

        stack = torch.cat([img, coords], 1)
        del img

        data[0] = self.crop(stack).squeeze(0)
        stack = stack.squeeze(0)

        stack_strided = None
        # for ls in range(self.log_size, 0, -1):
        #     n = 2 ** ls
        #     bias = self.resolution - n*self.crop_size + n
        #     bw = np.random.randint(bias)
        #     bh = np.random.randint(bias)
        #     stack_strided = stack[:, bw::n, bh::n]
        #     if stack_strided.size(2) != self.crop or stack_strided.size(1) != self.crop:
        #         data[ls] = self.crop(stack_strided.unsqueeze(0)).squeeze(0)
        #     else:
        #         data[ls] = stack_strided

        del stack
        del stack_strided

        return data, torch.tensor(h_start), torch.tensor(w_start)  # torch.tensor((h_start/(h/2))-1), torch.tensor((w_start/(w/2))-1)


class MultiScalePatchProgressivePairedDataset(Dataset):
    def __init__(self, path, transform, resolution=256, resolution_bpg=256, to_crop=False, crop_size=64, integer_values=False):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        self.crop_size = crop_size
        self.integer_values = integer_values
        self.n = resolution // crop_size
        self.log_size = int(math.log(self.n, 2))
        self.crop = tt.RandomCrop(crop_size)
        self.crop_resolution = tt.RandomCropFromReproducible(crop_size)
        self.crop_resolution_bpg = tt.RandomCropWithReproducible(crop_size)
        self.to_crop = to_crop
        self.coords = tt.convert_to_coord_format(1, resolution, resolution, integer_values=self.integer_values)
        self.coords_bpg = tt.convert_to_coord_format(1, resolution_bpg, resolution_bpg, integer_values=self.integer_values)
        self.relative_resolution = resolution / resolution_bpg

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

        self.resolution = resolution
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        data = {}

        with self.env.begin(write=False) as txn:
            key = f'{str(index).zfill(7)}'.encode('utf-8')
            img_bytes = txn.get(key)

        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        img = self.transform(img).unsqueeze(0)

        if self.to_crop:
            coords_bpg, (h_bpg, w_bpg, h_start_bpg, w_start_bpg, size_bpg) = self.crop_resolution_bpg(self.coords_bpg)
            img, (h, w, h_start, w_start, size) = self.crop_resolution(img, int(h_start_bpg * self.relative_resolution), int((h_start_bpg+size_bpg) * self.relative_resolution), int(w_start_bpg * self.relative_resolution), int((w_start_bpg+size_bpg) * self.relative_resolution))
            coords = self.coords[:, :, h_start: h_start + size, w_start: w_start + size]

        stack = torch.cat([img, coords, coords_bpg], 1)
        del img

        data[0] = self.crop(stack).squeeze(0)
        stack = stack.squeeze(0)

        stack_strided = None
        # for ls in range(self.log_size, 0, -1):
        #     n = 2 ** ls
        #     bias = self.resolution - n*self.crop_size + n
        #     bw = np.random.randint(bias)
        #     bh = np.random.randint(bias)
        #     stack_strided = stack[:, bw::n, bh::n]
        #     if stack_strided.size(2) != self.crop or stack_strided.size(1) != self.crop:
        #         data[ls] = self.crop(stack_strided.unsqueeze(0)).squeeze(0)
        #     else:
        #         data[ls] = stack_strided

        del stack
        del stack_strided

        return data, torch.tensor(h_start), torch.tensor(w_start), torch.tensor(h_start_bpg), torch.tensor(w_start_bpg), (h, w, h_start, w_start, size), (h_bpg, w_bpg, h_start_bpg, w_start_bpg, size_bpg)