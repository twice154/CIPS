import random

import torch


def convert_to_coord_format(b, h, w, device='cpu', integer_values=False):
    if integer_values:
        x_channel = torch.arange(w, dtype=torch.float, device=device).view(1, 1, 1, -1).repeat(b, 1, w, 1)
        y_channel = torch.arange(h, dtype=torch.float, device=device).view(1, 1, -1, 1).repeat(b, 1, 1, h)
    else:
        x_channel = torch.linspace(-1.5, 1.5, int(1.5*w), device=device).view(1, 1, 1, -1).repeat(b, 1, h, 1)
        y_channel = torch.linspace(-1, 1, h, device=device).view(1, 1, -1, 1).repeat(b, 1, 1, int(1.5*w))
        # x_channel = torch.linspace(-1, 1, w, device=device).view(1, 1, 1, -1).repeat(b, 1, w, 1)
        # y_channel = torch.linspace(-1, 1, h, device=device).view(1, 1, -1, 1).repeat(b, 1, 1, h)
    return torch.cat((x_channel, y_channel), dim=1)


def random_crop(tensor, size):
    assert tensor.dim() == 4, tensor.shape  # frames x channels x h x w
    h, w = tensor.shape[-2:]
    h_start = random.randint(0, h - size) if h - size > 0 else 0
    w_start = random.randint(0, w - size) if w - size > 0 else 0
    return tensor[:, :, h_start: h_start + size, w_start: w_start + size]


def random_crop_with_reproducible(tensor, size):
    assert tensor.dim() == 4, tensor.shape  # frames x channels x h x w
    h, w = tensor.shape[-2:]
    h_start = random.randint(0, h - size) if h - size > 0 else 0
    w_start = random.randint(0, w - size) if w - size > 0 else 0
    return tensor[:, :, h_start: h_start + size, w_start: w_start + size], (h, w, h_start, w_start, size)


def random_crop_from_reproducible(tensor, size, h_start_from, h_start_to, w_start_from, w_start_to):
    assert tensor.dim() == 4, tensor.shape  # frames x channels x h x w
    h, w = tensor.shape[-2:]
    h_start = random.randint(h_start_from, h_start_to - size) if h_start_to - size > 0 else 0
    w_start = random.randint(w_start_from, w_start_to - size) if w_start_to - size > 0 else 0
    return tensor[:, :, h_start: h_start + size, w_start: w_start + size], (h, w, h_start, w_start, size)


class RandomCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, tensor):
        return random_crop(tensor, self.size)


class RandomCropWithReproducible:
    def __init__(self, size):
        self.size = size

    def __call__(self, tensor):
        return random_crop_with_reproducible(tensor, self.size)


class RandomCropFromReproducible:
    def __init__(self, size):
        self.size = size

    def __call__(self, tensor, h_start_from, h_start_to, w_start_from, w_start_to):
        return random_crop_from_reproducible(tensor, self.size, h_start_from, h_start_to, w_start_from, w_start_to)


def random_horizontal_flip(tensor):
    flip = random.randint(0, 1)
    if flip:
        return tensor.flip(-1)
    else:
        return tensor


def identity(tensor):
    return tensor
