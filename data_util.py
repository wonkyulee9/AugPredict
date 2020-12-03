"""
Util for Data transforms and loading
"""

import torch
import random
from PIL import Image
import numpy as np

from torchvision import transforms, datasets
from torchvision.transforms import functional as F


class RandomCropWithLabel(transforms.RandomResizedCrop):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, img):
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        return F.resized_crop(img, i, j, h, w, self.size, self.interpolation), [i, j, h+i, w+j]


class RandomHFlipWithLabel(transforms.RandomHorizontalFlip):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, img):
        if torch.rand(1) < self.p:
            return F.hflip(img), [1]
        return img, [0]


class RandomGrayWithLabel(transforms.RandomGrayscale):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, img):
        num_output_channels = 1 if img.mode == 'L' else 3
        if random.random() < self.p:
            return F.to_grayscale(img, num_output_channels=num_output_channels), [1]
        return img, [0]


class ColorJitterWithLabel(transforms.ColorJitter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, img):
        brightness_factor, contrast_factor, saturation_factor, hue_factor = 0, 0, 0, 0
        fn_idx = torch.randperm(4)
        for fn_id in fn_idx:
            if fn_id == 0 and self.brightness is not None:
                brightness = self.brightness
                brightness_factor = torch.tensor(1.0).uniform_(brightness[0], brightness[1]).item()
                img = F.adjust_brightness(img, brightness_factor)

            if fn_id == 1 and self.contrast is not None:
                contrast = self.contrast
                contrast_factor = torch.tensor(1.0).uniform_(contrast[0], contrast[1]).item()
                img = F.adjust_contrast(img, contrast_factor)

            if fn_id == 2 and self.saturation is not None:
                saturation = self.saturation
                saturation_factor = torch.tensor(1.0).uniform_(saturation[0], saturation[1]).item()
                img = F.adjust_saturation(img, saturation_factor)

            if fn_id == 3 and self.hue is not None:
                hue = self.hue
                hue_factor = torch.tensor(1.0).uniform_(hue[0], hue[1]).item()
                img = F.adjust_hue(img, hue_factor)

        return img, [brightness_factor, contrast_factor, saturation_factor, hue_factor]


class CIFAR10WithAugLabel(datasets.CIFAR10):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.normalize = transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
        self.tensorize = transforms.ToTensor()

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        img_aug = img

        img_org = self.tensorize(img)
        img_org = self.normalize(img_org)

        imgs_list = []
        params_list = []
        for i in range(2):
            img_trans = img_aug
            aug_params = []
            for tran in self.transform:
                img_trans, aug_para = tran(img_trans)
                aug_para = torch.Tensor(aug_para)
                aug_params.append(aug_para)

            img_trans = self.tensorize(img_trans)
            img_trans = self.normalize(img_trans)
            imgs_list.append(img_trans)
            params_list.append(aug_params)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return [img_org, imgs_list[0], imgs_list[1]], target, params_list


class SVHNWithAugLabel(datasets.SVHN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.normalize = transforms.Normalize(mean=(0.4380, 0.4440, 0.4730), std=(0.1751, 0.1771, 0.1744))
        self.tensorize = transforms.ToTensor()

    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        img_aug = img

        img_org = self.tensorize(img)
        img_org = self.normalize(img_org)

        imgs_list = []
        params_list = []
        for i in range(2):
            img_trans = img_aug
            aug_params = []
            for tran in self.transform:
                img_trans, aug_para = tran(img_trans)
                aug_para = torch.Tensor(aug_para)
                aug_params.append(aug_para)

            img_trans = self.tensorize(img_trans)
            img_trans = self.normalize(img_trans)
            imgs_list.append(img_trans)
            params_list.append(aug_params)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return [img_org, imgs_list[0], imgs_list[1]], target, params_list


class RandRot90(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.angles = [0, 1, 2, 3]

    def forward(self, img):
        a = random.choice(self.angles)
        return torch.rot90(img, a, [2, 1]), a


class CIFAR10ROT(datasets.CIFAR10):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.normalize = transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
        self.tensorize = transforms.ToTensor()
        self.rotate = RandRot90()

    def __getitem__(self, index):
        img = self.data[index]
        img = Image.fromarray(img)

        img = self.tensorize(img)
        img = self.normalize(img)
        img, ang = self.rotate(img)
        return img, ang


class SVHNROT(datasets.SVHN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.normalize = transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
        self.tensorize = transforms.ToTensor()
        self.rotate = RandRot90()

    def __getitem__(self, index):
        img = self.data[index]
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))
        img = self.tensorize(img)
        img = self.normalize(img)
        img, ang = self.rotate(img)
        return img, ang


class SVHNROT2(datasets.SVHN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.normalize = transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
        self.tensorize = transforms.ToTensor()
        self.rotate = RandRot90()

    def __getitem__(self, index):
        img = self.data[index]
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))
        img = self.tensorize(img)
        img = self.normalize(img)
        img2, ang = self.rotate(img)
        return [img, img2], ang
