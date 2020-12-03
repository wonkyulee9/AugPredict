'''
Custom Dataset for Object Detection with SVHN
'''

import h5py
import os
import numpy as np
import torch
import pickle as pk
from PIL import Image
from torchvision import transforms


root_path = './datasets/SVHN_full'


def read_name(f, index):
    """Decode string from HDF5 file."""
    assert isinstance(f, h5py.File)
    assert index == int(index)
    ref = f['/digitStruct/name'][index][0]
    return ''.join(chr(v[0]) for v in f[ref])


def read_digits_raw(f, index):
    """Decode digits and bounding boxes from HDF5 file."""
    assert isinstance(f, h5py.File)
    assert index == int(index)

    ref = f['/digitStruct/bbox'][index].item()
    ddd = {}
    for key in ['label', 'left', 'top', 'width', 'height']:
        dset = f[ref][key]
        if len(dset) == 1:
            ddd[key] = [int(dset[0][0])]
        else:
            ddd[key] = []
            for i in range(len(dset)):
                ref2 = dset[i][0]
                ddd[key].append(int(f[ref2][0][0]))
    return ddd


def convert_labels(ddict):
    target = {}

    for idx, lab in enumerate(ddict['label']):
        if lab == 10:
            ddict['label'][idx] = 0

    target['labels'] = ddict['label']
    x1 = np.expand_dims(np.array(ddict['left']), 1)
    x2 = x1 + np.expand_dims(np.array(ddict['width']), 1)
    y1 = np.expand_dims(np.array(ddict['top']), 1)
    y2 = y1 + np.expand_dims(np.array(ddict['height']), 1)
    target['boxes'] = np.hstack((x1, y1, x2, y2))

    target['labels'] = torch.LongTensor(target['labels'])
    target['boxes'] = torch.FloatTensor(target['boxes'])
    return target


def get_targets(path):
    names = []
    targets = []
    with h5py.File(path) as f:
        length = len(f['digitStruct/name'])
        for i in range(length):
            name = read_name(f, i)
            names.append(name)
            ddict = read_digits_raw(f, i)
            target = convert_labels(ddict)
            targets.append(target)
    return names, targets


def create_pickle():
    for split in ['train', 'test']:
        if os.path.exists(root_path+'/'+split+'_names.pkl') and os.path.exists(root_path+'/'+split+'_targets.pkl'):
            return
        else:
            names, targets = get_targets('/'.join((root_path, split, "digitStruct.mat")))
            with open(root_path+'/'+split+'_names.pkl', 'wb') as f:
                pk.dump(names, f)
            with open(root_path+'/'+split+'_targets.pkl', 'wb') as f:
                pk.dump(targets, f)
    return


class SVHNFull(torch.utils.data.Dataset):
    def __init__(
            self,
            root: str = root_path,
            split: str = "train",
            transform = None,
            target_transform = None,
            download: bool = False,
            max_dim = 400
    ) -> None:
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.max_dim = max_dim
        if split == "train" or split == "test":
            self.split = split
        else:
            raise AssertionError("Split must be train or test")

        if download:
            raise AssertionError("Please download SVHN Full digits data to run this code")

        with open('/'.join((root, split, "_names.pkl")), 'rb') as f:
            self.names = pk.load(f)
        with open('/'.join((root, split, "_targets.pkl")), 'rb') as f:
            self.targets = pk.load(f)

    def __getitem__(self, index):
        img = Image.open(''.join((self.root, self.split, '/', self.names[index])))
        target = self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        x = img.shape[0]
        y = img.shape[1]

        size = x * y

        if size > self.max_dim * self.max_dim:
            r = y/x
            xt = torch.sqrt((self.max_dim ** 2) / r)
            yt = xt * r
            sc = xt / x

            img = transforms.Resize((xt, yt))
            target['boxes'] = target['boxes'] * sc

        return img, target

    def __len__(self):
        return len(self.names)

    def collate_fn(self, batch):
        images = list()
        targets = list()

        for b in batch:
            images.append(b[0])
            targets. append(b[1])

        return images, targets


if __name__ == '__main__':
    create_pickle()
