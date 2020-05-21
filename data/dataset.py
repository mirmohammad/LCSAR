import os
from glob import glob

import numpy as np
from PIL import Image
from torch.utils import data
from torchvision import transforms

from utils import SarSampler


class SAR(data.Dataset):
    def __init__(self, root_dir, kernel=(224, 224), stride=(224, 224), min_classes=2, max_count=0.8):
        assert len(kernel) == 2, 'argument "kernel" must be of size 2'
        assert len(stride) == 2, 'argument "stride" must be of size 2'

        self.kh = kernel[0]
        self.kw = kernel[1]

        self.sar_dir = os.path.join(root_dir, 'sar', '*.tif')
        self.lbl_dir = os.path.join(root_dir, 'lbl', '*.tif')

        self.sar_file = sorted(glob(self.sar_dir))
        self.lbl_file = sorted(glob(self.lbl_dir))

        assert len(self.sar_file) == 1, 'Only one TIF image is allowed in "sar" and "lbl" directories.'
        assert len(self.lbl_file) == 1, 'Only one TIF image is allowed in "sar" and "lbl" directories.'

        self.sar_file = self.sar_file[0]
        self.lbl_file = self.lbl_file[0]

        self.crops = SarSampler(lbl_file=self.lbl_file,
                                kernel=kernel,
                                stride=stride,
                                min_classes=min_classes,
                                max_count=max_count).get_crops()

        self.sar_tif = np.array(Image.open(self.sar_file))
        self.lbl_tif = np.array(Image.open(self.lbl_file), dtype=np.int8)

        # zero stays zero
        self.lbl_tif[self.lbl_tif == 21] = 1
        self.lbl_tif[self.lbl_tif == 31] = 2
        self.lbl_tif[self.lbl_tif == 41] = 3
        self.lbl_tif[self.lbl_tif == 51] = 4
        self.lbl_tif[self.lbl_tif == 25] = 5  # 25 is originally -999 (lossy conversion to int8)

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.crops)

    def __getitem__(self, idx):
        oh, ow = self.crops[idx]["crop"][0], self.crops[idx]["crop"][1]
        sar = self.sar_tif[oh: oh + self.kh, ow: ow + self.kw]
        lbl = self.lbl_tif[oh: oh + self.kh, ow: ow + self.kw]

        if self.transform:
            sar = self.transform(sar)
            lbl = self.transform(lbl)

        return sar, lbl


# Multi SAR dataset
class MSAR(data.Dataset):
    def __init__(self, root_dir, maps, train=True, kernel=(224, 224), stride=(224, 224), min_classes=2, max_count=0.8):
        assert 'train' in maps if train else 'valid' in maps, 'argument "maps" must contain either train or valid pairs'

        maps = maps['train' if train else 'valid']
        maps = [SAR(os.path.join(root_dir, x), kernel, stride, min_classes, max_count) for x in maps]

        self.lens = [len(x) for x in maps]
        self.rngs = []

        left = 0
        rngs = []
        for x in maps:
            rngs.append(range(left, left + len(x)))
            left = left + len(x)

        self.rngs = rngs
        self.maps = maps

        print(f'\n === Total {sum(self.lens)} in {self.lens} as {self.rngs} ===')

    def __len__(self):
        return sum(self.lens)

    def __getitem__(self, idx):
        for i, r in enumerate(self.rngs):
            if idx in r:
                idx = idx - sum(self.lens[0:i])
                return self.maps[i].__getitem__(idx)
