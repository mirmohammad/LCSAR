import os
from glob import glob

import numpy as np
from PIL import Image
from torch.utils import data
from torchvision import transforms

from utils import sampler


class SAR(data.Dataset):
    def __init__(self, root_dir, kernel=(224, 224), stride=(224, 224)):
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

        self.crops = sampler.SarSampler(lbl_file=self.lbl_file, kernel=kernel, stride=stride).get_crops()

        self.sar_tif = np.array(Image.open(self.sar_file))
        self.lbl_tif = np.array(Image.open(self.lbl_file))

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
