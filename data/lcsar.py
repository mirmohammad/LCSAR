import os
from glob import glob
from itertools import accumulate

import numpy as np
from PIL import Image
from torch.utils import data

Image.MAX_IMAGE_PIXELS = None


class LCSAR(data.Dataset):
    def __init__(self, root_dir, maps, kernel, stride, transform):
        assert len(kernel) == 2, 'argument "kernel" must be of size 2'
        assert len(stride) == 2, 'argument "stride" must be of size 2'

        self.kh = kernel[0]
        self.kw = kernel[1]

        self.sh = stride[0]
        self.sw = stride[1]

        self.transform = transform

        sar_dirs = [os.path.join(root_dir, m, 'sar', '*.tif') for m in maps]
        lbl_dirs = [os.path.join(root_dir, m, 'lbl', '*.tif') for m in maps]

        sar_files = [sorted(glob(sar)) for sar in sar_dirs]
        lbl_files = [sorted(glob(lbl)) for lbl in lbl_dirs]

        for s, l in zip(sar_files, lbl_files):
            assert len(s) == 1, 'Only one TIF image is allowed in "sar" and "lbl" directories.'
            assert len(l) == 1, 'Only one TIF image is allowed in "sar" and "lbl" directories.'

        sar_files = [file[0] for file in sar_files]
        lbl_files = [file[0] for file in lbl_files]

        # Load SAR images at the end for better performance
        self.lbl = [np.array(Image.open(img), dtype=np.int8) for img in lbl_files]  # NOTE: Conversion to int8 is lossy (-999 becomes 25)

        self.crops = [self.sample(img) for img in self.lbl]

        # 0 and 25(-999) not present in selected crops
        for lbl in self.lbl:
            lbl[lbl == 21] = 0
            lbl[lbl == 31] = 1
            lbl[lbl == 41] = 2
            lbl[lbl == 51] = 3

        lens = [len(c) for c in self.crops]
        self.cum_lens = list(accumulate(lens))
        self.ranges = [range(right - left, right) for left, right in zip(lens, self.cum_lens)]

        self.sar = [np.array(Image.open(img)) for img in sar_files]

    def __len__(self):
        return self.cum_lens[-1]

    def __getitem__(self, idx):
        for i, r in enumerate(self.ranges):
            if idx in r:
                h, w = self.crops[i][idx - r[0]][0], self.crops[i][idx - r[0]][1]
                sar = self.sar[i][h: h + self.kh, w: w + self.kw]
                lbl = self.lbl[i][h: h + self.kh, w: w + self.kw]

                # NOTE: sar & lbl are numpy arrays not PIL images
                if self.transform:
                    sar = self.transform(sar)
                    lbl = self.transform(lbl)

                return sar, lbl

    def sample(self, img):
        xx, yy = np.meshgrid(np.arange(0, img.shape[0] - self.kh + 1, self.sh), np.arange(0, img.shape[1] - self.kw + 1, self.sw))
        crops = np.stack((xx, yy), axis=2).reshape(-1, 2)  # Top left corner
        classes = [np.unique(img[crop[0]: crop[0] + self.kh, crop[1]: crop[1] + self.kw]) for crop in crops]
        return [crop for crop, cls in zip(crops, classes) if (cls != 0).all() and (cls != 25).all()]
