import multiprocessing as mlt
from collections import Counter

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


class SarSampler:
    def __init__(self, lbl_file, kernel, stride):
        assert len(kernel) == 2, 'argument "kernel" must be of size 2'
        assert len(stride) == 2, 'argument "stride" must be of size 2'

        self.kh = kernel[0]
        self.kw = kernel[1]
        self.img = np.array(Image.open(lbl_file))
        print(f'TIF dimensions are (H, W): {self.img.shape}')

        sh = np.arange(0, self.img.shape[0] - self.kh + 1, stride[0])
        sw = np.arange(0, self.img.shape[1] - self.kw + 1, stride[1])

        xx, yy = np.meshgrid(sh, sw)
        crops = np.stack((xx, yy), axis=2).reshape(-1, 2)  # Top left corner

        print(f'\nAssessing crops using kernel=({self.kh}*{self.kw}) and stride=({stride[0]}*{stride[1]}) ', end='...')

        classes = list()
        # Using multiprocessing to find unique pixels within each crop (intensive)
        # Values for number of processes and chunksize are experimental (i7-9700K)
        with mlt.Pool(8) as p:
            for m_class, m_count in p.imap(self.m_unique, crops, chunksize=12282):
                classes.append((m_class, m_count))

        print(' Done!')
        print(f'Total number of crops: {crops.shape[0]}')
        print(f'#crops per #classes: {Counter(map(lambda x: x[0].size, classes))}')
        print('\nFiltering crops', end='...')

        self.selected = list()
        # Filter out crops with following conditions
        for i, (m_class, m_count) in enumerate(classes):
            # First condition selects crops containing at least 2 different classes
            # Second condition selects crops containing at least 1:48 pixels of each class
            if m_class.size > 1 and (m_class != 0).all() and (m_count > (self.kh * self.kw) / 48).all():
                self.selected.append({"crop": crops[i], "classes": m_class, "pixels": m_count})

        print(' Done!')
        print(f'Number of selected crops: {len(self.selected)}')
        print(f'#selected per #classes: {Counter(map(lambda x: x["classes"].size, self.selected))}')

    def m_unique(self, crop):
        return np.unique(self.img[crop[0]: crop[0] + self.kh, crop[1]: crop[1] + self.kw], return_counts=True)

    def get_crops(self):
        return self.selected
