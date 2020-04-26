import time

import multiprocessing as mlt
from collections import Counter

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


class SarSampler:
    def __init__(self, lbl_file, kernel, stride, min_classes=2, max_count=0.8):
        assert len(kernel) == 2, 'argument "kernel" must be of size 2'
        assert len(stride) == 2, 'argument "stride" must be of size 2'

        self.kh = kernel[0]
        self.kw = kernel[1]
        self.min_classes = min_classes  # minimum number of classes
        self.count_thr = int((self.kh * self.kw) * max_count)  # maximum number of pixels of each class

        print(f'\nLoading image {lbl_file} ', end='...')
        start_time = time.time()

        # Much faster compared to default int
        # NOTE: Conversion to int8 is lossy (-999 becomes 25)
        self.img = np.array(Image.open(lbl_file), dtype=np.int8)

        print(f' Done in {(time.time() - start_time):.1f}s')
        print(f'TIF dimensions are (H, W): {self.img.shape}')

        sh = np.arange(0, self.img.shape[0] - self.kh + 1, stride[0])
        sw = np.arange(0, self.img.shape[1] - self.kw + 1, stride[1])

        xx, yy = np.meshgrid(sh, sw)
        crops = np.stack((xx, yy), axis=2).reshape(-1, 2)  # Top left corner

        # Values for number of processes and chunksize are experimental
        num_processes = 8  # crops.shape[0] // 12500
        chunksize = crops.shape[0] // num_processes

        print(f'\nAssessing crops using kernel=({self.kh}*{self.kw}) and stride=({stride[0]}*{stride[1]}) ', end='...')
        start_time = time.time()

        class_count = list()
        # Using multiprocessing to find unique pixels within each crop (intensive)
        with mlt.Pool(num_processes) as p:
            for m_class, m_count in p.imap(self.m_unique, crops, chunksize=chunksize):
                class_count.append((m_class, m_count))

        print(f' Done in {(time.time() - start_time):.1f}s')
        print(f'Total number of crops: {crops.shape[0]}')
        print(f'#crops per #classes: {Counter(map(lambda x: x[0].size, class_count))}')

        print('\nFiltering crops', end='...')
        start_time = time.time()

        self.selected = list()
        # Filter out crops with conditions evaluated using m_select function
        for i, (m_class, m_count) in enumerate(class_count):
            # First condition selects crops containing at least 2 different classes
            # Second condition selects crops containing at least 1:48 pixels of each class
            if m_class.size >= self.min_classes and (m_class != 0).all() and (m_count <= self.count_thr).all():
                self.selected.append({"crop": crops[i], "classes": m_class, "pixels": m_count})

        # new params for multiprocessing
        # num_processes = 2
        # # chunksize = chunksize // 2

        # with mlt.Pool(num_processes) as p:
        #     for i, (m_class, m_count, m_select) in enumerate(p.imap(self.m_select, class_count, chunksize=chunksize)):
        #         if m_select:
        #             self.selected.append({"crop": crops[i], "classes": m_class, "pixels": m_count})

        print(f' Done in {(time.time() - start_time):.1f}s')
        print(f'Number of selected crops: {len(self.selected)}')
        print(f'#selected per #classes: {Counter(map(lambda x: x["classes"].size, self.selected))}')

    def m_unique(self, m_crop):
        return \
            np.unique(self.img[m_crop[0]: m_crop[0] + self.kh, m_crop[1]: m_crop[1] + self.kw], return_counts=True)

    # def m_select(self, m_class_count):
    #     m_class, m_count = m_class_count
    #     return \
    #         m_class, \
    #         m_count, \
    #         m_class.size >= self.min_classes and (m_class != 0).all() and (m_count > self.count_thr).all()

    def get_crops(self):
        return self.selected
