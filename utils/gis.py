import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

from PIL import Image
import os
from glob import glob

Image.MAX_IMAGE_PIXELS = None

root_dir = '/data/Databases/MDA'
maps = ['Montreal', 'Ottawa', 'Quebec', 'Saskatoon', 'Toronto', 'Vancouver']

x = 0
for m in maps:
    print(m)
    lbl = os.path.join(root_dir, m, 'sar', '*.tif')
    x += os.path.getsize(sorted(glob(lbl))[0]) / 1e6
    print(os.path.getsize(sorted(glob(lbl))[0]) / 1e6)
    # x = np.array(Image.open(sorted(glob(lbl))[0]))
    # print(x.shape)

print(x)

exit(0)

x = Image.open('/data/Databases/MDA/Vancouver/lbl/Vancouver_F2NDR_LULC_Masked.tif')

x = np.array(img)

x[x == 21] = 1
x[x == 31] = 2
x[x == 41] = 3
x[x == 51] = 4
x[x == -999] = 5

print(x[x == 0])

print(np.unique(x))


