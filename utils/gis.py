import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

from PIL import Image

Image.MAX_IMAGE_PIXELS = None

img = Image.open('/data/Databases/MDA/Vancouver/lbl/Vancouver_F2NDR_LULC_Masked.tif')

x = np.array(img)

x[x == 21] = 1
x[x == 31] = 2
x[x == 41] = 3
x[x == 51] = 4
x[x == -999] = 5

print(x[x == 0])

print(np.unique(x))


