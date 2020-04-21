import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

from PIL import Image

Image.MAX_IMAGE_PIXELS = None

img = Image.open('/home/mir/Downloads/MF4AR_Training.tif')

x = np.array(img)

print(np.unique(x))


