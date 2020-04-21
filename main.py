import torch
import torch.nn as nn

import os

from data import SAR

root_dir = '/data/Databases/MDA'

maps = ['Vancouver']

train_dataset = SAR(root_dir=os.path.join(root_dir, maps[0]), kernel=(224, 224), stride=(32, 32))

for s, l in train_dataset:
    print(s.size())
    print(l.size())
