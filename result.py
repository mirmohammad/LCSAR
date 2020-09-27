import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

import torch
import torch.nn as nn
from torch.utils import data
from tqdm import tqdm

from data import MSAR
from segment import SegNet

cuda = torch.cuda.is_available()

root_dir = '/data/Databases/MDA'
batch_size = 32
num_workers = 2
num_classes = 4
kernel = 224
stride = 192

device = torch.device('cuda:0' if cuda else 'cpu')
tqdm.write('CUDA is not available!' if not cuda else 'CUDA is available!')
tqdm.write('')

maps = ['Montreal', 'Ottawa', 'Quebec', 'Saskatoon', 'Toronto', 'Vancouver']

maps = {'train': [], 'valid': [maps[i] for i in [4]]}

valid_dataset = MSAR(root_dir=root_dir, maps=maps, train=False,
                     kernel=(kernel, kernel),
                     stride=(stride, stride),
                     min_classes=1,
                     max_count=0.8)

valid_loader = data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

criterion = nn.CrossEntropyLoss()

model = SegNet(num_classes=num_classes)
model.load_state_dict(torch.load('params_0.8724430235924614_{0: 0.9059534054566584, 1: 0.9150954945563414, 2: 0.7869461704979367, 3: 0.8780908923917099}_.pt'))
model = model.to(device)


def plot_matrix(rm, title='Robot World'):
    cmap = colors.ListedColormap(['g', 'b', 'r', 'y'])
    plt.imshow(rm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def iterate():
    model.eval()
    run_acc = 0.
    run_class_samples = {0: 0., 1: 0., 2: 0., 3: 0.}
    run_class_correct = {0: 0., 1: 0., 2: 0., 3: 0.}
    run_class_acc = {0: 0., 1: 0., 2: 0., 3: 0.}

    monitor = tqdm(valid_loader, desc='extract')
    for sar, lbl in monitor:
        sar, lbl = sar.to(device), lbl.to(device).squeeze()

        outputs = model(sar)
        _, seg = torch.max(outputs.data, 1)

        # for i in range(32):
        #     plot_matrix(seg[i].detach().cpu().numpy())
        #     plot_matrix(lbl[i].detach().cpu().numpy())
        #
        # exit(0)

        # mapping = (seg == lbl)

        for i in range(32):
            mapping = (seg[i] == lbl[i])
            if (mapping.sum().item() / (lbl.size(1) * lbl.size(2))) > 0.8:
                plot_matrix(seg[i].detach().cpu().numpy(), title='Model Output')
                plot_matrix(lbl[i].detach().cpu().numpy(), title='Ground Truth')


if __name__ == "__main__":
    with torch.no_grad():
        iterate()
