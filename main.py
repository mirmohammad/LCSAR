import argparse
# import logging
import os

import torch
import torch.nn as nn
from torch.utils import data
from tqdm import tqdm

from data import SAR
from segment import DeconvNet

parser = argparse.ArgumentParser(description='DeepSAR | Land Classification for SAR imagery using Deep Learning')
parser.add_argument('dir', help='path to directory containing SAR raster directories')
# parser.add_argument('--log_dir', required=True, help='path to directory to store the results')
# parser.add_argument('--load_dir', default='', help='path to pre-trained model parameters')
parser.add_argument('--batch_size', default=16, type=int, help='batch size')
parser.add_argument('--num_workers', default=4, type=int, help='number of dataloader workers')
parser.add_argument('--num_epochs', default=100, type=int, help='number of epochs')
parser.add_argument('--learning_rate', default=1e-3, type=float, help='initial learning rate')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='l2 regularization factor')
parser.add_argument('--scheduler_step_size', default=25, type=int, help='scheduler step size')
parser.add_argument('--scheduler_gamma', default=0.1, type=float, help='scheduler gamma')
parser.add_argument('--num_classes', default=6, type=int, help='output dimension')
parser.add_argument('--resize', default=224, type=int, help='resize images in pixels')
# parser.add_argument('--xscale', default=4, type=int, help='scaling for temporal dim')
# parser.add_argument('--yscale', default=4, type=int, help='scaling for spatial dim')
# parser.add_argument('--random_seed', default=42, type=int, help='fix the random seed for reproducibility')
# parser.add_argument('--normalize_labels', action='store_true', help='zero-one normalization applied to labels')
# parser.add_argument('--pretrained', action='store_true', help='load pre-trained model from load_dir')
args = parser.parse_args()

cuda = torch.cuda.is_available()

root_dir = args.dir
# log_dir = args.log_dir
# load_dir = args.load_dir
batch_size = args.batch_size
num_workers = args.num_workers
num_epochs = args.num_epochs
learning_rate = args.learning_rate
weight_decay = args.weight_decay
scheduler_step_size = args.scheduler_step_size
scheduler_gamma = args.scheduler_gamma
num_classes = args.num_classes
resize = args.resize
# xscale = args.xscale
# yscale = args.yscale
# random_seed = args.random_seed

# torch.manual_seed(random_seed)
# np.random.seed(random_seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

device = torch.device('cuda:0' if cuda else 'cpu')
tqdm.write('CUDA is not available!' if not cuda else 'CUDA is available!')
tqdm.write('')

maps = ['Montreal', 'Ottawa', 'Quebec', 'Saskatoon', 'Vancouver']

train_maps = [maps[0]]
valid_maps = [maps[4]]

train_root = os.path.join(root_dir, train_maps[0])
valid_root = os.path.join(root_dir, valid_maps[0])

train_dataset = SAR(root_dir=train_root, kernel=(224, 224), stride=(192, 192))
valid_dataset = SAR(root_dir=valid_root, kernel=(224, 224), stride=(192, 192))

train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
valid_loader = data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

model = DeconvNet(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)


def iterate(ep, mode):
    if mode == 'train':
        model.train()
        loader = train_loader
    else:
        model.eval()
        loader = valid_loader
    num_samples = 0
    run_loss = 0.
    run_acc = 0.

    monitor = tqdm(loader, desc=mode)
    for sar, lbl in monitor:
        sar, lbl = sar.to(device), lbl.to(device).squeeze()

        outputs = model(sar)

        _, seg = torch.max(outputs.data, 1)
        loss = criterion(outputs, lbl.long())

        num_samples += lbl.size(0)
        run_loss += loss.item() * lbl.size(0)
        run_acc += ((seg == lbl).sum().item() / (lbl.size(1) * lbl.size(2)))

        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        monitor.set_postfix(ep=ep, loss=run_loss / num_samples, acc=run_acc / num_samples)

    scheduler.step()

    return run_acc / num_samples


if __name__ == "__main__":
    best_acc = 0.
    best_ep = -1
    for epoch in range(num_epochs):
        accuracy = iterate(epoch, 'train')
        tqdm.write(f'Train | Epoch {epoch} | Accuracy {accuracy}')
        with torch.no_grad():
            accuracy = iterate(epoch, 'valid')
            if accuracy >= best_acc:
                best_acc = accuracy
                best_ep = epoch
            tqdm.write(f'Valid | Epoch {epoch} | Accuracy {accuracy} | Best Accuracy {best_acc} | Best Epoch {best_ep}')
