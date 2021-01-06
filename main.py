import argparse
import logging
import os
import random

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils import data
from torchvision import transforms
from tqdm import tqdm

from data import LCSAR
from segment import SegNet

# CUDA
cuda = torch.cuda.is_available()

# ARG
parser = argparse.ArgumentParser(description='DeepSAR | Land Classification for SAR imagery using Deep Learning')
parser.add_argument('dir', help='path to yaml configuration file')
parser.add_argument('--gpu', default='0', type=str)
parser.add_argument('--log', action='store_true', help='log the loss and accuracy per epoch')
parser.add_argument('--save', action='store_true', help='save best model parameters')
args = parser.parse_args()

# YAML
cfg = yaml.safe_load(open(args.dir))

# default
num_classes = cfg['num_classes']
num_workers = cfg['num_workers']
# data & log directories
data_dir = cfg['data_dir']
log_dir = cfg['log_dir']
# training
batch_size = cfg['batch_size']
num_epochs = cfg['num_epochs']
# optimizer
learning_rate = cfg['learning_rate']
momentum = cfg['momentum']
weight_decay = cfg['weight_decay']
# lr scheduler
step = cfg['step']
gamma = cfg['gamma']
# random initialization
random_seed = cfg['random_seed']
manual_seed = cfg['manual_seed']
# train & valid maps
train_maps = cfg['train_maps']
valid_maps = cfg['valid_maps']
# sampling
kernel = cfg['kernel']
stride = cfg['stride']
# data augmentation
random_vertical_flip = cfg['random_vertical_flip']
random_horizontal_flip = cfg['random_horizontal_flip']

if manual_seed:
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

device = torch.device('cuda:' + args.gpu if cuda else 'cpu')
tqdm.write('CUDA is not available!' if not cuda else 'CUDA is available!')
tqdm.write('')

log_pad = 96
if args.log:
    log_file = f'All_Rotate_Flip_T{train_maps}_V{valid_maps}_K{kernel}_S{stride}_B{batch_size}.txt'
    logging.basicConfig(filename=os.path.join(log_dir, log_file),
                        filemode='w',
                        format='%(asctime)s, %(name)s - %(message)s',
                        datefmt='%D - %H:%M:%S',
                        level=logging.INFO)

print(f'*** Setting up TRAIN dataset using {train_maps} raster ***')
train_dataset = LCSAR(root_dir=data_dir, train=True, maps=train_maps, kernel=(kernel, kernel), stride=(stride, stride), transform=transforms.ToTensor())

print(f'*** Setting up VALID dataset using {valid_maps} raster ***')
valid_dataset = LCSAR(root_dir=data_dir, train=False, maps=valid_maps, kernel=(224, 224), stride=(192, 192), transform=transforms.ToTensor())

train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
valid_loader = data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

criterion = nn.CrossEntropyLoss()

model = SegNet(num_classes=num_classes).to(device)
# model = PSPNet(50, (1, 2, 3, 6), 0.1, num_classes, 8, True, criterion, pretrained=False).to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step, gamma=gamma)


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
    run_class_samples = {0: 0., 1: 0., 2: 0., 3: 0.}
    run_class_correct = {0: 0., 1: 0., 2: 0., 3: 0.}
    run_class_acc = {0: 0., 1: 0., 2: 0., 3: 0.}

    monitor = tqdm(loader, desc=mode)
    for sar, lbl in monitor:
        sar, lbl = sar.to(device), lbl.to(device).squeeze()

        # if mode == 'train':
        #     outputs, main_loss, aux_loss = model(sar, lbl)
        #     loss = main_loss + 0.4 * aux_loss  # criterion(outputs, lbl.long())
        # else:
        #     outputs, loss = model(sar, lbl)

        outputs = model(sar)
        loss = criterion(outputs, lbl.long())
        _, seg = torch.max(outputs.data, 1)

        num_samples += lbl.size(0)
        run_loss += loss.item() * lbl.size(0)

        # import ipdb; ipdb.set_trace()

        # TODO: detach and cpu to free cuda memory if needed
        mapping = (seg == lbl)
        run_acc += (mapping.sum().item() / (lbl.size(1) * lbl.size(2)))

        for lbl_class in torch.unique(lbl).detach().cpu().tolist():
            mapping_alias = (lbl == lbl_class)
            run_class_samples[lbl_class] += mapping_alias.sum().item()
            run_class_correct[lbl_class] += (mapping & mapping_alias).sum().item()
            run_class_acc[lbl_class] = run_class_correct[lbl_class] / run_class_samples[lbl_class]

        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        monitor.set_postfix(ep=ep, loss=run_loss / num_samples, acc=run_acc / num_samples,
                            c21=run_class_acc[0], c31=run_class_acc[1], c41=run_class_acc[2], c51=run_class_acc[3])

    if mode == 'train':
        scheduler.step()

    if args.log:
        logging.info(f'{mode.upper():5} | loss {(run_loss / num_samples):7.5f}, acc {(run_acc / num_samples):7.5f}, '
                     f'c21 {(run_class_acc[0]):7.5f}, c31 {(run_class_acc[1]):7.5f}, '
                     f'c41 {(run_class_acc[2]):7.5f}, c51 {(run_class_acc[3]):7.5f}')

    return run_acc / num_samples, run_class_acc


if __name__ == "__main__":
    best_acc = 0.
    best_cls = {0: 0., 1: 0., 2: 0., 3: 0.}
    best_ep = -1
    for epoch in range(num_epochs):

        if args.log:
            logging.info(f'Epoch {epoch:03} '.ljust(log_pad, '-'))

        accuracy, _ = iterate(epoch, 'train')
        tqdm.write(f'Train | Epoch {epoch} | Accuracy {accuracy}')
        with torch.no_grad():
            accuracy, accuracies = iterate(epoch, 'valid')
            if accuracy >= best_acc:
                if args.log:
                    logging.info(f'New best valid acc at {accuracy:7.5f}')

                best_acc = accuracy
                best_cls = accuracies
                best_ep = epoch
                if args.save:
                    torch.save(model.state_dict(), 'params_' + str(best_acc) + '_' + str(best_cls) + '_.pt')
            tqdm.write(f'Valid | Epoch {epoch} | Accuracy {accuracy} | Best Accuracy {best_acc} | Best Epoch {best_ep}')

    if args.log:
        logging.info(f'\nTrain {train_maps} | Valid {valid_maps}')
        logging.info(f'Best valid acc {best_acc:7.5f} at epoch {best_ep}')
        logging.info(f'c21 {(best_cls[0]):7.5f}, c31 {(best_cls[1]):7.5f}, c41 {(best_cls[2]):7.5f}, c51 {(best_cls[3]):7.5f}')

    # f = open('results_pair/preliminary_results.txt', 'a')
    # f.write(f'Train {train_maps} | Valid {valid_maps}\n')
    # f.write(f'Best valid acc {best_acc:7.5f} at epoch {best_ep}\n')
    # f.write(f'c21 {(best_cls[0]):7.5f}, c31 {(best_cls[1]):7.5f}, c41 {(best_cls[2]):7.5f}, c51 {(best_cls[3]):7.5f}\n\n')
    # f.close()
