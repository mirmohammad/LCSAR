import argparse
import logging
import os

import torch
import torch.nn as nn
from torch.utils import data
from tqdm import tqdm

from data import SAR, MSAR

parser = argparse.ArgumentParser(description='DeepSAR | Land Classification for SAR imagery using Deep Learning')
parser.add_argument('dir', help='path to directory containing SAR raster directories')
parser.add_argument('-o', '--log_dir', required=True, help='path to directory to store the results')
# parser.add_argument('--load_dir', default='', help='path to pre-trained model parameters')
parser.add_argument('-b', '--batch_size', default=32, type=int, help='batch size')
parser.add_argument('--num_workers', default=4, type=int, help='number of dataloader workers')
parser.add_argument('--num_epochs', default=100, type=int, help='number of epochs')
parser.add_argument('--learning_rate', default=1e-3, type=float, help='initial learning rate')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='l2 regularization factor')
parser.add_argument('--scheduler_step_size', default=25, type=int, help='scheduler step size')
parser.add_argument('--scheduler_gamma', default=0.1, type=float, help='scheduler gamma')
parser.add_argument('--num_classes', default=4, type=int, help='output dimension')
parser.add_argument('-t', '--train_maps', nargs='+', type=int, help='indices of train maps')
parser.add_argument('-v', '--valid_maps', nargs='+', type=int, help='indices of valid maps')
parser.add_argument('-k', '--kernel', default=225, type=int, help='sampling kernel size')
parser.add_argument('-s', '--stride', default=196, type=int, help='sampling stride size')
parser.add_argument('--random_seed', default=42, type=int, help='fix the random seed for reproducibility')
# parser.add_argument('--normalize_labels', action='store_true', help='zero-one normalization applied to labels')
# parser.add_argument('--pretrained', action='store_true', help='load pre-trained model from load_dir')
args = parser.parse_args()

cuda = torch.cuda.is_available()

root_dir = args.dir
log_dir = args.log_dir
# load_dir = args.load_dir
batch_size = args.batch_size
num_workers = args.num_workers
num_epochs = args.num_epochs
learning_rate = args.learning_rate
weight_decay = args.weight_decay
scheduler_step_size = args.scheduler_step_size
scheduler_gamma = args.scheduler_gamma
num_classes = args.num_classes
train_maps = args.train_maps
valid_maps = args.valid_maps
kernel = args.kernel
stride = args.stride
random_seed = args.random_seed

# torch.manual_seed(random_seed)
# np.random.seed(random_seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

device = torch.device('cuda:0' if cuda else 'cpu')
tqdm.write('CUDA is not available!' if not cuda else 'CUDA is available!')
tqdm.write('')

maps = ['Montreal', 'Ottawa', 'Quebec', 'Saskatoon', 'Toronto', 'Vancouver']

assert all(x < len(maps) for x in train_maps), f'Train map index must be between 0 and {len(maps)}'
assert all(x < len(maps) for x in valid_maps), f'Valid map index must be between 0 and {len(maps)}'

# ### LOGGING ###
log_pad = 96
log_file = f'{train_maps}_{valid_maps}.txt'
logging.basicConfig(filename=os.path.join(log_dir, log_file),
                    filemode='w',
                    format='%(asctime)s, %(name)s - %(message)s',
                    datefmt='%D - %H:%M:%S',
                    level=logging.INFO)
# ### LOGGING ###

# train_maps = [maps[train_maps]]
# valid_maps = [maps[valid_maps]]

# train_root = os.path.join(root_dir, train_maps[0])
# valid_root = os.path.join(root_dir, valid_maps[0])

maps = {'train': [maps[i] for i in train_maps], 'valid': [maps[i] for i in valid_maps]}

print(f'*** Setting up TRAIN dataset using {train_maps} raster ***')
train_dataset = MSAR(root_dir=root_dir, maps=maps, train=True,
                     kernel=(kernel, kernel),
                     stride=(stride, stride),
                     min_classes=1,
                     max_count=0.8)
print('**************************************************************')
print(f'*** Setting up VALID dataset using {valid_maps} raster ***')
valid_dataset = MSAR(root_dir=root_dir, maps=maps, train=False,
                     kernel=(kernel, kernel),
                     stride=(stride, stride),
                     min_classes=1,
                     max_count=0.8)
print('**************************************************************')

# train_dataset = SAR(root_dir=train_root, kernel=(224, 224), stride=(192, 192), min_classes=1, max_count=0.8)
# valid_dataset = SAR(root_dir=valid_root, kernel=(224, 224), stride=(192, 192), min_classes=2, max_count=0.8)

train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
valid_loader = data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

criterion = nn.CrossEntropyLoss()

from segment import DeconvNet
from segment import SegNet
from segment import PSPNet

#model = SegNet(num_classes=num_classes).to(device)
model = PSPNet(50, (1, 2, 3, 6), 0.1, num_classes, 8, True, criterion, pretrained=False).to(device)
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
    run_class_samples = {0: 0., 1: 0., 2: 0., 3: 0.}
    run_class_correct = {0: 0., 1: 0., 2: 0., 3: 0.}
    run_class_acc = {0: 0., 1: 0., 2: 0., 3: 0.}

    monitor = tqdm(loader, desc=mode)
    for sar, lbl in monitor:
        sar, lbl = sar.to(device), lbl.to(device).squeeze()

        if mode == 'train':
            outputs, main_loss, aux_loss = model(sar, lbl)
            loss = main_loss + 0.4 * aux_loss  # criterion(outputs, lbl.long())
        else:
            outputs, loss = model(sar, lbl)

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

    scheduler.step()

    # ### LOGGING ###
    logging.info(f'{mode.upper():5} | loss {(run_loss / num_samples):7.5f}, acc {(run_acc / num_samples):7.5f}, '
                 f'c21 {(run_class_acc[0]):7.5f}, c31 {(run_class_acc[1]):7.5f}, '
                 f'c41 {(run_class_acc[2]):7.5f}, c51 {(run_class_acc[3]):7.5f}')
    # ### LOGGING ###

    return run_acc / num_samples


if __name__ == "__main__":
    best_acc = 0.
    best_ep = -1
    for epoch in range(num_epochs):
        # ### LOGGING ###
        logging.info(f'Epoch {epoch:03} '.ljust(log_pad, '-'))
        # ### LOGGING ###

        accuracy = iterate(epoch, 'train')
        tqdm.write(f'Train | Epoch {epoch} | Accuracy {accuracy}')
        with torch.no_grad():
            accuracy = iterate(epoch, 'valid')
            if accuracy >= best_acc:
                # ### LOGGING ###
                logging.info(f'New best valid acc at {accuracy:7.5f}')
                # logging.info('Saving model parameters ...')
                # ### LOGGING ###

                best_acc = accuracy
                best_ep = epoch
            tqdm.write(f'Valid | Epoch {epoch} | Accuracy {accuracy} | Best Accuracy {best_acc} | Best Epoch {best_ep}')

    # ### LOGGING ###
    logging.info(f'\nTrain {train_maps} | Valid {valid_maps}')
    logging.info(f'Best valid acc {best_acc:7.5f} at epoch {best_ep}')
    # logging.info('Saving model parameters ...')
    # ### LOGGING ###
