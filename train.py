import os
import time
import logging
import argparse
import sys

import numpy as np
import torch
import wandb

import utils
import module
import dataset

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
logger = logging.getLogger(__name__)

def make_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--module_type', type=str, default='skip')

    # signal parameters
    parser.add_argument('--max_D', type=float, default=14)
    parser.add_argument('--num_samples', type=int, default=30000)
    parser.add_argument('--label_size', type=int, default=140, help='size of the 1-D pesudo-spectrum')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--ratio', type=float, default=0.1, help='ratio of val_set to train_set ')
    parser.add_argument('--sig', type=float, default=0.1, help='width of gaussian distribution')
    parser.add_argument('--output_dir_dataset', type=str, default='./Dataset')
    parser.add_argument('--floor_amp', type=float, default=0.3)
    parser.add_argument('--min_sep', type=float, default=0.5)
    parser.add_argument('--signal_dim', type=int, default=32, help='dimension of the input signal')

    # fr module parameters
    parser.add_argument('--fr_n_layers', type=int, default=24, help='number of convolutional layers in the fr module')
    parser.add_argument('--fr_n_filters', type=int, default=128, help='number of filters per layer in the fr module')
    parser.add_argument('--fr_kernel_size', type=int, default=3,
                        help='filter size in the convolutional blocks of the fr module')
    parser.add_argument('--fr_kernel_out', type=int, default=35, help='size of the conv transpose kernel')
    parser.add_argument('--fr_inner_dim', type=int, default=70, help='dimension after first linear transformation')

    # training parameters
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--criterion', type=str, default='mseloss')
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=0.003, help='learning rate')

    parser.add_argument('--save_epoch_freq', type=int, default=10,
                        help='frequency of saving checkpoints at the end of epochs')

    parser.add_argument('--output_dir', type=str, default='./checkpoint')
    return parser


def train_epoch(args, DILT_module, optimizer, scheduler, criterion, train_loader, val_loader, epoch):
    epoch_start_time = time.time()
    DILT_module.train()
    loss = 0

    for __, (train_input, train_label) in enumerate(train_loader):
        train_input, train_label = train_input.cuda(), train_label.cuda()
        optimizer.zero_grad()  #TODO
        output = DILT_module(train_input)
        # loss_epoch = criterion(output, train_label)
        loss_epoch = torch.pow(((output) - (train_label)),2)
        loss_epoch = torch.sum(loss_epoch).to(torch.float32)
        loss_epoch.backward()
        optimizer.step()
        loss += loss_epoch.data.item()
    
    loss /= (args.num_samples * (1 - args.ratio))
    scheduler.step(loss) # TODO
    
    logger.info("Epochs: %d / %d, Time: %.1f, training loss: %.3f", epoch, args.epochs, time.time() - epoch_start_time, loss)
    
    return loss


def get_criterion(type_c):

    if type_c == 'mseloss':
        return torch.nn.MSELoss(reduction='sum')
    elif type_c == 'huberloss':
        return torch.nn.HuberLoss()

def get_optimizer(type_o):

    if type_o == 'adam':
        return torch.optim.Adam(DILT_module.parameters(), lr=config['lr'])
    elif type_o == 'adadelta':
        return torch.optim.Adadelta(DILT_module.parameters(), lr=config['lr'])
    elif type_o == 'adagrad':
        return torch.optim.Adagrad(DILT_module.parameters(), lr=config['lr'])
    elif type_o == 'adamw':
        return torch.optim.AdamW(DILT_module.parameters(), lr=config['lr'])
    elif type_o == 'rmsprop':
        return torch.optim.RMSprop(DILT_module.parameters(), lr=config['lr'], alpha=0.9)

if __name__ == '__main__':
    args = make_parser().parse_args()
    wandb.init(config=args)
    config = wandb.config
    if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
    
    file_handler = logging.FileHandler(filename=os.path.join(args.output_dir, 'run.log'))
    stdout_handler = logging.StreamHandler(sys.stdout)
    handlers = [file_handler, stdout_handler]
    logging.basicConfig(
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,
        format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
        handlers=handlers
    )

    train_loader, val_loader = dataset.load_dataloader_new(config['num_samples'], config['max_D'], config['label_size'], config['batch_size'], 
                                            config['ratio'], config['sig'], config['output_dir_dataset'], config['floor_amp'], config['min_sep'])
    # train_loader, val_loader = dataset.load_dataloader_exist(config['batch_size'])

    DILT_module = module.set_module(config)
    
    optimizer = get_optimizer(config['optimizer'])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=7, factor=0.5, verbose=True)  # TODO
    criterion = get_criterion(config['criterion'])

    min_loss = np.inf
    for epoch in range(1, config["epochs"] + 1):

        loss = train_epoch(args, DILT_module, optimizer, scheduler, criterion, train_loader, val_loader, epoch)
        metrics = {
                "train_loss": loss
        }
        wandb.log(metrics)

        if epoch % args.save_epoch_freq==0:
            checkpoint = {
                    'model': DILT_module.state_dict(),
                    'args': args
            }
            cp_path = os.path.join(args.output_dir, 'epoch_'+str(epoch)+'.pth')

            torch.save(checkpoint, cp_path)
            utils.up_img(epoch)

        elif min_loss > loss:
            min_loss = loss
            checkpoint = {
                    'model': DILT_module.state_dict(),
                    'args': args
            }
            cp_path = os.path.join(args.output_dir, 'epoch_best.pth')

            torch.save(checkpoint, cp_path)

    utils.up_img()
