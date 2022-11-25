import numpy as np
import torch
import wandb
from module import set_module


def up_img(epoch=None):

    train_input = np.load("./Dataset/train_input.npy")
    train_label = np.load("./Dataset/train_label.npy")
    val_input = np.load("./Dataset/val_input.npy")
    val_label = np.load("./Dataset/val_label.npy")


    with torch.no_grad():
        if epoch == None:
            checkpoint = torch.load('checkpoint/epoch_best.pth', map_location=torch.device('cuda'))
        else:
            checkpoint = torch.load('checkpoint/epoch_{num}.pth'.format(num=epoch))
        args = checkpoint['args']
        module = set_module(args)
        module.load_state_dict(checkpoint['model'])
        module.cpu()
        module.eval()

    sample = 7
    test_input = np.reshape(train_input[sample], (1, 32))
    test_input = torch.tensor(test_input)
    test_input = test_input.to(torch.float32)

    test_out = module(test_input)
    test_out = test_out.cpu().detach().numpy()

    data = [[x, y] for (x, y) in zip(np.arange(len(train_label[sample])), train_label[sample])]
    table = wandb.Table(data=data, columns=["train_label_D", "train_label_amp"])
    wandb.log({"train_label_1" : wandb.plot.line(table, "train_label_D", "train_label_amp", title="Train_label")})

    if epoch == None:
        data = [[x, y] for (x, y) in zip(np.arange(len(test_out[0, :])), test_out[0, :])]
        table = wandb.Table(data=data, columns=["test_out_D", "test_out_amp"])
        wandb.log({"epoch_best" : wandb.plot.line(table, "test_out_D", "test_out_amp", title="epoch_best")})
    else:
        data = [[x, y] for (x, y) in zip(np.arange(len(test_out[0, :])), test_out[0, :])]
        table = wandb.Table(data=data, columns=["test_out_D", "test_out_amp"])
        wandb.log({'epoch_{num}'.format(num=epoch) : wandb.plot.line(table, "test_out_D", "test_out_amp", 
                                                                    title='epoch_{num}'.format(num=epoch))})
