import os

import torch
import numpy as np

from sklearn.model_selection import train_test_split as train_val

def load_dataloader_new(num_samples, max_D, label_size, batch_size, ratio, sig, output_dir_dataset, floor_amp, min_sep):

    clean_signals, label = gen_signal(num_samples, max_D, label_size, sig, floor_amp, min_sep)
    
    train_input, val_input, train_label, val_label = train_val(clean_signals, label, test_size=ratio, random_state=42)

    train_input = torch.from_numpy(train_input).float()
    val_input = torch.from_numpy(val_input).float()
    train_label = torch.from_numpy(train_label).float()
    val_label = torch.from_numpy(val_label).float()

    np.save(os.path.join(output_dir_dataset, "train_input"), train_input)
    np.save(os.path.join(output_dir_dataset, "val_input"), val_input)
    np.save(os.path.join(output_dir_dataset, "train_label"), train_label)
    np.save(os.path.join(output_dir_dataset, "val_label"), val_label)

    train_dataset = torch.utils.data.TensorDataset(train_input, train_label)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = torch.utils.data.TensorDataset(val_input, val_label)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader


def load_dataloader_exist(batch_size):

    train_input = np.load("./Dataset/train_input.npy")
    train_label = np.load("./Dataset/train_label.npy")
    val_input = np.load("./Dataset/val_input.npy")
    val_label = np.load("./Dataset/val_label.npy")

    train_input = torch.from_numpy(train_input).float()
    val_input = torch.from_numpy(val_input).float()
    train_label = torch.from_numpy(train_label).float()
    val_label = torch.from_numpy(val_label).float()

    train_dataset = torch.utils.data.TensorDataset(train_input, train_label)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = torch.utils.data.TensorDataset(val_input, val_label)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader


# to generate gaussian distribution
def Gaussian_distribution(max_D, avg, num, sig):
    # print("sig = ", sig)
    xgrid = np.linspace(0, max_D, num)
    sqrt_2pi=np.power(2*np.pi,0.5)
    coef=1/(sqrt_2pi*sig)
    powercoef=-1/(2*np.power(sig,2))
    mypow=powercoef*(np.power((xgrid-avg),2))
    result = coef*(np.exp(mypow))
    return result/np.max(result)


def gen_signal(num_samples, max_D, label_size, sig, floor_amp, min_sep, signal_dim=32):
    s = np.zeros([num_samples, signal_dim])
    b = np.arange(signal_dim)*0.01
    label = np.zeros([num_samples, label_size])

    for i in np.arange(num_samples):

        if i < 0:
            D = np.random.random() * max_D
        else:
            while True:
                D = np.random.random(2) * max_D
                D = np.sort(D)
                Dr = np.roll(D, 1)
                if np.min(np.abs(D-Dr)) > min_sep:
                    break
        
        amp = np.abs(np.random.randn(len(D))) + floor_amp
        amp = amp/np.sum(amp)  # amp normalization

        for j in np.arange(len(D)):
            exp_D = amp[j] * np.exp(-D[j]*b)
            s[i] += exp_D
            
            label[i] += amp[j] * Gaussian_distribution(max_D, D[j], label_size, sig=sig)

        s[i] = s[i]/s[i, 0]  # s normalizaton
            
    return s, label