import sys
import copy
import time
import argparse

import torch
import torch.nn.functional as F
import math

from kernels import gram_llap_2d, gram_clap_2d
from myutils import kernel_regression

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float64)

def kernel_regression(K_trtr, K_tetr, y_tr, y_te):
    alpha = torch.linalg.inv(K_trtr) @ y_tr
    f = K_tetr @ alpha
    mse = (f - y_te).pow(2).mean()
    return mse

parser = argparse.ArgumentParser()

parser.add_argument('--filtersize', type=int, required=True)
parser.add_argument('--stride', type=int, required=True)

args = parser.parse_args()

filename = 'gramMNIST'
filename += '_s' + str(args.filtersize)
filename += '_str' + str(args.stride)
gram = torch.load('/home/cagnetta/krr_mnist_gram/' + filename + '.pt', map_location=device)

gram_trtr = gram['gram_trtr']
gram_tetr = gram['gram_tetr']
y_train = gram['y_train']
y_test = gram['y_test']

shuffled = torch.randperm(16384)
plist = [128, 256, 512, 1024, 2048, 4096, 8192]

error = []
std = []

for p in plist:

    nexp = 16384 // p
    print(p, nexp, flush=True)

    mse = []
    for exp in range(nexp):

        indices = shuffled[exp*p:(exp+1)*p]
        K_trtr = gram_trtr[indices,:]
        K_trtr = K_trtr[:,indices]
        y_tr = y_train[indices]
        K_tetr = gram_tetr[:,indices]

        mse.append(kernel_regression(K_trtr, K_tetr, y_tr, y_test).item())

    print(mse, flush=True)
    
    error.append(torch.tensor(mse).mean().item())
    std.append(torch.tensor(mse).std().item())

torch.save({
        'plist': plist,
        'mse': error,
        'std': std
        }, f'mnist_mse_llap_s{args.filtersize}_str{args.stride}.pt')
