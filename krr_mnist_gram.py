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

parser = argparse.ArgumentParser()

parser.add_argument('--filtersize', type=int, required=True)
parser.add_argument('--stride', type=int, required=True)

args = parser.parse_args()

dataset = torch.load('/home/cagnetta/izar/krr_mnist_code/nfMNIST.pt', map_location=device)

x_train = torch.cat([dataset['train'][i][:2048] for i in range(10)], 0).double()
y_train = torch.ones( 2048*10, device=device)
for i in range(10):
    y_train[(2048 * i):(2048 * (i+1))] = float(i)

x_test = torch.cat([dataset['test'][i][:512] for i in range(10)], 0).double()
y_test = torch.ones( 512*10, device=device)
for i in range(10):
    y_test[(512 * i):(512 * (i+1))] = float(i)

gram_trtr = gram_llap_2d(x_train, x_train, args.filtersize, args.stride, sigma=4)
gram_tetr = gram_llap_2d(x_test, x_train, args.filtersize, args.stride, sigma=4)


filename = 'gramMNIST'
filename += '_s' + str(args.filtersize)
filename += '_str' + str(args.stride)

torch.save({
            'args': args,
            'gram_trtr': gram_trtr,
            'gram_tetr': gram_tetr,
            'y_train': y_train,
            'y_test': y_test,
           }, filename + '.pt')
