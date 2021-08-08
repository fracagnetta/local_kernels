import sys
import copy
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from kernels import compute_kernels, gram_lap_1d, gram_llap_1d, gram_clap_1d
from myutils import hypersphere_random_sampler, hypercube_random_sampler, grf_generator, kernel_regression

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float64)

# Parser

parser = argparse.ArgumentParser(description='kernel regression with empirical NTKs')
'''
	TEACHER ARGS
'''
parser.add_argument('--trainsize', metavar='P', type=int, help='size of the training set')
parser.add_argument('--testsize', metavar='TEST', type=int, help='size of the training set')
parser.add_argument('--distro', type=str, help='cube, sphere')
parser.add_argument('--imagesize', metavar='IM.SIZE', type=int, help='size of the images')
parser.add_argument('--patternsize', metavar='PAT.SIZE', type=int, help='size of the local patterns')
parser.add_argument('--teacher', type=str, help='teacher: loc, conv')
'''
	STUDENT ARGS
'''
parser.add_argument('--filtersize', metavar='FILT.SIZE', type=int, help='size of the student filters')
parser.add_argument('--hidden', metavar='H', type=int, default=8192)
parser.add_argument('--bias', action='store_true', default=False)
parser.add_argument('--ridge', type=float, default=None)
parser.add_argument('--avg', type=int, default=None)
'''
	OUTPUT ARGS
'''
parser.add_argument('--array', type=int, help='index for array runs', default=None)
args = parser.parse_args()

trainsize = args.trainsize
imagesize = args.imagesize
patternsize = args.patternsize
testsize = args.testsize
filtersize = args.filtersize
hidden = args.hidden


'''
	TEACHER: GENERATE GAUSSIAN FIELD WITH CONVOLUTIONAL COV.
'''
if args.distro == 'cube':
    x = hypercube_random_sampler(trainsize + testsize, imagesize, device)
elif args.distro == 'sphere':
    x = hypersphere_random_sampler(trainsize + testsize, imagesize, device)
else:
    raise AssertionError ('distribution not implemented')
if args.teacher == 'loc':
    teacher_cov = gram_llap_1d(x.reshape(trainsize + testsize, 1, imagesize), x.reshape(trainsize + testsize, 1, imagesize), patternsize, sigma=args.patternsize, pbc=True)
if args.teacher == 'conv':
    teacher_cov = gram_clap_1d(x.reshape(trainsize + testsize, 1, imagesize), x.reshape(trainsize + testsize, 1, imagesize), patternsize, sigma=args.patternsize, pbc=True)
else:
    raise AssertionError ('teacher not implemented')
y = grf_generator( teacher_cov, device)

x_train = x[:trainsize].unsqueeze(1)
y_train = y[:trainsize]

x_test = x[trainsize:].unsqueeze(1)
y_test = y[trainsize:]

'''
	STUDENT: NTK OF A MINIMAL CNN.
'''
class Conv1d_PBC_N(nn.Module):

    def __init__(self, in_channels, out_channels, filter_size, bias=False):

        super(Conv1d_PBC_N, self).__init__()

        # initialise the filter with unit-variance Gaussian RV
        self.fs = filter_size
        self.filt = nn.Parameter( torch.randn( out_channels, in_channels, filter_size))
        if bias:
            self.bias = nn.Parameter( torch.randn( out_channels))
        else:
            self.register_parameter('bias', None)

    # return convolution of the input x with PBC with the filter
    def forward(self, x):
        x_pbc = F.pad(x, (0, self.fs-1), mode='circular')
        return F.conv1d(x_pbc, self.filt, self.bias)

class AvgPool1d(nn.Module):

    def __init__(self):

        super(AvgPool1d, self).__init__()

    def forward(self, x):
        return F.avg_pool1d(x, x.size(2))

class Linear_N(nn.Module):

    def __init__(self, in_features, out_features, bias=False):

        super(Linear_N, self).__init__()

        self.h = in_features
        self.weight = nn.Parameter( torch.randn( out_features, in_features))
        if bias:
            self.bias = nn.Parameter( torch.randn( out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

class miniCNN_1d(nn.Module):
    
    def __init__(self, hidden, filtersize, bias=False):
        super().__init__()
        
        self.h = hidden
        self.s = filtersize
        self.bias = bias

        self.cnn1 = Conv1d_PBC_N(1, self.h, filter_size=self.s, bias=self.bias)
        self.pool1 = AvgPool1d()
        self.fc1 = Linear_N(self.h, 1, bias=False)

    # Forward pass    
    def forward(self, x):
        pre_act = self.cnn1(x/( float(self.s) ** 0.5))
        post_act = F.relu(pre_act)
        pooled = self.pool1(post_act)
        pooled = pooled.view(pooled.size(0), -1)
        out = self.fc1(pooled)/(float(self.h) ** 0.5)
        return out

model = miniCNN_1d(hidden, filtersize, bias=args.bias).to(device)
stud_trtr, stud_tetr, dump = compute_kernels(model, x_train, x_test)

if args.avg is not None:

    for counter in range(args.avg-1):
        model = miniCNN_1d(hidden, filtersize, bias=args.bias).to(device)
        new_trtr, new_tetr, dump = compute_kernels(model, x_train, x_test)
        stud_trtr.add_(new_trtr)
        stud_tetr.add_(new_tetr)
        del new_trtr, new_tetr, dump

    stud_trtr /= args.avg
    stud_tetr /= args.avg

if args.ridge is None:
    ridge = 0

mse = kernel_regression(stud_trtr, stud_tetr, y_train, y_test, ridge, device)


'''
	SAVE RESULTS (MSE)
'''
filename = 'grf_convntk'
filename += '_d' + str(args.imagesize)
if args.distro == 'cube':
    filename += 'cb'
elif args.distro == 'sphere':
    filename += 'sp'
filename += '_' + str(args.teacher)
filename += '_t' + str(args.patternsize)
filename += '_P' + str(args.trainsize) + '-' + str(args.testsize)

filename += '_s' + str(args.filtersize)
filename += '_h' + str(args.hidden)
if args.avg is not None:
    filename += '_avg' + str(args.avg)
if args.bias:
    filename += '_bias'
if args.ridge is not None:
    filename += '_r' + str(args.ridge)
if args.array is not None:
    filename += '_' + str(args.array)

torch.save({
            'args': args,
            'mse': mse,
           }, filename +'.pt')
