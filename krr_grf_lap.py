import sys
import copy
import argparse

import torch
import torch.nn.functional as F
import math

from kernels import compute_kernels, gram_lap_1d, gram_llap_1d, gram_clap_1d
from myutils import hypersphere_random_sampler, hypercube_random_sampler, grf_generator, kernel_regression

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float64)

parser = argparse.ArgumentParser(description='kernel regression with gaussian random field')
'''patterns
	DATASET ARGS
'''
parser.add_argument('--trainsize', metavar='P', type=int, help='size of the training set')
parser.add_argument('--testsize', metavar='TEST', type=int, help='size of the training set')
parser.add_argument('--distro', type=str, help='cube, sphere')
parser.add_argument('--imagesize', metavar='IM.SIZE', type=int, help='size of the images')
parser.add_argument('--patternsize', metavar='PAT.SIZE', type=int, help='size of the local patterns')
parser.add_argument('--teacher', type=str, help='teacher: loc, conv')
parser.add_argument('--student', type=str, help='student: fc, loc, conv')
parser.add_argument('--filtersize', metavar='FILT.SIZE', type=int, help='size of the student filters', default=None)
parser.add_argument('--ridge', type=float, help='regularisation', default=None)
parser.add_argument('--scale', action='store_true', default=False)

'''
	OUTPUT ARGS
'''
parser.add_argument('--array', type=int, help='index for array runs', default=None)
args = parser.parse_args()

trainsize = args.trainsize
imagesize = args.imagesize
patternsize = args.patternsize
testsize = args.testsize
'''
	GENERATE GAUSSIAN FIELD WITH CONVOLUTIONAL COV.
'''
if args.distro == 'cube':
    x = hypercube_random_sampler(trainsize + testsize, imagesize, device)
elif args.distro == 'sphere':
    x = hypersphere_random_sampler(trainsize + testsize, imagesize, device)
else:
    raise AssertionError ('distribution not implemented')
if args.teacher == 'loc':
    teacher_cov = gram_llap_1d(x.reshape(trainsize + testsize, 1, imagesize), x.reshape(trainsize + testsize, 1, imagesize), patternsize, sigma=patternsize, pbc=True)
if args.teacher == 'conv':
    teacher_cov = gram_clap_1d(x.reshape(trainsize + testsize, 1, imagesize), x.reshape(trainsize + testsize, 1, imagesize), patternsize, sigma=patternsize, pbc=True)
y = grf_generator( teacher_cov, device)

x_train = x[:trainsize]
y_train = y[:trainsize]

x_test = x[trainsize:]
y_test = y[trainsize:]

if args.student == 'fc':

    student_trtr = gram_lap_1d(x_train.reshape(trainsize, 1, imagesize), x_train.reshape(trainsize, 1, imagesize))
    student_tetr = gram_lap_1d(x_test.reshape(testsize, 1, imagesize), x_train.reshape(trainsize, 1, imagesize))

else:
    assert args.filtersize is not None, "provide filtersize"
    
    if args.student == 'loc':
        student_trtr = gram_llap_1d(x_train.reshape(trainsize, 1, imagesize), x_train.reshape(trainsize, 1, imagesize), filtersize, sigma=filtersize, pbc=True)
        student_tetr = gram_llap_1d(x_test.reshape(testsize, 1, imagesize), x_train.reshape(trainsize, 1, imagesize), filtersize, sigma=filtersize, pbc=True)
    elif args.student == 'conv':
        student_trtr = gram_clap_1d(x_train.reshape(trainsize, 1, imagesize), x_train.reshape(trainsize, 1, imagesize), filtersize, sigma=filtersize, pbc=True)
        student_tetr = gram_clap_1d(x_test.reshape(testsize, 1, imagesize), x_train.reshape(trainsize, 1, imagesize), filtersize, sigma=filtersize, pbc=True)

if args.ridge is not None:
    ridge = args.ridge
    if args.scale:
        ridge *= trainsize

else:
    ridge = 0

mse = kernel_regression(student_trtr, student_tetr, y_train, y_test, ridge, device)

filename = 'grf_krr'
filename += '_d' + str(args.imagesize)
filename += '_s' + str(args.patternsize)
filename += '_P' + str(args.trainsize)
if args.ridge is not None:
    filename += '_l' + str(args.ridge)
    if args.scale:
        filename += 'P'
filename += '_' + args.student
if args.filtersize is not None:
    filename += '_fs' + str(args.filtersize)
if args.array is not None:
    filename += '_' + str(args.array)
    
torch.save({
            'args': args,
            'mse': mse,
           }, filename +'.pt')
