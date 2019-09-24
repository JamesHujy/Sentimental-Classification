import argparse
import os
import sys

from cnn_model import TextCNN
from lstm_model import LSTM
import train
from dataloader import dataset
from torch.utils.data import DataLoader,Dataset

parser = argparse.ArgumentParser(description='CNN text classificer')
parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
parser.add_argument('-epochs', type=int, default=32, help='number of epochs for train [default: 256]')
parser.add_argument('-batch_size', type=int, default=64, help='batch size for training [default: 64]')
parser.add_argument('-logger-interval',  type=int, default=10,   help='how many steps to wait to log training status')
parser.add_argument('-test-interval', type=int, default=100, help='how many steps to wait before testing [default: 100]')
parser.add_argument('-save-interval', type=int, default=50, help='how many steps to wait before saving [default:500]')
parser.add_argument('-save-dir', type=str, default='./checkpoints/', help='where to save the snapshot')
parser.add_argument('-max_length', type=int, default=1500, help='whether to save when get best performance')
parser.add_argument('-model_path', type=str, default='./checkpoints/cnn_300_0.001_epoch20.pt', help='set the model saved path to load model')
parser.add_argument('-nn_kind', type=str, default='cnn', help='set the type of the nn')
parser.add_argument('-classify', type=bool, default=False, help='set whether classify or regression')
# data
parser.add_argument('-shuffle', action='store_true', default=False, help='shuffle the data every epoch')
# model
parser.add_argument('-dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
parser.add_argument('-max-norm', type=float, default=3.0, help='l2 constraint of parameters [default: 3.0]')
parser.add_argument('-embed-size', type=int, default=300, help='number of embedding dimension [default: 300]')
parser.add_argument('-kernel-num', type=int, default=100, help='number of each kind of kernel')
parser.add_argument('-kernel-size', type=str, default='3,4,5', help='comma-separated kernel size to use for convolution')
parser.add_argument('-hidden_size', type=int, default=300, help='set the LSTM hidden_size')
parser.add_argument('-n_layers', type=int, default=2, help='set the number of layers')
parser.add_argument('-static', action='store_true', default=False, help='fix the embedding')
# device
parser.add_argument('-device', type=int, default=2, help='device to use for iterate data, -1 mean cpu [default: -1]')
# option
parser.add_argument('-predict', type=str, default=None, help='predict the sentence given')
parser.add_argument('-test', type=bool, default=False, help='train or test')
args = parser.parse_args()

args.kernel_size = [int(k) for k in args.kernel_size.split(',')]
args.class_num = 8

training_set = dataset(args)
label_weight = training_set.labelweight()
training_iter = DataLoader(dataset=training_set, batch_size=args.batch_size, num_workers=args.device, shuffle=True)

embed = training_set.getembed()
if args.nn_kind == 'cnn':
    model = TextCNN(args, embed)
else:
    model = LSTM(args, embed)

test_set = dataset(args, train=False)
test_iter = DataLoader(dataset=test_set, batch_size=1, num_workers=args.device, shuffle=True)

if not args.test:
    train.train(model, args, training_iter,test_iter,label_weight)
else:
    train.test(model, args, test_iter, args.model_path)




