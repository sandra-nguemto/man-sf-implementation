from __future__ import division, print_function

import os
import glob
import time
import random
import argparse
import numpy as np
import torch
from collections import defaultdict
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from sklearn.metrics import f1_score, matthews_corrcoef, confusion_matrix, classification_report
import pickle

from utils import load_data, accuracy  # load_data: load wiki relations data
from model import GAT

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no_cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--sparse', action='store_true', default=False, help='GAT with sparse version or not.')
parser.add_argument('--seed', type=int, default=14, help='Random seed.')
parser.add_argument('--epochs', type=int, default=5000, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=5e-4, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=64, help='Number of hidden units.')
parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.38, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=100, help='Patience')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

device = torch.device("cuda:0" if args.cuda else "cpu")
print("Device:", device)

# Load data for GAT
adj = load_data()
stock_num = adj.size(0)
adj = adj.to(device)

train_price_path = "./Data/train_price/"
train_label_path = "./Data/train_label/"
train_text_path = "./Data/train_text/"
test_price_path = "./Data/test_price/"
test_label_path = "./Data/test_label/"
test_text_path = "./Data/test_text/"
num_samples = len(os.listdir(train_price_path))

cross_entropy = nn.CrossEntropyLoss(weight=torch.tensor([1.00, 1.00]).to(device))

def train(epoch, TRAIN_SIZE):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    i = np.random.randint(TRAIN_SIZE)
    train_text = torch.tensor(np.load(train_text_path + str(i).zfill(10) + '.npy'), dtype=torch.float32).to(device)
    train_price = torch.tensor(np.load(train_price_path + str(i).zfill(10) + '.npy'), dtype=torch.float32).to(device)
    train_label = torch.unsqueeze(torch.LongTensor(np.load(train_label_path + str(i).zfill(10) + '.npy')), 0).to(device)
    output = model(train_text, train_price, adj)
    loss_train = cross_entropy(output, train_label)
    acc_train = accuracy(output, train_label)
    loss_train.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print("Epoch:", epoch, ", Training loss =", loss_train.item(), ", Accuracy =", acc_train.item())

def test(TEST_SIZE):
    model.eval()
    test_acc = []
    test_loss = []
    li_pred = []
    li_true = []
    with torch.no_grad():
        for i in range(TEST_SIZE):
            test_text = torch.tensor(np.load(test_text_path + str(i).zfill(10) + '.npy'), dtype=torch.float32).to(device)
            test_price = torch.tensor(np.load(test_price_path + str(i).zfill(10) + '.npy'), dtype=torch.float32).to(device)
            test_label = torch.unsqueeze(torch.LongTensor(np.load(test_label_path + str(i).zfill(10) + '.npy')), 0).to(device)
            output = model(test_text, test_price, adj)
            loss_test = cross_entropy(output, test_label)
            acc_test = accuracy(output, test_label)
            a = output.argmax(1).cpu().numpy()
            b = test_label.cpu().numpy()
            li_pred.append(a)
            li_true.append(b)
            test_loss.append(loss_test.item())
            test_acc.append(acc_test.item())
    iop = f1_score(np.array(li_true).reshape((-1,)), np.array(li_pred).reshape((-1,)), average='micro')
    mat = matthews_corrcoef(np.array(li_true).reshape((-1,)), np.array(li_pred).reshape((-1,)))
    print("Test set results:",
          "loss= {:.4f}".format(np.array(test_loss).mean()),
          "accuracy= {:.4f}".format(np.array(test_acc).mean()),
          "F1 score={:.4f}".format(iop),
          "MCC = {:.4f}".format(mat))

model = GAT(text_input_shape=512,
            price_input_shape=3,
            num_stocks=stock_num,
            hidden_size=args.hidden,
            num_heads=args.nb_heads,
            num_classes=2,
            dropout=args.dropout,
            alpha=args.alpha).to(device)

optimizer = optim.Adam(model.parameters(),
                       lr=args.lr,
                       weight_decay=args.weight_decay)

TRAIN_SIZE = 300
TEST_SIZE = 90
print('epochs:', args.epochs)
for epoch in range(args.epochs):
    train(epoch, TRAIN_SIZE)
    if epoch % 100 == 0:
        torch.save(model.state_dict(), "./weight.pth")
        test(TEST_SIZE)
print("Optimization Finished!")
