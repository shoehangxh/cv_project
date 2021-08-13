from __future__ import print_function
import argparse
from math import log10

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from voc_dataset import vocDataset as DataSet
from model import EzDetectNet
from model import EzDetectConfig
from loss import EzDetectLoss

#Training settings
parser = argparse.ArgumentParser(description='EasyDetect by pytorch')
parser.add_argument('--batchSize', type=int, default=16, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=4, help='testing batch size')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=1024, help='random seed to use')
parser.add_argument('--gpu',dest='gpu', action='stor_true')
parser.set_defaults(gpu=True)
opt = parser.parse_args()
torch.cuda.set_device(1)

print('--->loading datasets')
ezConfig = EzDetectConfig(opt.batchSize, opt.gpu)
train_set = DataSet(ezConfig, True)
test_set = DataSet(ezConfig, False)
train_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
test_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.batchSize)
print('----> building model')
mymodel = EzDetectNet(ezConfig, True)
myloss = EzDetectLoss(ezConfig)
optimizer = optim.SGD(mymodel.parameters(), lr=opt.lr, momentum=0.9, weight_decay=1e-4)
if ezConfig.gpu == True:
    mymodel.cuda()
    myloss.cuda()
def adjust_learning_rate(optimizer, epoch):
    '''每迭代10个epoch，学习率下降0.1倍'''
    lr = opt.lr * (0.1 ** (epoch // 10))
    for parma_group in optimizer.param_groups:
        param_group['lr'] = lr

def doTrain(t):
    mymodel.train()
    for i, batch in enumerate(train_data_loader):
        batchX = batch[0]
        target = batch[1]
        if ezConfig.gpu:
            batchX = batch[0].cuda()
            target = batch[1].cuda()

        x = torch.autograd.Variable(batchX, requires_grad=False)
        confOut, bboxOut = mymodel(x)
        confLoss, bboxLoss = myloss(confbboxOut, bboxOut, target)
        totalLoss = confLoss * 4 + bboxLoss
        print(confLoss, bboxLoss)
        print('{} : {} / {} >>>>>>: {}'.format(t, i, len(train_data_loader), totalLoss.data[0]))
        optimizer.zero_grad()
        totalLoss.backward()
        optimizer.step()

def doValidate():
    mymodel.eval()
    lossSum = 0.0
    for i, batch in enumerate(test_data_loader):
        batchX = batch[0]
        target = batch[1]
        if ezConfig.gpu:
            batchX = batch[0].cuda()
            target = batch[1].cuda()

        x = torch.autograd.Variable(batchX, requires_grad=False)
        confOut, bboxOut = mymodel(x)
        confLoss, bboxLoss = myloss(confbboxOut, bboxOut, target)
        totalLoss = confLoss * 4 + bboxLoss
        print(confLoss, bboxLoss)
        print('Test : {} / {} >>>>>>: {}'.format(i, len(test_data_loader), totalLoss.data[0]))
        lossSum = totalLoss.data[0] + lossSum
    score = lossSum / len(test_data_loader)
    print('######:{}'.format(score))
    return score
#main function
for t in range(50):
    adjust_learning_rate(optimizer, 5)
    doTrain(t)
    score = doValidate()
    if(t % 5 == 0):
        torch.save(mymodel.state_dict(), 'model/model_{}_{}.pth'.format(t, str(score)[:4]))