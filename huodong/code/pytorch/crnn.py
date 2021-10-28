import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from config import *
import lmmd

class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
        self.pool1 = torch.nn.MaxPool2d(2, stride=(3,2))
        self.conv2 = torch.nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.pool2 = torch.nn.MaxPool2d(2, stride=(3,2))
        self.conv3 = torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)

    def forward(self, input):
        x = F.relu(self.conv1(input), inplace=True)
        x = self.pool1(x)
        x = F.relu(self.conv2(x), inplace=True)
        x = self.pool2(x)
        x = F.relu(self.conv3(x), inplace=True)
        return x

class BLSTM(torch.nn.Module):
    def __init__(self, nIn, nHidden, classes):
        super(BLSTM, self).__init__()
        self.rnn = nn.LSTM(input_size=nIn, hidden_size=nHidden, num_layers=1, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, classes)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        time_steps, batch_size, channel = recurrent.size()
        recurrent = recurrent.view(time_steps * batch_size, channel)
        output = self.embedding(recurrent)
        output = output.view(time_steps, batch_size, -1)
        return output[-1,:,:]

class CRNN(torch.nn.Module):
    def __init__(self, class_num=num_classes, hidden_unit=hidden_size):
        super(CRNN, self).__init__()
        self.CNN = torch.nn.Sequential()
        self.CNN.add_module('CNN', CNN())
        self.RNN = torch.nn.Sequential()
        self.RNN.add_module('BLSTM', BLSTM(fc_size, hidden_unit, class_num))

    def forward(self, source, target, slabel, tlabel):
        s = self.CNN(source)
        b, c, h, w = s.size()
        s = s.view(b, c*h, w)
        s = s.permute(2, 0, 1)
        output = self.RNN(s)
        t = self.CNN(target)
        b2, c2, h2, w2 = t.size()
        t = t.view(b2, c2 * h2, w2)
        t = t.permute(2, 0, 1)
        output2 = self.RNN(t)
        loss = lmmd.cal_mmd(output, output2, torch.argmax(slabel, 1), torch.argmax(tlabel, 1))
        return output, loss
