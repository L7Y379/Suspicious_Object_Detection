import torch
import crnn
import random
import numpy as np
from torch import nn
from torch.autograd import Variable
from config import *

x = np.genfromtxt('data/phase_data.csv', dtype=np.float32, delimiter=',', encoding='utf-8')
y = np.genfromtxt('data/phase_label.csv', dtype=np.float32, delimiter=',', encoding='utf-8')
y = np.argmax(y, 1)

s = 720
train_x = np.concatenate([x[:s],x[s+240:]])#, x[s+5:s+240:10]
train_y = np.concatenate([y[:s],y[s+240:]])#, y[s+5:s+240:10]
test_x = np.concatenate([x[s+1:s+240:10],x[s+2:s+240:10],x[s+3:s+240:10],x[s+4:s+240:10],x[s+0:s+240:10],x[s+6:s+240:10],x[s+7:s+240:10],x[s+8:s+240:10],x[s+9:s+240:10]])
test_y = np.concatenate([y[s+1:s+240:10],y[s+2:s+240:10],y[s+3:s+240:10],y[s+4:s+240:10],y[s+0:s+240:10],y[s+6:s+240:10],y[s+7:s+240:10],y[s+8:s+240:10],y[s+9:s+240:10]])

train_x = torch.from_numpy(train_x)
train_y = torch.from_numpy(train_y)
test_x = torch.from_numpy(test_x)
test_y = torch.from_numpy(test_y)

print('train dataset:', train_x.shape, train_y.shape, 'test dataset:', test_x.shape, test_y.shape)

model = crnn.CRNN()
model.cuda()
opt = torch.optim.Adam(model.parameters())
loss = nn.CrossEntropyLoss()
max_acc = 0

for epoch in range(1, num_epochs+1):
    index = list(range(len(train_x)))
    random.shuffle(index)
    i = 0
    while i * batch_size < len(index):
        batch_x = train_x[index[i*batch_size:(i+1)*batch_size]]
        batch_y = train_y[index[i*batch_size:(i+1)*batch_size]]
        batch_x = batch_x.view(-1, 1, input_size, time_step)
        batch_x, batch_y = Variable(batch_x).cuda(), Variable(batch_y).cuda()
        output = model(batch_x)
        cost = loss(output, batch_y.long())
        opt.zero_grad()
        cost.backward()
        opt.step()
        train_out = model(batch_x)
        test_out = model(Variable(test_x.view(-1, 1, input_size, time_step)).cuda())

        train_pred_y = torch.max(train_out, 1)[1].cpu().numpy()
        test_pred_y = torch.max(test_out, 1)[1].cpu().numpy()

        train_acc = (train_pred_y == batch_y.cpu().numpy()).mean()
        test_acc = (test_pred_y == test_y.numpy()).mean()
        max_acc = max(max_acc, test_acc)
        if i % 5 == 0:
            print('Epoch: ', epoch, '| loss: %.4f' % cost.data.cpu().numpy(), '| train acc: %.3f' % train_acc,
                  '| test acc: %.3f' % test_acc, '| max acc: %.3f' % max_acc)
        i += 1
print('max test acc: ', max_acc)