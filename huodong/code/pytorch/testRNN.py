import torch
import rnn
from config import *
import numpy as np
from torch import nn
import random
from torch.autograd import Variable

x = np.genfromtxt(data_path, dtype=np.float32, delimiter=',', encoding='utf-8')
y = np.genfromtxt(label_path, dtype=np.float32, delimiter=',', encoding='utf-8')
y_ = np.genfromtxt('E:/gzyPyCharmWork/datasets/dtw_test_data/420_dtw_soft_label.csv', dtype=np.float32, delimiter=',', encoding='utf-8')

length = len(x)
train_x = x
train_y = np.concatenate([y[0:40], y_[0:40], y[80:120], y_[40:80], y[160:200], y_[80:120], y[240:280], y_[120:160]])
test_x = np.concatenate([x[40:80], x[120:160], x[200:240], x[280:320]])
test_y = np.concatenate([y[40:80], y[120:160], y[200:240], y[280:320]])

train_y = np.argmax(train_y, 1)
test_y = np.argmax(test_y, 1)
train_x = torch.from_numpy(train_x)
train_y = torch.from_numpy(train_y)
test_x = torch.from_numpy(test_x)
test_y = torch.from_numpy(test_y)

print('training shape:', train_x.shape, train_y.shape, 'testing shape:', test_x.shape, test_y.shape)

model = rnn.RNN()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_func = nn.CrossEntropyLoss()
max_acc = 0

for epoch in range(1, num_epochs + 1):
    index = list(range(len(train_x)))
    random.shuffle(index)
    i = 0
    while i * batch_size < len(index):
        batch_x, batch_y = train_x[index[i * batch_size:(i + 1) * batch_size]], train_y[index[i * batch_size:(i + 1) * batch_size]]
        batch_x = batch_x.view(-1, time_step, input_size)
        batch_x, batch_y = Variable(batch_x), Variable(batch_y)
        output,_ = model(batch_x, 0, 0, 0, False)
        loss = loss_func(output, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_output, _ = model(train_x.view(-1, time_step, input_size), 0, 0, 0, False)
        test_output, _ = model(test_x.view(-1, time_step, input_size), 0, 0, 0, False)
        train_pred_y = torch.max(train_output, 1)[1].data.numpy()
        test_pred_y = torch.max(test_output, 1)[1].data.numpy()
        train_accuracy = float((train_pred_y == train_y.numpy()).astype(int).sum()) / float(train_pred_y.size)
        test_accuracy = float((test_pred_y == test_y.numpy()).astype(int).sum()) / float(test_pred_y.size)
        max_acc = max(max_acc, test_accuracy)
        if i % 5 == 0:
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| train accuracy: %.2f' % train_accuracy,'| test accuracy: %.2f' % test_accuracy)
        i += 1
print('max test acc:', max_acc)