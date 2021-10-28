# -*-coding: utf-8 -*-
import torch
import torch.nn as nn
import lmmd
from config import *
from torch.autograd import Variable


# RNN Model (Many-to-One)
class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, s, t, s_label, t_label, flag):
        # 设置初始状态h_0与c_0的状态是初始的状态，一般设置为0，尺寸是,x.size(0)
        # h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda())
        # c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda())

        # Forward propagate RNN
        s_out, (h_n, c_n) = self.lstm(s, None)  # 送入一个初始的x值，作为输入以及(h0, c0)
        if flag:
            t_out, _ = self.lstm(t, None)
            loss = lmmd.cal_mmd(s_out[:,-1,:], t_out[:,-1,:], s_label, t_label)
        else:
            loss = 0
        # Decode hidden state of last time step
        out = self.fc(s_out[:, -1, :])  # output也是batch_first, 实际上h_n与c_n并不是batch_first
        return out, loss

# rnn = RNN(input_size, hidden_size, num_layers, num_classes)
# rnn.cuda()
#
# # Loss and Optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)
#
# # Train the Model
# for epoch in range(num_epochs):
#     for i, (images, labels) in enumerate(train_loader):
#         # a = images.numpy()
#         images = Variable(images.view(-1, sequence_length, input_size)).cuda()  # 100*1*28*28 -> 100*28*28
#         # b = images.data.cpu().numpy()
#         labels = Variable(labels).cuda()
#
#         # Forward + Backward + Optimize
#         optimizer.zero_grad()
#         outputs = rnn(images)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#
#         if (i+1) % 100 == 0:
#             print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
#                    %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.data[0]))
#
# # Test the Model
# correct = 0
# total = 0
# for images, labels in test_loader:
#     images = Variable(images.view(-1, sequence_length, input_size)).cuda()
#     outputs = rnn(images)
#     _, predicted = torch.max(outputs.data, 1)
#     total += labels.size(0)
#     correct += (predicted.cpu() == labels).sum()
#
# print('Test Accuracy of the model on the 10000 test images: %d %%' % (100 * correct / total))

# Save the Model
# torch.save(rnn.state_dict(), 'rnn.pkl')