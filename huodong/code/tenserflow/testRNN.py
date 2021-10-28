import tensorflow as tf
import numpy as np


def Rnn(data, weight, biase):
    x = tf.reshape(data, [-1, 200, 90])
    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, 90])
    x = tf.split(x, 200)

    lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(128, forget_bias=1.0)
    lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(128, forget_bias=1.0)

    out, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype=tf.float32)
    out = tf.add(tf.matmul(out[-1], weight), biase)
    return out, tf.nn.softmax(out)

data = np.genfromtxt('datasets/widar3/link1/room1/position/data.csv', dtype=float, delimiter=',',encoding='utf-8')
label = np.genfromtxt('datasets/widar3/link1/room1/position/label.csv', dtype=float, delimiter=',',encoding='utf-8')
# test_data = np.zeros([40, 54000])
# train_data = np.zeros([160, 54000])
# test_label = np.zeros([40, 4])
# train_label = np.zeros([160, 4])
# m,n = 0,0
# for i in range(960):
#     if i % 5 == 0:
#         test_data[m, :] = data[i, :]
#         test_label[m, :] = label[i, :]
#         m = m + 1
#     else:
#         train_data[n, :] = data[i, :]
#         train_label[n, :] = label[i, :]
#         n = n + 1
# train_data = np.concatenate([data[:600:10,:], data[1:600:10,:], data[2:600:10,:], data[3:600:10,:],
#                              data[4:600:10,:], data[5:600:10,:], data[6:600:10,:], data[7:600:10,:],
#                              data[8:600:10,:]])
# test_data = data[9:600:10, :]
# train_label = np.concatenate([label[:600:10,:], label[1:600:10,:], label[2:600:10,:], label[3:600:10,:],
#                               label[4:600:10,:], label[5:600:10,:], label[6:600:10,:], label[7:600:10,:],
#                               label[8:600:10,:]])
# test_label = label[9:600:10, :]
train_data = data[120:480,:]
test_data = data[:120,:]
train_label = label[120:480,:]
test_label = label[:120,:]

X = tf.placeholder(tf.float32, shape=[None, 18000])
Y = tf.placeholder(tf.float32, shape=[None, 6])
W = tf.Variable(tf.random_normal([2 * 128, 6]))
B = tf.Variable(tf.random_normal([6]))
logits, pred = Rnn(X, W, B)
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1)), tf.float32))
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=logits))
optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
max_test_acc = 0
for epoch in range(1000):
    state = np.random.get_state()
    np.random.shuffle(train_data)
    np.random.set_state(state)
    np.random.shuffle(train_label)
    step = 0
    while step * 20 < len(train_data):
        batch_x = train_data[step * 20:(step + 1) * 20]
        batch_y = train_label[step * 20:(step + 1) * 20]
        sess.run(optimizer, feed_dict={X: batch_x, Y: batch_y})
        step += 1
    train_acc = sess.run(accuracy, feed_dict={X: train_data, Y: train_label})
    test_acc = sess.run(accuracy, feed_dict={X: test_data, Y: test_label})

    if max_test_acc < test_acc:
        max_test_acc = test_acc

    if (epoch + 1) % 10 == 0:
        print('Epoch:{}'.format(epoch + 1), 'train acc:{:.4}'.format(train_acc),
              'test acc:{:.4}'.format(test_acc))
print('Train over! ', 'max test acc:{:.4}'.format(max_test_acc))