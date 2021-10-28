import tensorflow as tf
import numpy as np
import random
import time
import datetime
from scipy import stats
from sklearn.metrics import confusion_matrix

class Model(object):
    def __init__(self):
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        self.input = 54000
        self.output = 3
        # 标签模型参数
        self.hidden = 128
        self.timestep = 200
        self.feature = 270
        self.learning_rate = 1e-4
        self.epochs = 500
        self.batch_size = 40
        self.display = 50
        self.prediction = None
        self.score = None
        self.decode_target = None
        self.decode_unseen = None
        # 分类模型参数
        self.encode_1 = 1024
        self.encode_2 = 256
        self.decode_1 = 1024

    def Rnn(self, date, weight, biase):
        x = tf.reshape(date, [-1, self.timestep, self.feature])
        x = tf.transpose(x, [1, 0, 2])
        x = tf.reshape(x, [-1, self.feature])
        x = tf.split(x, self.timestep)

        lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(self.hidden, forget_bias=1.0)
        lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(self.hidden, forget_bias=1.0)

        out, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype=tf.float32)
        out = tf.add(tf.matmul(out[-1], weight), biase)
        return out, tf.nn.softmax(out)

    def AutoEncode(self, data, weight, biase):
        data = tf.nn.relu(tf.add(tf.matmul(data, weight[0]), biase[0]))
        data = tf.nn.relu(tf.add(tf.matmul(data, weight[1]), biase[1]))
        data = tf.nn.relu(tf.add(tf.matmul(data, weight[2]), biase[2]))
        data = tf.add(tf.matmul(data, weight[3]), biase[3])
        return data


    def ComputeScore(self, pred, true):
        pred = np.argmax(pred, 1)
        true = np.argmax(true, 1)
        accuracy = []
        for i in range(self.output):
            if len(pred[pred == i]) == 0:
                accuracy.append(0.)
            else:
                tmp = np.equal(pred[true == i], true[true == i])
                accuracy.append(len(tmp[tmp == True]) / len(pred[pred == i]))
        # print('各类别精度：', accuracy)
        return np.array(accuracy)

    def TrainLabel(self, train_data, train_label, test_data, test_label, target_data, target_label):
        with self.sess.as_default():
            with self.graph.as_default():
                #---------------------模型构建-----------------------#
                X = tf.placeholder(tf.float32, shape=[None, self.input])
                Y = tf.placeholder(tf.float32, shape=[None, self.output])
                W = tf.Variable(tf.random_normal([2 * self.hidden, self.output]))
                B = tf.Variable(tf.random_normal([self.output]))
                logits, pred = self.Rnn(X, W, B)
                accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1)), tf.float32))
                loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=logits))
                optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)
                # ---------------------训练模型-----------------------#
                self.sess.run(tf.global_variables_initializer())
                max_test_acc, max_target_acc = 0.,0.
                for epoch in range(self.epochs):
                    state = np.random.get_state()
                    np.random.shuffle(train_data)
                    np.random.set_state(state)
                    np.random.shuffle(train_label)
                    step = 0
                    while step * self.batch_size < len(train_data):
                        batch_x = train_data[step*self.batch_size:(step+1)*self.batch_size]
                        batch_y = train_label[step*self.batch_size:(step+1)*self.batch_size]
                        self.sess.run(optimizer, feed_dict={X:batch_x, Y:batch_y})
                        step += 1
                    train_acc = self.sess.run(accuracy, feed_dict={X:train_data, Y:train_label})
                    test_acc = self.sess.run(accuracy, feed_dict={X:test_data, Y:test_label})
                    target_acc = self.sess.run(accuracy, feed_dict={X:target_data, Y:target_label})
                    max_target_acc = max(max_target_acc, target_acc)
                    if max_test_acc < test_acc:
                        max_test_acc = test_acc
                        test_pred = self.sess.run(pred, feed_dict={X:test_data})
                        self.score = self.ComputeScore(test_pred, test_label) * test_acc
                        self.prediction = self.sess.run(pred, feed_dict={X:target_data})
                        for predict in self.prediction:
                            predict = np.multiply(predict, self.score)
                    if (epoch+1) % self.display == 0:
                        print('Epoch:{}'.format(epoch+1), 'train acc:{:.4}'.format(train_acc),
                              'test acc:{:.4}'.format(test_acc), 'target acc:{:.4}'.format(target_acc))
                print('Train over! ','max test acc:{:.4}'.format(max_test_acc),'max target acc:{:.4}'.format(max_target_acc))
        self.sess.close()
        return self.prediction

    def TrainEncoder(self, source_data, target_data, unseen_data):
        with self.sess.as_default():
            with self.graph.as_default():
                #---------------------模型构建-----------------------#
                X = tf.placeholder(tf.float32, shape=[None, self.input])
                Y = tf.placeholder(tf.float32, shape=[None, self.input])
                W_En1 = tf.Variable(tf.random_normal([self.input, self.encode_1]))
                B_En1 = tf.Variable(tf.random_normal([self.encode_1]))
                W_En2 = tf.Variable(tf.random_normal([self.encode_1, self.encode_2]))
                B_En2 = tf.Variable(tf.random_normal([self.encode_2]))
                W_De1_s = tf.Variable(tf.random_normal([self.encode_2, self.decode_1]))
                B_De1_s = tf.Variable(tf.random_normal([self.decode_1]))
                W_De2_s = tf.Variable(tf.random_normal([self.decode_1, self.input]))
                B_De2_s = tf.Variable(tf.random_normal([self.input]))
                W_De1_t = tf.Variable(tf.random_normal([self.encode_2, self.decode_1]))
                B_De1_t = tf.Variable(tf.random_normal([self.decode_1]))
                W_De2_t = tf.Variable(tf.random_normal([self.decode_1, self.input]))
                B_De2_t = tf.Variable(tf.random_normal([self.input]))
                # 源域模型
                logits_s = self.AutoEncode(X, [W_En1,W_En2,W_De1_s,W_De2_s], [B_En1,B_En2,B_De1_s,B_De2_s])
                loss_s = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=logits_s))
                optimizer_s = tf.train.AdamOptimizer(self.learning_rate).minimize(loss_s)
                # 目标域模型
                logits_t = self.AutoEncode(X, [W_En1,W_En2,W_De1_t,W_De2_t], [B_En1,B_En2,B_De1_t,B_De2_t])
                loss_t = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=logits_t))
                optimizer_t = tf.train.AdamOptimizer(self.learning_rate).minimize(loss_t)
                # ---------------------训练模型-----------------------#
                self.sess.run(tf.global_variables_initializer())
                for epoch in range(self.epochs):
                    step = 0
                    while step * self.batch_size < len(source_data):
                        batch_x = source_data[step*self.batch_size:(step+1)*self.batch_size]
                        batch_y = source_data[step*self.batch_size:(step+1)*self.batch_size]
                        _,source_loss = self.sess.run([optimizer_s,loss_s], feed_dict={X:batch_x, Y:batch_y})
                        batch_x = target_data[step*self.batch_size:(step+1)*self.batch_size]
                        batch_y = source_data[step*self.batch_size:(step+1)*self.batch_size]
                        _,target_loss = self.sess.run([optimizer_t,loss_t], feed_dict={X:batch_x, Y:batch_y})
                        step += 1
                    if (epoch+1) % self.display == 0:
                        print('Epoch:{}'.format(epoch+1), 'source loss:{:.4}'.format(source_loss),
                              'target loss:{:.4}'.format(target_loss))
                print('Train over! ')
                self.decode_target = self.sess.run(logits_t, feed_dict={X:target_data})
                self.decode_unseen = self.sess.run(logits_t, feed_dict={X:unseen_data})
        self.sess.close()
        return self.decode_target, self.decode_unseen
    
    def TrainClassifier(self, source_data, source_label, target_data, target_label):
        with self.sess.as_default():
            with self.graph.as_default():
                #---------------------模型构建-----------------------#
                X = tf.placeholder(tf.float32, shape=[None, self.input])
                Y = tf.placeholder(tf.float32, shape=[None, self.output])
                W = tf.Variable(tf.random_normal([2 * self.hidden, self.output]))
                B = tf.Variable(tf.random_normal([self.output]))
                logits, pred = self.Rnn(X, W, B)
                accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1)), tf.float32))
                loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=logits))
                # train_loss_summary = tf.summary.scalar('train_loss', loss)
                # test_loss_summary = tf.summary.scalar('test_loss', loss)
                optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)
                # ---------------------训练模型-----------------------#
                self.sess.run(tf.global_variables_initializer())
                # train_merged = tf.summary.merge([train_loss_summary])
                # test_merged = tf.summary.merge([test_loss_summary])
                # writer = tf.summary.FileWriter('420_single_classifier_logs', self.sess.graph)
                max_test_acc, max_target_acc = 0.,0.
                length = len(source_data)
                index = list(range(length))
                random.shuffle(index)
                train_data = source_data[index[:length // 10 * 8]]
                train_label = source_label[index[:length // 10 * 8]]
                test_data = source_data[index[length // 10 * 8:]]
                test_label = source_label[index[length // 10 * 8:]]
                start_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
                for epoch in range(self.epochs):
                    state = np.random.get_state()
                    np.random.shuffle(train_data)
                    np.random.set_state(state)
                    np.random.shuffle(train_label)
                    step = 0
                    while step * self.batch_size < len(train_data):
                        batch_x = train_data[step*self.batch_size:(step+1)*self.batch_size]
                        batch_y = train_label[step*self.batch_size:(step+1)*self.batch_size]
                        self.sess.run(optimizer, feed_dict={X:batch_x, Y:batch_y})
                        step += 1
                    # writer.add_summary(self.sess.run(train_merged, feed_dict={X: train_data, Y: train_label}), epoch)
                    # writer.add_summary(self.sess.run(test_merged, feed_dict={X: test_data, Y: test_label}), epoch)
                    train_acc = self.sess.run(accuracy, feed_dict={X:train_data, Y:train_label})
                    test_start_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
                    test_acc = self.sess.run(accuracy, feed_dict={X:test_data, Y:test_label})
                    test_end_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
                    target_acc = self.sess.run(accuracy, feed_dict={X:target_data, Y:target_label})
                    max_test_acc = max(max_test_acc, test_acc)
                    max_target_acc = max(max_target_acc, target_acc)
                    if (epoch+1) % self.display == 0:
                        print('Epoch:{}'.format(epoch+1), 'train acc:{:.4}'.format(train_acc),
                              'test acc:{:.4}'.format(test_acc), 'target acc:{:.4}'.format(target_acc))
                        print("运行时间 ：",
                              datetime.datetime.strptime(test_end_time, '%Y-%m-%d %H:%M:%S') - datetime.datetime.strptime(
                                  test_start_time,
                                  '%Y-%m-%d %H:%M:%S'))
                print('Train over! ','max test acc:{:.4}'.format(max_test_acc),'max target acc:{:.4}'.format(max_target_acc))
                end_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
                print("运行时间 ：",
                      datetime.datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S') - datetime.datetime.strptime(start_time,
                                                                                                             '%Y-%m-%d %H:%M:%S'))
                test_pred = self.sess.run(pred, feed_dict={X: test_data})
                target_pred = self.sess.run(pred, feed_dict={X: target_data})
                print('test confusion matrix:\n', confusion_matrix(np.argmax(test_label, 1), np.argmax(test_pred, 1)))
                print('target confusion matrix:\n', confusion_matrix(np.argmax(target_label, 1), np.argmax(target_pred, 1)))
                self.prediction = np.argmax(self.sess.run(pred, feed_dict={X: target_data}), 1)
        self.sess.close()
        return self.prediction

if __name__ == '__main__':
    # 数据导入
    data = np.genfromtxt('datasets/different_location_activity/420_gzy_P8A4.csv', dtype=float, delimiter=',', encoding='utf-8')
    label = np.genfromtxt('datasets/different_location_activity/420_gzy_P8A4_Label.csv', dtype=float, delimiter=',', encoding='utf-8')
    data2 = np.genfromtxt('datasets/different_location_activity/5L_P20_A3.csv', dtype=float, delimiter=',', encoding='utf-8')
    label2 = np.genfromtxt('datasets/different_location_activity/5L_P20_A3_Label.csv', dtype=float, delimiter=',',
                          encoding='utf-8')
    # 数据划分
    data = data[:720]
    label = label[:720, :3]
    train = np.concatenate([data[:720:3], data2[0:450:6], data2[1:450:6]])
    train_label = np.concatenate([label[:720:3], label2[0:450:6], label2[1:450:6]])
    test = np.concatenate([data2[2:450:6], data2[3:450:6], data2[4:450:6]])
    test_label = np.concatenate([label2[2:450:6], label2[3:450:6], label2[4:450:6]])

    print(train.shape, train_label.shape, test.shape, test_label.shape)
    # _ = model.TrainLabel(source_train,source_train_label,source_test,source_test_label,target_train,target_train_label)
    # target, unseen = model.TrainEncoder(source_train,target_train,target_test)
    prediction = np.zeros([len(test), 1])
    for i in range(1):
        model = Model()
        prediction[:, i] = model.TrainClassifier(train,train_label,test,test_label)
    # print(prediction)
    # print('target confusion matrix:\n', confusion_matrix(np.argmax(test_label, 1), stats.mode(prediction, 1)[0]))