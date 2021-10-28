import math
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import random
import time
import datetime
from scipy import stats
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from minisom import MiniSom

class Model(object):
    def __init__(self):
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        self.input = 54000 # 输入样本维度
        self.output = 4 # 输出类别个数
        # RNN模型参数
        self.hidden = 128 # 隐藏层节点个数
        self.timestep = 200 # 时间步长
        self.feature = 270 # 特征维度
        self.learning_rate = 1e-4 # 学习率
        self.epochs = 500 # 训练轮次
        self.batch_size = 40 # 批大小
        self.display = 50 # 显示结果轮次

        # Meta模型参数
        self.weights = 6 * 4 # 权值
        self.biases = 4 # 偏置

        # 其他参数
        self.plabel = None
        self.train_res = None
        self.test_res = None
        self.target_res = None

    def Dtw(self, data, pattern):
        distance, path = fastdtw(data, pattern, dist=euclidean)
        return distance

    def Rnn(self, data, weight, biase):
        x = tf.reshape(data, [-1, self.timestep, self.feature])
        x = tf.transpose(x, [1, 0, 2])
        x = tf.reshape(x, [-1, self.feature])
        x = tf.split(x, self.timestep)

        lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(self.hidden, forget_bias=1.0)
        lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(self.hidden, forget_bias=1.0)

        out, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype=tf.float32)
        out = tf.add(tf.matmul(out[-1], weight), biase)
        return out, tf.nn.softmax(out)

    def Meta(self, data, weight, biase):
        data = tf.nn.relu(tf.add(tf.matmul(data, weight[0]), biase[0]))
        data = tf.add(tf.matmul(data, weight[1]), biase[1])
        return data, tf.nn.softmax(data)

    def TrainLabel(self, label_data, unlabel_data, label, unlabel):
        self.plabel = np.zeros([len(unlabel_data), len(label_data)])
        # 计算每个样本与模板之间的距离
        for i in range(len(unlabel_data)):
            unlabel_tmp = unlabel_data[i, :]
            unlabel_tmp = unlabel_tmp.reshape((self.timestep, self.feature)).T
            for j in range(len(label_data)):
                label_tmp = label_data[j, :]
                label_tmp = label_tmp.reshape((self.timestep, self.feature)).T
                sum_tmp = 0
                for k in range(self.feature):
                    sum_tmp = sum_tmp + self.Dtw(unlabel_tmp[k,:], label_tmp[k,:])
                sum_tmp = sum_tmp / self.feature
                self.plabel[i, j] = sum_tmp
        # 距离排序，计算相似度得分
        index = np.argmax(label, 1)
        score = np.zeros([len(unlabel_data), self.output])
        for i in range(len(unlabel_data)):
            dict = {}
            for j in range(len(label_data)):
                dict[self.plabel[i, j]] = index[j]
            sorted_dis = np.sort(self.plabel[i, :], 1)
            for k in range(len(sorted_dis)):
                score[i, dict[sorted_dis[k]]] = score[i, dict[sorted_dis[k]]] + (1 / sorted_dis[k]) * ((len(label) - k) / len(label))
        acc = np.argmax(score, 1) - np.argmax(unlabel, 1)
        print("伪标签精度：", np.sum(acc == 0) / len(acc))
        fake_label = np.zeros_like(score)
        score = np.argmax(score, 1)
        for i in range(len(score)):
            fake_label[i, score[i]] = 1
        return fake_label

    def TrainRnn(self, train_data, train_label, test_data, test_label, target_data, target_label):
        with self.sess.as_default():
            with self.graph.as_default():
                # ---------------------模型构建-----------------------#
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
                max_test_acc, max_target_acc = 0., 0.
                for epoch in range(self.epochs):
                    state = np.random.get_state()
                    np.random.shuffle(train_data)
                    np.random.set_state(state)
                    np.random.shuffle(train_label)
                    step = 0
                    while step * self.batch_size < len(train_data):
                        batch_x = train_data[step * self.batch_size:(step + 1) * self.batch_size]
                        batch_y = train_label[step * self.batch_size:(step + 1) * self.batch_size]
                        self.sess.run(optimizer, feed_dict={X: batch_x, Y: batch_y})
                        step += 1
                    train_acc = self.sess.run(accuracy, feed_dict={X: train_data, Y: train_label})
                    test_acc = self.sess.run(accuracy, feed_dict={X: test_data, Y: test_label})
                    target_acc = self.sess.run(accuracy, feed_dict={X: target_data, Y: target_label})
                    max_target_acc = max(max_target_acc, target_acc)
                    if max_test_acc < test_acc:
                        max_test_acc = test_acc
                        self.train_res = self.sess.run(pred, feed_dict={X: train_data})
                        self.test_res = self.sess.run(pred, feed_dict={X: test_data})
                        self.target_res = self.sess.run(pred, feed_dict={X: target_data})

                    if (epoch + 1) % self.display == 0:
                        print('Epoch:{}'.format(epoch + 1), 'train acc:{:.4}'.format(train_acc),
                              'test acc:{:.4}'.format(test_acc), 'target acc:{:.4}'.format(target_acc))
                print('Train over! ', 'max test acc:{:.4}'.format(max_test_acc),
                      'max target acc:{:.4}'.format(max_target_acc))
        self.sess.close()
        return self.train_res, self.test_res, self.target_res

    def TrainMeta(self, train_data, train_label, test_data, test_label, target_data, target_label):
        with self.sess.as_default():
            with self.graph.as_default():
                # ---------------------模型构建-----------------------#
                X = tf.placeholder(tf.float32, shape=[None, self.output * 6])
                Y = tf.placeholder(tf.float32, shape=[None, self.output])
                W1 = tf.Variable(tf.random_normal([self.output * 6, self.output * 3]))
                B1 = tf.Variable(tf.random_normal([self.output * 3]))
                W2 = tf.Variable(tf.random_normal([self.output * 3, self.output]))
                B2 = tf.Variable(tf.random_normal([self.output]))
                logits, pred = self.Meta(X, [W1, W2], [B1, B2])
                accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1)), tf.float32))
                loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=logits))
                optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)
                # ---------------------训练模型-----------------------#
                self.sess.run(tf.global_variables_initializer())
                max_test_acc, max_target_acc = 0., 0.
                for epoch in range(self.epochs):
                    state = np.random.get_state()
                    np.random.shuffle(train_data)
                    np.random.set_state(state)
                    np.random.shuffle(train_label)
                    step = 0
                    while step * self.batch_size < len(train_data):
                        batch_x = train_data[step * self.batch_size:(step + 1) * self.batch_size]
                        batch_y = train_label[step * self.batch_size:(step + 1) * self.batch_size]
                        self.sess.run(optimizer, feed_dict={X: batch_x, Y: batch_y})
                        step += 1
                    train_acc = self.sess.run(accuracy, feed_dict={X: train_data, Y: train_label})
                    test_acc = self.sess.run(accuracy, feed_dict={X: test_data, Y: test_label})
                    target_acc = self.sess.run(accuracy, feed_dict={X: target_data, Y: target_label})
                    max_target_acc = max(max_target_acc, target_acc)

                    if max_test_acc < test_acc:
                        max_test_acc = test_acc
                        self.target_res = self.sess.run(pred, feed_dict={X: target_data})

                    if (epoch + 1) % self.display == 0:
                        print('Epoch:{}'.format(epoch + 1), 'train acc:{:.4}'.format(train_acc),
                              'test acc:{:.4}'.format(test_acc), 'target acc:{:.4}'.format(target_acc))
                print('Train over! ', 'max test acc:{:.4}'.format(max_test_acc),
                      'max target acc:{:.4}'.format(max_target_acc))
                self.sess.close()
                return self.target_res

    def TrainSom(self, source_data, source_label, target_data, target_label):

        N = source_data.shape[0]
        M = source_data.shape[1]
        size = math.ceil(np.sqrt(5 * np.sqrt(N))) # 经验公式
        max_iter = 500
        som = MiniSom(size, size, M, sigma=3, learning_rate=0.5, neighborhood_function='bubble')
        som.pca_weights_init(source_data)
        som.train_batch(source_data, max_iter, verbose=False)
        winmap = som.labels_map(source_data, source_label)
        # 分类函数
        def classify(som_, data_, winmap_):
            from numpy import sum as npsum
            default_class = npsum(list(winmap_.values())).most_common()[0][0]
            result = []
            for d in data_:
                win_position = som_.winner(d)
                if win_position in winmap_:
                    result.append(winmap_[win_position].most_common()[0][0])
                else:
                    result.append(default_class)
            return result
        y_pred = classify(som, target_data, winmap)
        print(y_pred)
        # 可视化
        label_name_map_number = {"up":0, "down":1, "walk":2, "jump":3}
        class_names = ["up", "down", "walk", "jump"]
        from matplotlib.gridspec import GridSpec
        plt.figure(figsize=(9, 9))
        the_grid = GridSpec(size, size)
        for position in winmap.keys():
            label_fracs = [winmap[position][label] for label in [0, 1, 2, 3]]
            plt.subplot(the_grid[position[1], position[0]], aspect=1)
            patches, texts = plt.pie(label_fracs)
            plt.text(position[0] / 100, position[1] / 100, str(len(list(winmap[position].elements()))),
                     color='black', fontdict={'weight': 'bold', 'size': 15},
                     va='center', ha='center')
        plt.legend(patches, class_names, loc='center right', bbox_to_anchor=(-1, 9), ncol=4)
        plt.show()

if __name__ == '__main__':
    # 数据导入
    data = np.genfromtxt('datasets/different_location_activity/420_gzy_P8A4.csv', dtype=float, delimiter=',',
                         encoding='utf-8')
    label = np.genfromtxt('datasets/different_location_activity/420_gzy_P8A4_Label.csv', dtype=float, delimiter=',',
                          encoding='utf-8')
    data2 = np.genfromtxt('datasets/different_location_activity/5L_P20_A3.csv', dtype=float, delimiter=',',
                          encoding='utf-8')
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
        prediction[:, i] = model.TrainClassifier(train, train_label, test, test_label)
    # print(prediction)
    # print('target confusion matrix:\n', confusion_matrix(np.argmax(test_label, 1), stats.mode(prediction, 1)[0]))