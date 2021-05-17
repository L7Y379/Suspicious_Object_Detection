import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import gridspec
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.cluster import KMeans
import pandas as pd
import math
# Progressbar
# bar = progressbar.ProgressBar(widgets=['[', progressbar.Timer(), ']', progressbar.Bar(), '(', progressbar.ETA(), ')'])

# Get the MNIST data
#mnist = input_data.read_data_sets('./Data', one_hot=True)

# Parameters
# input_dim = 784
# n_l1 = 1000
# n_l2 = 1000
# z_dim = 2
input_dim = 270
n_l1 = 500
n_l2 = 500
z_dim = 2
batch_size = 100
n_epochs = 50
learning_rate = 0.001
beta1 = 0.9
results_path = 'Results/Adversarial_Autoencoder'

# Placeholders for input data and the targets
x_input = tf.placeholder(dtype=tf.float32, shape=[batch_size, input_dim], name='Input')
x_target = tf.placeholder(dtype=tf.float32, shape=[batch_size, input_dim], name='Target')
real_distribution = tf.placeholder(dtype=tf.float32, shape=[batch_size, z_dim], name='Real_distribution')
decoder_input = tf.placeholder(dtype=tf.float32, shape=[1, z_dim], name='Decoder_input')

# 欧氏距离计算
def distEclud(x, y):
    return np.sqrt(np.sum((x - y) ** 2))  # 计算欧氏距离

# 为给定数据集构建一个包含K个随机质心的集合
def randCent(dataSet, k):
    m, n = dataSet.shape
    centroids = np.zeros((k, n))
    for i in range(k):
        index = int(np.random.uniform(0, m))  #
        centroids[i, :] = dataSet[index, :]
    return centroids

# k均值聚类
def KMeans1(dataSet, k):
    m = np.shape(dataSet)[0]  # 行的数目
    # 第一列存样本属于哪一簇
    # 第二列存样本的到簇的中心点的误差
    clusterAssment = np.mat(np.zeros((m, 2)))
    clusterChange = True

    # 第1步 初始化centroids
    centroids = randCent(dataSet, k)
    while clusterChange:
        clusterChange = False

        # 遍历所有的样本（行数）
        for i in range(m):
            minDist = 100000.0
            minIndex = -1

            # 遍历所有的质心
            # 第2步 找出最近的质心
            for j in range(k):
                # 计算该样本到质心的欧式距离
                distance = distEclud(centroids[j, :], dataSet[i, :])
                if distance < minDist:
                    minDist = distance
                    minIndex = j
            # 第 3 步：更新每一行样本所属的簇
            if clusterAssment[i, 0] != minIndex:
                clusterChange = True
                clusterAssment[i, :] = minIndex, minDist ** 2
        # 第 4 步：更新质心
        for j in range(k):
            pointsInCluster = dataSet[np.nonzero(clusterAssment[:, 0].A == j)[0]]  # 获取簇类所有的点
            centroids[j, :] = np.mean(pointsInCluster, axis=0)  # 对矩阵的行求均值

    print("Congratulations,cluster complete!")
    return centroids, clusterAssment


def showCluster(dataSet, k, centroids, clusterAssment):
    m, n = dataSet.shape
    if n != 2:
        print("数据不是二维的")
        return 1

    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
    if k > len(mark):
        print("k值太大了")
        return 1

    # 绘制所有的样本
    for i in range(m):
        markIndex = int(clusterAssment[i, 0])
        plt.plot(dataSet[i, 0], dataSet[i, 1], mark[markIndex])

    mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
    # 绘制质心
    for i in range(k):
        plt.plot(centroids[i, 0], centroids[i, 1], mark[i])

    plt.show()

def file_array():#训练和测试文件名数组
    filepath = 'D:/my bad/Suspicious object detection/data/CSV/'
    filetype = '.csv'
    filenames = []
    trainfile = []
    testfile = []
    for j in ["0", "2Mhid"]:  # "1S", "2S"
        for i in [i for i in range(0, 30)]:
            fn = filepath + "zb-2.5-M/" + "zb-" + str(j) + "-" + str(i) + filetype
            filenames += [fn]
        np.random.shuffle(filenames)
        trainfile += filenames[:20]
        testfile += filenames[20:]
        filenames = []
    trainfile = np.array(trainfile)#20*2
    testfile = np.array(testfile)#10*2
    #print(testfile);
    return trainfile, testfile

def read_data(filenames):#读取文件中数据，并贴上标签
    i = 0
    feature = []
    label = []
    for filename in filenames:
        if os.path.exists(filename) == False:
            print(filename + " doesn't exit.")
            exit(1)
        csvdata = pd.read_csv(filename, header=None)
        csvdata = np.array(csvdata, dtype=np.float64)
        csvdata = csvdata[:, 0:270]
        idx = np.array([j for j in range(int(csvdata.shape[0] / 2)-120 ,
                                         int(csvdata.shape[0] / 2) +120, 2)])#取中心点处左右分布数据
        temp_feature = csvdata[idx,]
        # 贴标签
        temp_label = -1  # 初始化
        if ('-0-' in filename):
            temp_label = 0
        elif ('-1M-' in filename):
            temp_label = 1
        elif ('2M' in filename):
            temp_label = 2
        elif ('-3M-' in filename):
            temp_label = 3
        temp_label = np.tile(temp_label, (temp_feature.shape[0],))
        #temp_label = tf.Session().run(tf.one_hot(temp_label, N_CLASS))
        if i == 0:
            feature = temp_feature
            label = temp_label
            i = i + 1
        else:
            feature = np.concatenate((feature, temp_feature), axis=0)  # 拼接
            #label = np.concatenate((label, temp_label), axis=0)
    #data = np.concatenate((feature, label), axis=1)
    #np.random.shuffle(feature)
    return np.array(feature[:, :270]), np.array(feature[:, 270:])
    #return np.array(feature[:, 134:136]), np.array(feature[:, 134:136])

def form_results():
    """
    Forms folders for each run to store the tensorboard files, saved models and the log files.
    :return: three string pointing to tensorboard, saved models and log paths respectively.
    """
    folder_name = "/AAE_Kmeans"
    tensorboard_path = results_path + folder_name + '/Tensorboard'
    saved_model_path = results_path + folder_name + '/Saved_models/'
    log_path = results_path + folder_name + '/log'
    if not os.path.exists(results_path + folder_name):
        os.mkdir(results_path + folder_name)
        os.mkdir(tensorboard_path)
        os.mkdir(saved_model_path)
        os.mkdir(log_path)
    return tensorboard_path, saved_model_path, log_path


def generate_image_grid(sess, op):
    """
    Generates a grid of images by passing a set of numbers to the decoder and getting its output.
    :param sess: Tensorflow Session required to get the decoder output
    :param op: Operation that needs to be called inorder to get the decoder output
    :return: None, displays a matplotlib window with all the merged images.
    """
    x_points = np.arange(-10, 10, 2).astype(np.float32)
    y_points = np.arange(-10, 10, 2).astype(np.float32)

    nx, ny = len(x_points), len(y_points)#nx=14,ny=14
    plt.subplot()
    gs = gridspec.GridSpec(nx, ny, hspace=0.05, wspace=0.05)

    for i, g in enumerate(gs):#i=0~195,g=GridSpec(14, 14)[0:1, 0:1]~GridSpec(14, 14)[13:14, 13:14]
        z = np.concatenate(([x_points[int(i / ny)]], [y_points[int(i % nx)]]))#z=[-10. -10.]~[9.5 9.5]
        z = np.reshape(z, (1, 2))#z=[[-10. -10.]]~[[9.5 9.5]]
        x = sess.run(op, feed_dict={decoder_input: z})
        ax = plt.subplot(g)
        #img = np.array(x.tolist()).reshape(28, 28)
        img = np.array(x.tolist()).reshape(15, 18)
        ax.imshow(img, cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('auto')
    plt.show()

def mid_Kmeans(sess, op):
    trainfile_array, testfile_array = file_array()
    train_feature_all, train_label_all = read_data(trainfile_array)
    test_feature_all, test_label_all = read_data(testfile_array)
    #train_feature_all = train_feature_all.astype('float32')
    #test_feature_all = test_feature_all.astype('float32')
    train_feature_all = train_feature_all.astype('float32') / 72.0
    test_feature_all = test_feature_all.astype('float32') / 72.0
    train_feature_all=pow(train_feature_all, 2.0/3)
    test_feature_all = pow(test_feature_all, 2.0/3)
    n_batches1 = int(len(train_feature_all) / batch_size)
    for b in range(1, n_batches1+1):
        train_feature = train_feature_all[(b - 1) * 100:b * 100]
        train_feature_mid = sess.run(op, feed_dict={x_input:train_feature })
        if (b ==1):
            train_feature_mid_all = train_feature_mid
        else:
            train_feature_mid_all = np.vstack((train_feature_mid_all,train_feature_mid ))
    n_batches2 = int(len(test_feature_all) / batch_size)
    for b in range(1, n_batches2+1):
        test_feature = test_feature_all[(b - 1) * 100:b * 100]
        test_feature_mid = sess.run(op, feed_dict={x_input: test_feature})
        if (b ==1):
            test_feature_mid_all = test_feature_mid
        else:
            test_feature_mid_all = np.vstack((test_feature_mid_all, test_feature_mid))
    print(test_feature_mid_all)
    k = 2
    centroids, clusterAssment = KMeans1(train_feature_mid_all, k)
    kmeans = KMeans(n_clusters=2).fit(train_feature_mid_all)
    pred_train = kmeans.predict(train_feature_mid_all)

    print(pred_train)
    print(len(pred_train))
    a1=[0,0]
    a2=[0,0]
    for i in range(0,int(len(pred_train)/2)):
        if pred_train[i]==0:a1[0]=a1[0]+1
        if pred_train[i]==1:a1[1]=a1[1]+1
    for j in range(int(len(pred_train)/2),int(len(pred_train))):
        if pred_train[j] == 0: a2[0] = a2[0] + 1
        if pred_train[j] == 1: a2[1] = a2[1] + 1
    print(a1)
    print(a2)

    pred_test = kmeans.predict(test_feature_mid_all)
    print(pred_test)
    b=[0,0]
    b1 = [0, 0]
    b2 = [0, 0]
    for i in range(0, int(len(pred_test) / 2)):
        if pred_test[i] == 0:
            b1[0] = b1[0] + 1
            b[0]=b[0]+1
        if pred_test[i] == 1:
            b1[1] = b1[1] + 1
            b[1] = b[1] + 1
    for j in range(int(len(pred_test) / 2), int(len(pred_test))):
        if pred_test[j] == 0:
            b2[0] = b2[0] + 1
            b[0]=b[0]+1
        if pred_test[j] == 1:
            b2[1] = b2[1] + 1
            b[1] = b[1] + 1
    print(b)
    print(b1)
    print(b2)
    # print(pred2[100:])
    showCluster(train_feature_mid_all, k, centroids, clusterAssment)

def mid_Kmeans2(sess, op,train_feature_all,test_feature_all):
    n_batches1 = int(len(train_feature_all) / batch_size)
    for b in range(1, n_batches1 + 1):
        train_feature = train_feature_all[(b - 1) * 100:b * 100]
        train_feature_mid = sess.run(op, feed_dict={x_input: train_feature})
        if (b == 1):
            train_feature_mid_all = train_feature_mid
        else:
            train_feature_mid_all = np.vstack((train_feature_mid_all, train_feature_mid))
    n_batches2 = int(len(test_feature_all) / batch_size)
    for b in range(1, n_batches2 + 1):
        test_feature = test_feature_all[(b - 1) * 100:b * 100]
        test_feature_mid = sess.run(op, feed_dict={x_input: test_feature})
        if (b == 1):
            test_feature_mid_all = test_feature_mid
        else:
            test_feature_mid_all = np.vstack((test_feature_mid_all, test_feature_mid))
    k = 2
    print(test_feature_mid_all)
    centroids, clusterAssment = KMeans1(train_feature_mid_all, k)
    kmeans = KMeans(n_clusters=2).fit(train_feature_mid_all)
    pred_train = kmeans.predict(train_feature_mid_all)

    print(pred_train)
    print(len(pred_train))
    a1 = [0, 0]
    a2 = [0, 0]
    for i in range(0, int(len(pred_train) / 2)):
        if pred_train[i] == 0: a1[0] = a1[0] + 1
        if pred_train[i] == 1: a1[1] = a1[1] + 1
    for j in range(int(len(pred_train) / 2), int(len(pred_train))):
        if pred_train[j] == 0: a2[0] = a2[0] + 1
        if pred_train[j] == 1: a2[1] = a2[1] + 1
    print(a1)
    print(a2)

    pred_test = kmeans.predict(test_feature_mid_all)
    print(pred_test)
    b = [0, 0]
    b1 = [0, 0]
    b2 = [0, 0]
    for i in range(0, int(len(pred_test) / 2)):
        if pred_test[i] == 0:
            b1[0] = b1[0] + 1
            b[0] = b[0] + 1
        if pred_test[i] == 1:
            b1[1] = b1[1] + 1
            b[1] = b[1] + 1
    for j in range(int(len(pred_test) / 2), int(len(pred_test))):
        if pred_test[j] == 0:
            b2[0] = b2[0] + 1
            b[0] = b[0] + 1
        if pred_test[j] == 1:
            b2[1] = b2[1] + 1
            b[1] = b[1] + 1
    print(b)
    print(b1)
    print(b2)
    # print(pred2[100:])
    showCluster(train_feature_mid_all, k, centroids, clusterAssment)

def dense(x, n1, n2, name):
    """
    Used to create a dense layer.
    :param x: input tensor to the dense layer
    :param n1: no. of input neurons
    :param n2: no. of output neurons
    :param name: name of the entire dense layer.i.e, variable scope name.
    :return: tensor with shape [batch_size, n2]
    """
    with tf.variable_scope(name, reuse=None):
        weights = tf.get_variable("weights", shape=[n1, n2],
                                  initializer=tf.random_normal_initializer(mean=0., stddev=0.01))
        bias = tf.get_variable("bias", shape=[n2], initializer=tf.constant_initializer(0.0))
        out = tf.add(tf.matmul(x, weights), bias, name='matmul')
        return out


# The autoencoder network
def encoder(x, reuse=False):
    """
    Encode part of the autoencoder.
    :param x: input to the autoencoder
    :param reuse: True -> Reuse the encoder variables, False -> Create or search of variables before creating
    :return: tensor which is the hidden latent variable of the autoencoder.
    """
    if reuse:
        tf.get_variable_scope().reuse_variables()
    with tf.name_scope('Encoder'):
        e_dense_1 = tf.nn.relu(dense(x, input_dim, n_l1, 'e_dense_1'))
        e_dense_2 = tf.nn.relu(dense(e_dense_1, n_l1, n_l2, 'e_dense_2'))
        latent_variable = dense(e_dense_2, n_l2, z_dim, 'e_latent_variable')
        return latent_variable


def decoder(x, reuse=False):
    """
    Decoder part of the autoencoder.
    :param x: input to the decoder
    :param reuse: True -> Reuse the decoder variables, False -> Create or search of variables before creating
    :return: tensor which should ideally be the input given to the encoder.
    """
    if reuse:
        tf.get_variable_scope().reuse_variables()
    with tf.name_scope('Decoder'):
        d_dense_1 = tf.nn.relu(dense(x, z_dim, n_l2, 'd_dense_1'))
        d_dense_2 = tf.nn.relu(dense(d_dense_1, n_l2, n_l1, 'd_dense_2'))
        output = tf.nn.sigmoid(dense(d_dense_2, n_l1, input_dim, 'd_output'))
        #output = tf.nn.relu(dense(d_dense_2, n_l1, input_dim, 'd_output'))
        return output


def discriminator(x, reuse=False):
    """
    Discriminator that is used to match the posterior distribution with a given prior distribution.
    :param x: tensor of shape [batch_size, z_dim]
    :param reuse: True -> Reuse the discriminator variables,
                  False -> Create or search of variables before creating
    :return: tensor of shape [batch_size, 1]
    """
    if reuse:
        tf.get_variable_scope().reuse_variables()
    with tf.name_scope('Discriminator'):
        dc_den1 = tf.nn.relu(dense(x, z_dim, n_l1, name='dc_den1'))
        dc_den2 = tf.nn.relu(dense(dc_den1, n_l1, n_l2, name='dc_den2'))
        output = dense(dc_den2, n_l2, 1, name='dc_output')
        return output


def train(train_model=True):
    """
    Used to train the autoencoder by passing in the necessary inputs.
    :param train_model: True -> Train the models, False -> Load the latest trained models and show the image grid.
    :return: does not return anything
    """
    with tf.variable_scope(tf.get_variable_scope()):
        encoder_output = encoder(x_input)
        decoder_output = decoder(encoder_output)

    with tf.variable_scope(tf.get_variable_scope()):
        d_real = discriminator(real_distribution)
        d_fake = discriminator(encoder_output, reuse=True)

    with tf.variable_scope(tf.get_variable_scope()):
        decoder_image = decoder(decoder_input, reuse=True)

    # Autoencoder loss
    autoencoder_loss = tf.reduce_mean(tf.square(x_target - decoder_output))

    # Discrimminator Loss
    dc_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_real), logits=d_real))
    dc_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(d_fake), logits=d_fake))
    dc_loss = dc_loss_fake + dc_loss_real

    # Generator loss
    generator_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_fake), logits=d_fake))

    all_variables = tf.trainable_variables()
    dc_var = [var for var in all_variables if 'dc_' in var.name]
    en_var = [var for var in all_variables if 'e_' in var.name]

    # Optimizers
    autoencoder_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                                   beta1=beta1).minimize(autoencoder_loss)
    discriminator_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                                     beta1=beta1).minimize(dc_loss, var_list=dc_var)
    generator_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                                 beta1=beta1).minimize(generator_loss, var_list=en_var)

    init = tf.global_variables_initializer()

    # Reshape immages to display them
    # input_images = tf.reshape(x_input, [-1, 28, 28, 1])
    # generated_images = tf.reshape(decoder_output, [-1, 28, 28, 1])
    input_images = tf.reshape(x_input, [-1, 15, 18, 1])
    generated_images = tf.reshape(decoder_output, [-1, 15, 18, 1])
    # Tensorboard visualization
    tf.summary.scalar(name='Autoencoder Loss', tensor=autoencoder_loss)
    tf.summary.scalar(name='Discriminator Loss', tensor=dc_loss)
    tf.summary.scalar(name='Generator Loss', tensor=generator_loss)
    tf.summary.histogram(name='Encoder Distribution', values=encoder_output)
    tf.summary.histogram(name='Real Distribution', values=real_distribution)
    tf.summary.image(name='Input Images', tensor=input_images, max_outputs=10)
    tf.summary.image(name='Generated Images', tensor=generated_images, max_outputs=10)
    summary_op = tf.summary.merge_all()

    # Saving the models
    saver = tf.train.Saver()
    step = 0
    with tf.Session() as sess:
        if train_model:
            print('train_model=Ture')
            tensorboard_path, saved_model_path, log_path = form_results()
            #print(tensorboard_path)
            #print(log_path)

            sess.run(init)
            writer = tf.summary.FileWriter(logdir=tensorboard_path, graph=sess.graph)
            trainfile_array, testfile_array = file_array()
            train_feature_all, train_label_all = read_data(trainfile_array)
            test_feature_all, test_label_all = read_data(testfile_array)
            # train_feature_all = train_feature_all.astype('float32')
            # test_feature_all = test_feature_all.astype('float32')
            train_feature_all = train_feature_all.astype('float32') / 73.0
            test_feature_all = test_feature_all.astype('float32') / 73.0
            for i in range(n_epochs):


                n_batches = int(len(train_feature_all) / batch_size)
                print("------------------Epoch {}/{}------------------".format(i, n_epochs))
                #for b in range(1, n_batches + 1):
                for b in range(1, n_batches+1):
                    z_real_dist = np.random.randn(batch_size, z_dim) * 5.#标准正态分布N（0，1）

                    #batch_x, _ = mnist.train.next_batch(batch_size)
                    train_feature=train_feature_all[(b-1)*100:b*100]
                    #sess.run(autoencoder_optimizer, feed_dict={x_input: batch_x, x_target: batch_x})
                    sess.run(autoencoder_optimizer, feed_dict={x_input: train_feature, x_target: train_feature})
                    #sess.run(discriminator_optimizer,feed_dict={x_input: batch_x, x_target: batch_x, real_distribution: z_real_dist})
                    sess.run(discriminator_optimizer,feed_dict={x_input: train_feature, x_target: train_feature, real_distribution: z_real_dist})
                    #sess.run(generator_optimizer, feed_dict={x_input: batch_x, x_target: batch_x})
                    sess.run(generator_optimizer, feed_dict={x_input: train_feature, x_target: train_feature})
                    #if b % 50 == 0:
                    if b % 1 == 0:
                        # a_loss, d_loss, g_loss, summary = sess.run(
                        #     [autoencoder_loss, dc_loss, generator_loss, summary_op],
                        #     feed_dict={x_input: batch_x, x_target: batch_x,
                        #                real_distribution: z_real_dist})
                        a_loss, d_loss, g_loss, summary = sess.run(
                            [autoencoder_loss, dc_loss, generator_loss, summary_op],
                            feed_dict={x_input: train_feature, x_target: train_feature,
                                       real_distribution: z_real_dist})
                        writer.add_summary(summary, global_step=step)
                        print("Epoch: {}, iteration: {}".format(i, b))
                        print("Autoencoder Loss: {}".format(a_loss))
                        print("Discriminator Loss: {}".format(d_loss))
                        print("Generator Loss: {}".format(g_loss))
                        with open(log_path + '/log.txt', 'a') as log:
                            log.write("Epoch: {}, iteration: {}\n".format(i, b))
                            log.write("Autoencoder Loss: {}\n".format(a_loss))
                            log.write("Discriminator Loss: {}\n".format(d_loss))
                            log.write("Generator Loss: {}\n".format(g_loss))
                    step += 1

                saver.save(sess, save_path=saved_model_path, global_step=step)
            print(z_real_dist)
            mid_Kmeans2(sess, op=encoder_output,train_feature_all=train_feature_all,test_feature_all=test_feature_all)
        else:
            # Get the latest results folder
            print('train_model=False')
            all_results = 'AAE_Kmeans'
            saver.restore(sess, save_path=tf.train.latest_checkpoint(results_path + '/' + all_results + '/Saved_models/'))
            generate_image_grid(sess, op=decoder_image)
            mid_Kmeans(sess, op=encoder_output)

if __name__ == '__main__':
    #train(train_model=False)
    train(train_model=True)
