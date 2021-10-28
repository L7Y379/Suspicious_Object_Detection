import tensorflow as tf
import numpy as np
 
# Visualize decoder setting
# Parameters

train = np.genfromtxt("source.csv", delimiter=',')
test = np.genfromtxt("target.csv", delimiter=',')
print(train.shape, test.shape)
learning_rate = 0.001
batch_size = 10
display_step = 1
examples_to_show = 10
 
# Network Parameters
n_input = 200  # 28x28 pix，即 784 Features
 
# tf Graph input (only pictures)
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_input])
 
# hidden layer settings
n_hidden_1 = 128 # 经过第一个隐藏层压缩至256个
n_hidden_2 = 64 # 经过第二个压缩至128个
#两个隐藏层的 weights 和 biases 的定义
weights = {
    'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'decoder1_h1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
    'decoder1_h2': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
    'decoder2_h1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
    'decoder2_h2': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'decoder1_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder1_b2': tf.Variable(tf.random_normal([n_input])),
    'decoder2_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder2_b2': tf.Variable(tf.random_normal([n_input])),
}
 
# Building the encoder
def encoder(x):
    # Encoder Hidden layer 使用的 Activation function 是 sigmoid #1
    layer_1 = tf.nn.leaky_relu(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.leaky_relu(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    return layer_2
 
 
# Building the decoder
def decoder1(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.leaky_relu(tf.add(tf.matmul(x, weights['decoder1_h1']),
                                   biases['decoder1_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.add(tf.matmul(layer_1, weights['decoder1_h2']),
                                   biases['decoder1_b2'])
    return layer_2

# Building the decoder
def decoder2(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.leaky_relu(tf.add(tf.matmul(x, weights['decoder2_h1']),
                                   biases['decoder2_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.add(tf.matmul(layer_1, weights['decoder2_h2']),
                                   biases['decoder2_b2'])
    return layer_2

# Construct model
encoder_op = encoder(X)
decoder_op1 = decoder1(encoder_op)
decoder_op2 = decoder2(encoder_op)
 
# Prediction
y_pred1 = decoder_op1
y_pred2 = decoder_op2
# Targets (Labels) are the input data.
y_true1 = X
y_true2 = Y
 
# Define loss and optimizer, minimize the squared error
#比较原始数据与还原后的拥有 784 Features 的数据进行 cost 的对比，
#根据 cost 来提升我的 Autoencoder 的准确率
loss1 = tf.reduce_mean(tf.pow(y_true1 - y_pred1, 2))#进行最小二乘法的计算(y_true - y_pred)^2
loss2 = tf.reduce_mean(tf.pow(y_true2 - y_pred2, 2))#进行最小二乘法的计算(y_true - y_pred)^2
decode1_summary = tf.summary.scalar('decode1_loss', loss1)
decode2_summary = tf.summary.scalar('decode2_loss', loss2)
#loss = tf.reduce_mean(tf.square(y_true - y_pred))
optimizer1 = tf.train.AdamOptimizer(learning_rate).minimize(loss1)
optimizer2 = tf.train.AdamOptimizer(learning_rate).minimize(loss2, var_list=[weights['decoder2_h1'],
                                                                             weights['decoder2_h2'],
                                                                             biases['decoder2_b1'],
                                                                             biases['decoder2_b2']])
 
 
# Launch the graph
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    decode1_merged = tf.summary.merge([decode1_summary])
    decode2_merged = tf.summary.merge([decode2_summary])
    writer = tf.summary.FileWriter('autoencoder_logs', sess.graph)
    training_epochs = 10000
    # Training cycle
    for epoch in range(training_epochs):#到好的的效果，我们应进行10 ~ 20个 Epoch 的训练
        # Loop over all batches
        _, c = sess.run([optimizer1, loss1], feed_dict={X: train})
        writer.add_summary(sess.run(decode1_merged, feed_dict={X: train}), epoch)
        print("Epoch:", '%04d' % (epoch+1),"cost=", "{:.4f}".format(c))
    print("Optimization Finished!")
    source_result = sess.run(y_pred1, feed_dict={X: train})
    # np.savetxt('source_result.csv', source_result, '%.4f', delimiter=',')
    for epoch in range(training_epochs):#到好的的效果，我们应进行10 ~ 20个 Epoch 的训练
        # Loop over all batches
        _, c = sess.run([optimizer2, loss2], feed_dict={X: test, Y: train})
        writer.add_summary(sess.run(decode2_merged, feed_dict={X: test, Y: train}), epoch)
        print("Epoch:", '%04d' % (epoch+1),"cost=", "{:.4f}".format(c))
    print("Optimization Finished!")
    target_result = sess.run(y_pred2, feed_dict={X: test})
    # np.savetxt('target_result.csv', target_result, '%.4f', delimiter=',')
