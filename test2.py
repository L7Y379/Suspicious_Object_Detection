import tensorflow as tf
import  numpy as np
temp_label = np.tile(0, (6,))
print(temp_label)

temp_label = tf.Session().run(tf.one_hot(temp_label, 3))
print(temp_label)