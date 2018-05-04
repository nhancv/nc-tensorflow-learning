from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

# Define the data: define some inputs, x, and the expected output for each input, y_true
x = tf.constant(np.array([[0, 0, 1],
                          [0, 1, 1],
                          [1, 0, 1],
                          [1, 1, 1]]), dtype=tf.float32)
y_true = tf.constant(np.array([[0, 0, 1, 1]]).T, dtype=tf.float32)

# Define the model:
# - build a simple linear model, with 1 output

linear_model = tf.layers.Dense(units=1)
y_pred = linear_model(x)
# - loss
loss = tf.losses.mean_squared_error(labels=y_true, predictions=y_pred)
# - train flow
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)

# Training: evaluate the predictions
init = tf.global_variables_initializer()
summaries = tf.summary.merge_all()

# - start session
sess = tf.Session()
sess.run(init)

writer = tf.summary.FileWriter("logs", sess.graph)

for i in range(1000):
    _, loss_value = sess.run((train, loss))
    # print(loss_value)

# Print y_pred of input training data
print(sess.run(y_pred))

# TEST
print(sess.run(y_pred, feed_dict={x: np.array([[0, 0, 0],
                                               [0, 0, 1],
                                               [0, 1, 0],
                                               [0, 1, 1]])}))

print(sess.run(y_pred, feed_dict={x: np.array([[1, 0, 0],
                                               [1, 0, 1],
                                               [1, 1, 0],
                                               [1, 1, 1]])}))
# - close session
sess.close()