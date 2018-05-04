import tensorflow as tf

# parameters for the net
syn0 = tf.Variable(tf.random_uniform(shape=[3, 1], minval=-1, maxval=1, name='syn0'))

# start session
sess = tf.Session()


def train():
    # placeholders
    x = tf.placeholder(tf.float32, [4, 3], name='x-inputs')
    y = tf.placeholder(tf.float32, [4, 1], name='y-inputs')

    # set up the model calculations
    l1 = tf.sigmoid(tf.matmul(x, syn0))

    # cost function is avg error over training samples
    cost = tf.losses.mean_squared_error(labels=y, predictions=l1)

    # training step is gradient descent
    train_step = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

    # declare training data
    training_x = [[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]]
    training_y = [[0], [0], [1], [1]]

    # init session
    init = tf.global_variables_initializer()
    sess.run(init)

    # training
    for i in range(10000):
        sess.run(train_step, feed_dict={x: training_x, y: training_y})


def test(inputs):
    # redefine the shape of the input to a single unit with 2 features
    xtest = tf.placeholder(tf.float32, [1, 3], name='x-inputs')

    # redefine the model in terms of that new input shape
    output = tf.sigmoid(tf.matmul(xtest, syn0))

    print(inputs, '1' if (sess.run(output, feed_dict={xtest: [inputs]})[0][0] > 0.5) else '0')


print('Start training')
train()

print('Start testing')
test([0, 0, 0])
test([0, 0, 1])
test([0, 1, 0])
test([0, 1, 1])
test([1, 0, 0])
test([1, 0, 1])
test([1, 1, 0])
test([1, 1, 1])

sess.close()
