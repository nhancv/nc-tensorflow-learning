import tensorflow as tf

# enable tensorboard
en_tensorboard = False

# parameters for the net
syn0 = tf.Variable(tf.random_uniform(shape=[3, 1], minval=-1, maxval=1, name='syn0'))

# start session
sess = tf.Session()


def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def train():
    train_writer = None
    # placeholders
    x = tf.placeholder(tf.float32, [4, 3], name='x-inputs')
    y = tf.placeholder(tf.float32, [4, 1], name='y-inputs')
    tf.summary.histogram('x-inputs', x)
    tf.summary.histogram('y-input', y)

    # set up the model calculations
    l1 = tf.sigmoid(tf.matmul(x, syn0))
    tf.summary.histogram('y-predictions', l1)

    # cost function is avg error over training samples
    cost = tf.losses.mean_squared_error(labels=y, predictions=l1)
    tf.summary.histogram('cost', cost)

    # training step is gradient descent
    train_step = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

    # declare training data
    training_x = [[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]]
    training_y = [[0], [0], [1], [1]]

    # init session
    if en_tensorboard:
        train_writer = tf.summary.FileWriter('logs', sess.graph)
    init = tf.global_variables_initializer()
    sess.run(init)

    # training
    for i in range(10000):
        if en_tensorboard:
            merged = tf.summary.merge_all()
            summary, batch_loss, new_state, _ = sess.run([merged, cost, l1, train_step],
                                                         feed_dict={x: training_x, y: training_y})
            train_writer.add_summary(summary, i)
        else:
            sess.run(train_step,
                     feed_dict={x: training_x, y: training_y})


def test(inputs):
    # redefine the shape of the input to a single unit with 2 features
    x = tf.placeholder(tf.float32, [1, 3], name='x-inputs')

    # redefine the model in terms of that new input shape
    output = tf.sigmoid(tf.matmul(x, syn0))

    print(inputs, '1' if (sess.run(output, feed_dict={x: [inputs]})[0][0] > 0.5) else '0')


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
