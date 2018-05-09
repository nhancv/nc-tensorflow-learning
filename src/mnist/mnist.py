# Combine:
# https://www.tensorflow.org/get_started/custom_estimators
# - https://github.com/tensorflow/models/blob/master/official/mnist/mnist.py
# https://www.tensorflow.org/tutorials/layers
# - https://github.com/tensorflow/tensorflow/blob/r1.8/tensorflow/examples/tutorials/layers/cnn_mnist.py
# @nhancv

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from src.mnist import dataset

import os
import numpy as np
import tensorflow as tf
import numbers

tf.logging.set_verbosity(tf.logging.INFO)
LEARNING_RATE = 1e-4


def past_stop_threshold(stop_threshold, eval_metric):
    """Return a boolean representing whether a model should be stopped.
    Args:
      stop_threshold: float, the threshold above which a model should stop
        training.
      eval_metric: float, the current value of the relevant metric to check.
    Returns:
      True if training should stop, False otherwise.
    Raises:
      ValueError: if either stop_threshold or eval_metric is not a number
    """
    if stop_threshold is None:
        return False

    if not isinstance(stop_threshold, numbers.Number):
        raise ValueError("Threshold for checking stop conditions must be a number.")
    if not isinstance(eval_metric, numbers.Number):
        raise ValueError("Eval metric being checked against stop conditions "
                         "must be a number.")

    if eval_metric >= stop_threshold:
        tf.logging.info(
            "Stop threshold of {} was passed with metric value {}.".format(
                stop_threshold, eval_metric))
        return True

    return False


def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    # Reshape X to 4-D tensor: [batch_size, width, height, channels]
    # MNIST images are 28x28 pixels, and have one color channel
    image = features
    if isinstance(image, dict):
        image = features['image']

    input_layer = tf.reshape(image, [-1, 28, 28, 1])

    # Convolutional Layer #1
    # Computes 32 features using a 5x5 filter with ReLU activation.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 28, 28, 1]
    # Output Tensor Shape: [batch_size, 28, 28, 32]
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #1
    # First max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 28, 28, 32]
    # Output Tensor Shape: [batch_size, 14, 14, 32]
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2
    # Computes 64 features using a 5x5 filter.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 14, 14, 32]
    # Output Tensor Shape: [batch_size, 14, 14, 64]
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #2
    # Second max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 14, 14, 64]
    # Output Tensor Shape: [batch_size, 7, 7, 64]
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Flatten tensor into a batch of vectors
    # Input Tensor Shape: [batch_size, 7, 7, 64]
    # Output Tensor Shape: [batch_size, 7 * 7 * 64]
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

    # Dense Layer
    # Densely connected layer with 1024 neurons
    # Input Tensor Shape: [batch_size, 7 * 7 * 64]
    # Output Tensor Shape: [batch_size, 1024]
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

    # Add dropout operation; 0.6 probability that element will be kept
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits layer
    # Input Tensor Shape: [batch_size, 1024]
    # Output Tensor Shape: [batch_size, 10]
    logits = tf.layers.dense(inputs=dropout, units=10)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, export_outputs={
            'classify': tf.estimator.export.PredictOutput(predictions)
        })

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def running(is_training=False, predict_input=np.load(os.path.join(os.path.dirname(__file__), 'images/examples.npy'))):
    # Load training and eval data
    # export_dir = "/tmp/mnist_saved_model"
    # model_dir = "/tmp/mnist_convnet_model"
    # data_dir = "/tmp/mnist_data"
    print(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'mnist_saved_model'))
    export_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'model/mnist_saved_model')
    model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'model/mnist_convnet_model')
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'model/mnist_data')

    epochs_between_eval = 15
    train_epochs = 10
    batch_size = 100
    stop_threshold = 0.9

    # Set up training and evaluation input functions.
    def train_input_fn():
        """Prepare data for training."""

        # When choosing shuffle buffer sizes, larger sizes result in better
        # randomness, while smaller sizes use less memory. MNIST is a small
        # enough dataset that we can easily shuffle the full epoch.
        ds = dataset.train(data_dir)
        ds = ds.cache().shuffle(buffer_size=50000).batch(batch_size)

        # Iterate through the dataset a set number (`epochs_between_evals`) of times
        # during each training session.
        ds = ds.repeat(epochs_between_eval)
        return ds

    def eval_input_fn():
        return dataset.test(data_dir).batch(batch_size).make_one_shot_iterator().get_next()

    # Create the Estimator
    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir=model_dir)

    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=100)

    # Train and evaluate model.
    if is_training:
        for _ in range(train_epochs):
            mnist_classifier.train(input_fn=train_input_fn, hooks=[logging_hook])
            eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
            print('\nEvaluation results:\n\t%s\n' % eval_results)

            if past_stop_threshold(stop_threshold, eval_results['accuracy']):
                break

        # Export the model
        if export_dir is not None:
            image = tf.placeholder(tf.float32, [None, 28, 28])
            input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn({
                'image': image,
            })
            mnist_classifier.export_savedmodel(export_dir, input_fn)

    print('Predict')
    """
    Input data: 28x28 grey format, float32 and background is black = 0.0
    """

    np.set_printoptions(precision=1, suppress=True)
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'image': predict_input},
        # x=np.load('images/examples.npy'),
        num_epochs=1,
        shuffle=False)

    predict_results = mnist_classifier.predict(input_fn=predict_input_fn)
    res = []
    for el in predict_results:
        probability = el['probabilities']
        res.append(probability.argmax(axis=0))
        print(probability)
    print(res)
    if is_training is False:
        return res


def main(_):
    running(is_training=True)


if __name__ == "__main__":
    tf.app.run()
