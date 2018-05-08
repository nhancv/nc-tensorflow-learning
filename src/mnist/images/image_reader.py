import os
import numpy as np
from skimage import io


def load_image(filename, revert=False):
    im = np.array(io.imread(os.path.join('data', filename), as_grey=True), dtype=np.float32)
    if revert:
        im = np.vectorize(lambda t: 1.0 - t)(im)
    return im


def save_image(filename, arr):
    return io.imsave(os.path.join('data', filename), arr)


def show_image(filename):
    io.imshow(load_image(filename))
    io.show()


"""
https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.load.html

>>> np.save('/tmp/123', np.array([[1, 2, 3], [4, 5, 6]]))
>>> np.load('/tmp/123.npy')
array([[1, 2, 3],
       [4, 5, 6]])
"""
np.save('examples.npy', np.array([load_image('example0.png'),
                                  load_image('example1.png'),
                                  load_image('example3.png', True),
                                  load_image('example5.png', True),
                                  load_image('example7.png')], dtype=np.float32))
print(np.load('examples.npy'))

"""
Predict: saved_model_cli run --dir /tmp/mnist_saved_model/TIMESTAMP --tag_set serve --signature_def classify --inputs image=examples.npy
saved_model_cli run --dir /tmp/mnist_saved_model/1525618088 --tag_set serve --signature_def classify --inputs image=/Volumes/Data/Projects/nhancv/nc-tensorflow-learning/src/mnist/images/examples.npy

"""
