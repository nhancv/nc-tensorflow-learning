import os
import numpy as np
from skimage import io


def load_image(filename):
    return io.imread(os.path.join('data', filename))


def save_image(filename, arr):
    return io.imsave(os.path.join('data', filename), arr)


def show_image(filename):
    io.imshow(load_image(filename))
    io.show()
