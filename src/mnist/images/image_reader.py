import os
import numpy as np
from skimage import io


def load_image(filename):
    return io.imread(os.path.join('data', filename))


ex3 = load_image('example3.png')
io.imshow(ex3)
io.show()
# io.imsave('data/test.png', ex3)
