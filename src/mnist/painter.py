from Tkinter import *

from sys import platform as sys_pf

if sys_pf == 'darwin':
    import matplotlib

    matplotlib.use("TkAgg")

from PIL import Image, ImageTk
# from PIL import Image, ImageTk, ImageGrab  # For Windows & OSx
# https://pypi.org/project/pyscreenshot/
import pyscreenshot as ImageGrab  # For Linux
import numpy as np
import skimage.io as ski_io
# from package.MyEnumClass import MyEnumClass
from src.mnist import mnist

IMG_SIZE = 500
DEFAULT_COLOR = 'white'


class Paint(object):

    def __init__(self):
        self.root = Tk()
        self.root.resizable(False, False)
        # Canvas in left side
        self.c = Canvas(self.root, bg='black', width=IMG_SIZE, height=IMG_SIZE)
        self.c.grid(row=0, columnspan=6)

        # Info panel in right side
        self.infoP = PanedWindow(self.root, orient=VERTICAL, width=IMG_SIZE, height=IMG_SIZE)
        self.infoP.grid(row=0, column=6)

        self.eraser_button = Button(self.root, text='eraser', command=self.use_eraser)
        self.eraser_button.grid(row=0)
        self.choose_size_button = Scale(self.root, from_=20, to=30, orient=HORIZONTAL)
        self.choose_size_button.grid(row=1)

        self.predict_res = StringVar()
        self.predict_info = Entry(self.root, textvariable=self.predict_res, state=DISABLED, disabledforeground='black')
        self.predict_info.grid(row=3)
        self.predict_res.set('predict...')

        self.infoP.add(self.eraser_button)
        self.infoP.add(self.choose_size_button)
        self.infoP.add(self.predict_info)

        # Initialize variables
        self.old_x = None
        self.old_y = None
        self.c.bind('<B1-Motion>', self.paint)
        self.c.bind('<ButtonRelease-1>', self.reset)

        self.root.mainloop()

    def use_eraser(self):
        self.c.delete("all")
        self.predict_res.set('Predict...')

    def paint(self, event):
        if self.old_x and self.old_y:
            self.c.create_line(self.old_x, self.old_y, event.x, event.y,
                               width=self.choose_size_button.get(), fill=DEFAULT_COLOR,
                               capstyle=ROUND, smooth=TRUE, splinesteps=36)
        self.old_x = event.x
        self.old_y = event.y

    def reset(self, event):
        self.old_x, self.old_y = None, None
        self.predict()

    def predict(self):
        img = ImageGrab.grab(bbox=self._canvas()).convert('L')
        img.thumbnail((28, 28), Image.ANTIALIAS)
        img.save("painter.png")
        predict_input = np.array([np.array(ski_io.imread("painter.png", as_grey=True), dtype=np.float32)])
        predict_res = mnist.running(is_training=False, predict_input=predict_input)
        self.predict_res.set('Predict: {0}'.format(predict_res))

    def _canvas(self):
        x = self.c.winfo_rootx() + self.c.winfo_x()
        y = self.c.winfo_rooty() + self.c.winfo_y()
        x1 = x + self.c.winfo_width()
        y1 = y + self.c.winfo_height()
        padding = (self.c.winfo_width() - IMG_SIZE) / 2
        box = (x + padding, y + padding, x1 - padding, y1 - padding)
        return box


if __name__ == '__main__':
    ge = Paint()
