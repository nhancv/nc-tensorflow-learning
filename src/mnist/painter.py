from Tkinter import *


class Paint(object):
    DEFAULT_COLOR = 'white'

    def __init__(self):
        self.root = Tk()
        self.root.resizable(False, False)
        # Canvas in left side
        self.c = Canvas(self.root, bg='black', width=100, height=100)
        self.c.grid(row=0, columnspan=6)

        # Info panel in right side
        self.infoP = PanedWindow(self.root, orient=VERTICAL, width=100, height=100)
        self.infoP.grid(row=0, column=6)

        self.eraser_button = Button(self.root, text='eraser', command=self.use_eraser)
        self.eraser_button.grid(row=0)
        self.choose_size_button = Scale(self.root, from_=1, to=10, orient=HORIZONTAL)
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
        self.predict_res.set('predict...')

    def paint(self, event):
        if self.old_x and self.old_y:
            self.c.create_line(self.old_x, self.old_y, event.x, event.y,
                               width=self.choose_size_button.get(), fill=self.DEFAULT_COLOR,
                               capstyle=ROUND, smooth=TRUE, splinesteps=36)
        self.old_x = event.x
        self.old_y = event.y

    def reset(self, event):
        self.old_x, self.old_y = None, None


if __name__ == '__main__':
    ge = Paint()
