# trainer.py
"""

  Execute training on cached data

"""
from keras.models import Sequential
from keras import layers, models, applications
from keras.layers import Conv2D, MaxPooling2D, Flatten, BatchNormalization, Lambda, Dense
from keras import backend as K
import numpy as np

from loader import Loader


xpixels = 480
ypixels = 640
rgb = 3


class Trainer(object):


    def __init__(self):
        pass


    def train(self, name):
        """ Train on a model """
        ldr = Loader(name)
        ldr.load()
        arr = np.load("cache/" + name + ".npz")["arr"]
        print("Loaded cached array of shape {}".format(arr.shape))
        num_frames, num_x_pixels, num_y_pixels, rgb_ct = arr.shape

        # combine x and y pixel dimensions
        x = arr.reshape(arr.shape[:-3] + (-1, 3))

        # ignore rgb dimension
        x = x[:,:,0]

        print("Reshaped array to {}".format(x.shape))

        y = self.first_bounding_boxes(name)
        print("Bounding boxes: {}".format(y.shape))

        # train regression model image -> bounding box
        model = self.get_model1()
        model.fit(x, y)


    def get_model1(self):
        """ Per-frame feed-forward network mapping flattened pixels
        to the bounding boxes. """
        model = Sequential()
        model.add(Dense(200, input_dim=xpixels*ypixels, activation='relu'))
        model.add(Dense(4, activation='relu'))
        model.compile('adadelta', 'mse')
        return model


    def get_bounding_boxes(self, name):
        """
           The file format should be the same as the ground truth file,
            which is a CSV text-file containing one object instance per line. Each line must contain 10 values:

           <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
           The conf value contains the detection confidence in the ETH-Bahnhof-det.txt files. For the ground truth and results,
           it acts as a flag whether the entry is to be considered. A value of 0 means that this particular instance is ignored
           in the evaluation, while any other value can be used to mark it as active. The world coordinates x,y,z are ignored for the 2D
           challenge and can be filled with -1.
           Similarly, the bounding boxes are ignored for the 3D challenge. However, each line is still required to contain 10 values.

           """
        # create numpy array of shape (num bounding boxes, 6)
        # 2nd dimension holds the bounding box coordinates + confidence + frame
        flname = "labels/" + name + ".txt"
        rows = []
        with open(flname, 'r') as fl:
            for ln in fl:
                frame, id, bb_left, bb_top, bb_width, bb_height, conf, x, y, z = ln.strip().split(",")
                bb_box = [bb_left, bb_top, bb_width, bb_height]
                rows.append(np.array(bb_box + [conf, frame]).reshape(6,))
        rows = np.array(rows)
        return rows

    def first_bounding_boxes(self, name):
        """ Returns only the first bounding box in each frame. For easy training. """
        flname = "labels/" + name + ".txt"
        rows = []
        with open(flname, 'r') as fl:
            frame, last_frame = None, None
            for ln in fl:
                frame, id, bb_left, bb_top, bb_width, bb_height, conf, x, y, z = ln.strip().split(",")
                bb_box = [bb_left, bb_top, bb_width, bb_height]
                if frame != last_frame:
                    rows.append(np.array(bb_box).reshape(4, ))
                last_frame = frame
        rows = np.array(rows)
        return rows


if __name__ == "__main__":
    trainer = Trainer()
    name = "ETH-Bahnhof-det"
    trainer.train(name)
