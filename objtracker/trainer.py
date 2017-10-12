# trainer.py
"""

  Execute training on cached data

"""
from keras.models import Sequential
from keras import layers, models, applications
from keras.layers import Conv2D, MaxPooling2D, Flatten, BatchNormalization, Lambda

from keras import backend as K
import numpy as np

from loader import Loader


class Trainer(object):

    def __init__(self):
        pass

    def train(self, name):
        ldr = Loader(name)
        ldr.load()
        arr = np.load("cache/" + name + ".npz")["arr"]
        print("Loaded cached arr of shape {}".format(arr.shape))

        num_frames, num_x_pixels, num_y_pixels, rgb_ct = arr.shape

        # model ...

    def get_model(self):
        model = Sequential()
        model.add(Conv2D)

    def get_bounding_boxes(self, name):
        """
           The file format should be the same as the ground truth file,
            which is a CSV text-file containing one object instance per line. Each line must contain 10 values:

           <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
           The conf value contains the detection confidence in the det.txt files. For the ground truth and results,
           it acts as a flag whether the entry is to be considered. A value of 0 means that this particular instance is ignored
           in the evaluation, while any other value can be used to mark it as active. The world coordinates x,y,z are ignored for the 2D
           challenge and can be filled with -1.
           Similarly, the bounding boxes are ignored for the 3D challenge. However, each line is still required to contain 10 values.

           """
        pass


if __name__ == "__main__":
    trainer = Trainer()
    trainer.train("ETH-Bahnhof-det")