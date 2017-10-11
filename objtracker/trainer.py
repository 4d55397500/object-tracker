# trainer.py
"""

  Execute training on data sources

"""

import tensorflow as tf
from keras import layers, models, applications
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, BatchNormalization, Lambda
import cv2
from keras import backend as K

import numpy as np

NP_ARR = "./sources.npz"
GROUNDTRUTH_ARR = "./groundtruth.npz"



def train():

    raw_arr = np.load(NP_ARR)["sources"]
    print raw_arr.shape
    K.set_learning_phase(0)

    video = layers.Input(shape=(None, 480, 640, 3), name='video_input')
    cnn = applications.InceptionV3(
        weights='imagenet',
        include_top=False,
        pooling='avg'
    )
    cnn.trainable = False
    # wrap cnn into Lambda and pass it into TimeDistributed
    encoded_frame = layers.TimeDistributed(Lambda(lambda x: cnn(x)))(video)
    encoded_vid = layers.LSTM(256)(encoded_frame)
    outputs = layers.Dense(128, activation='relu')(encoded_vid)
    model = models.Model(inputs=[video], outputs=outputs)
    model.compile(optimizer='adam', loss='mean_squared_logarithmic_error')


    # Generate random targets
    y = np.random.random(size=(128,))
    y = np.reshape(y, (-1, 128))
    frame_sequence = raw_arr
    model.fit(x=frame_sequence, y=y, validation_split=0.0, shuffle=False, batch_size=1, epochs=5)


def get_bounding_boxes():
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
    pass