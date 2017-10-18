# loader.py
"""
    Load data sources to cache
"""
import os
import numpy as np
import cv2


xpixels = 480
ypixels = 640
rgb = 3


class Loader(object):


    """ Load data """


    def __init__(self, name):
        self.name = name
        self.source_path = "sources/" + self.name + ".mp4"


    def load(self):
        """ Load source mp4 into cache if it does not yet exist """

        print("Caching from {}".format(self.source_path))
        name = self.source_path.split("/")[-1].split(".")[0]
        cache_path = "cache/" + name

        if os.path.isfile(cache_path + ".npz"):
            # already in cache
            print("{} already cached".format(name))
            pass
        else:
            arr = self.make_frames()
            print("Finished frame extraction for {}".format(name))
            np.savez_compressed("cache/" + name, arr=arr)
            print("Finished caching for {}".format(name))


    def make_frames(self):
        """ Return numpy array of frames """
        cap = cv2.VideoCapture(self.source_path)
        success = True
        i = 0
        images = []
        log_period = 100
        while success:
            success, image = cap.read()
            if image is None:
                break
            images.append(image)
            i += 1
            if i % log_period == 0:
                print("Finished extracting {} frames".format(i))
        arr = np.array(images).reshape((i, xpixels, ypixels, rgb))
        return arr



if __name__ == "__main__":
    pass
