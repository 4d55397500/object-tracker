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

    def __init__(self):
        pass

    def load(self, source_path):
        """ Load source mp4 into cache if it does not yet exist """

        name = source_path.split("/")[-1].split(".")[0]
        cache_path = "cache/" + name

        if os.path.isfile(cache_path):
            # already in cache
            pass
        else:
            arr = Loader.make_frames(source_path)
            print("Finished frame extraction for {}".format(name))
            np.savez_compressed("cache/" + name, arr=arr)
            print("Finished caching for {}".format(name))

    @staticmethod
    def make_frames(source_path):
        cap = cv2.VideoCapture(source_path)
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
