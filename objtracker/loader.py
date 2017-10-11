# loader.py
"""
  Load data sources to appropriate
  data systems in GCP.
"""
import numpy as np
import yaml
from google.cloud import storage
import cv2


YAML = "./conf.yaml"

xpixels = 480
ypixels = 640
rgb = 3



class Loader:

    """ Load data """

    def __init__(self, config):
        self.config = config


    def load(self):
        """ Load videos into cloud storage as compressed
        numpy arrays. """
        data = self.config['data']
        local = self.config['local']
        for kind in ['sources', 'groundtruth', 'detections']:
            urlpath = data[kind]
            filename = local[kind]
            download_video(urlpath, filename)
            arr = get_frames_array(filename)
            print arr

        """
        bucket = self.config["gcs"]["bucket"]
        client = storage.Client()

        bkt = client.lookup_bucket(bucket)
        if bkt is None:
            client.create_bucket(bucket_name=bucket)
            bkt = client.get_bucket(bucket)       
        """



def download_video(urlpath, filename):
    import urllib
    print("Retrieving {}".format(urlpath))
    urllib.urlretrieve(urlpath, filename)
    print("Finished download")



def save_arr(arr, path):
    np.savez_compressed(path, key=arr)



def get_frames_array(filename):

        cap = cv2.VideoCapture(filename)
        success = True
        i = 0
        images = []
        while success:
            success, image = cap.read()
            if image is None:
                break
            images.append(image)
            i += 1
            print i
        arr = np.array(images).reshape((i, xpixels, ypixels, rgb))
        return arr



def read_conf():
    with open(YAML, 'r') as fl:
        config = yaml.load(fl)
        return config

def make_arr_unlabeled():
    filename = "./ETH-Bahnhof.mp4"
    arr = get_frames_array(filename)
    np.savez_compressed("./sources", raw=arr)


def make_arr_labeled():
    filename = "./ETH-Bahnhof-gt.mp4"
    arr = get_frames_array(filename)
    np.savez_compressed("./groundtruth", groundtruth=arr)


if __name__ == "__main__":
   make_arr_labeled()