import numpy as np
import cv2



xpixels = 540
ypixels = 960
rgb = 3



def read_input(video_path, filepath):
        cap = cv2.VideoCapture(video_path)
        success = True
        i = 0
        images = []
        while success:
            success, image = cap.read()
            if image is None:
                break
            images.append(image)
            i += 1
        arr = np.array(images).reshape((i, xpixels, ypixels, rgb))




if __name__ == "__main__":
    video_path = "/Users/samirshah/Downloads/Venice-2.mp4"
    read_input(video_path)