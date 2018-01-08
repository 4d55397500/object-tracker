# pretrained.py
"""
  Run pretrained model

"""
import numpy as np
import tensorflow as tf
from PIL import Image


# may not be in repository (check .gitignore)
PRETRAINED_MODEL = "../faster_rcnn_resnet50_coco_2017_11_08/frozen_inference_graph.pb"
SAMPLE_IMAGE = "../sample_image.jpg"
LABELS_INDEX = "../mscoco_label_map.pbtxt"



def run_pretrained(img_path, labels_index):

    graph = tf.Graph()
    with graph.as_default():
        graph_def = tf.GraphDef()
        with tf.gfile.GFile(PRETRAINED_MODEL, 'rb') as fl:
            print("Loading frozen graph for pretrained model ...")
            frozen_graph = fl.read()
            graph_def.ParseFromString(frozen_graph)
            tf.import_graph_def(graph_def, name='')
            print("Finished loading frozen graph")

    with graph.as_default():
        with tf.Session(graph=graph) as sess:
            print("Running detection ...")
            image_tensor = graph.get_tensor_by_name('image_tensor:0')
            num_detections = graph.get_tensor_by_name('num_detections:0')
            detection_classes = graph.get_tensor_by_name('detection_classes:0')
            detection_boxes = graph.get_tensor_by_name('detection_boxes:0')
            detection_scores = graph.get_tensor_by_name('detection_scores:0')
            image = Image.open(img_path)
            (im_width, im_height) = image.size
            image_np = np.array(image.getdata()).reshape(
                (im_height, im_width, 3)).astype(np.uint8)
            # shape [1, None, None, 3]
            X = np.expand_dims(image_np, axis=0)
            n_detect, d_classes, boxes, scores = sess.run([num_detections, detection_classes, detection_boxes, detection_scores],
                                                  feed_dict={image_tensor: X})
            print("{} objects detected".format(int(n_detect[0])))
            for i, cls in enumerate(d_classes[0]):
                print("Detected {} at {} with confidence {}".format(labels_index[cls], boxes[0][i], scores[0][i]))


def get_labels_index():
    d = {}
    with open(LABELS_INDEX) as fl:
        current_id = -1
        for ln in fl:
            if "id: " in ln:
                id = int(ln.split(":")[-1].strip())
                current_id = id
            elif "display_name: " in ln:
                name = ln.split(":")[-1].strip().strip('"')
                d[current_id] = name
    return d





if __name__ == "__main__":
    labels_index = get_labels_index()
    run_pretrained(SAMPLE_IMAGE, labels_index)