# describe.py
"""

  Script describes the appearance and
  disappearance of tracked objects/people
  across frames from a label file ("*-gt.txt")

 Potential questions to be answered:
   What happened to object A?
   Which objects know one another?



"""
import sys


def describe(gt_label_filename):
    if not "gt" in gt_label_filename.split("-")[-1]:
        print("Error: file passed is not a ground truth label file.")
        sys.exit(0)
    else:
        print("Reading ground truth label file ...")
        with open(gt_label_filename, "r") as fl:
            current_objects = set()
            previous_objects = set()
            current_frame = 0
            for ln in fl:
                frame, id, bb_left, bb_top, bb_width, bb_height, conf, x, y, z = ln.strip().split(",")
                bb_box = [bb_left, bb_top, bb_width, bb_height]

                if frame != current_frame:
                    # new frame
                    print("----- Frame {} -----".format(current_frame))
                    print("Number of objects: {}".format(len(current_objects)))
                    new_objects = current_objects.difference(previous_objects)
                    disappeared_objects = previous_objects.difference(current_objects)
                    for n_obj in new_objects:
                        print("New object {} appeared in this frame".format(n_obj))
                    for d_obj in disappeared_objects:
                        print("Object {} disappeared from this frame".format(d_obj))
                    previous_objects = current_objects.copy()
                    current_objects = set()

                current_objects.add(id)
                current_frame = frame





if __name__ == "__main__":
    describe(sys.argv[1])
