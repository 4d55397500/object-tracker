object-tracker
---

![Alt Text](./sample.gif)

An intelligence system for tracking based on data from the [Muliple Object Tracking Benchmark](https://motchallenge.net/).

System consists of a `loader`, `trainer`, and `indexer`.


**sources** holds mp4 files. There are 3 sorts of sources:

 * *.mp4: *raw video*
 * *-det.mp4: *identified but not classified objects/people*
 * *-gt.mp4: *identified and classified objects/people*


**cache** holds compressed numpy array representations of frame sequences


**labels** holds frame sequence labels, including bounding boxes


### Download sources

Download mp4 video files from the [Muliple Object Tracking Benchmark](https://motchallenge.net/).

```
$ ./download_sources
```
### Train

```
$ python objtracker/trainer.py
```

### Describe

A script is provided to give a text description of the sequence of tracked objects or people coming in or out of the scene.

```
$ python objtracker/describe.py objtracker/labels/*-gt.txt

----- Frame 1 -----
Number of objects: 6
New object 1 appeared in this frame
New object 3 appeared in this frame
New object 2 appeared in this frame
New object 5 appeared in this frame
New object 4 appeared in this frame
New object 6 appeared in this frame
----- Frame 2 -----
Number of objects: 6
----- Frame 3 -----
Number of objects: 6
----- Frame 4 -----
Number of objects: 6
New object 7 appeared in this frame
Object 1 disappeared from this frame
....
```