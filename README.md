object-tracker
---
Object tracking and persistence for learned information in Tensorflow.

System consists of a `loader`, `trainer`, and `indexer`.


**sources** holds mp4 files


**cache** holds compressed numpy array representations of frame sequences


**labels** holds frame sequence labels, including bounding boxes


#### Download mp4 sources
```
$ ./download_sources
```
#### Train

```
$ python objtracker/trainer.py
```