# indexer.py
"""
 Uses the results of training
 and prediction to index frames
 and accompanying learned quantities
 for meaningful information retrieval

 Examples: number of people in a given time range
           motion of a given individual over frames

"""
from trainer import Trainer



class Indexer(object):

    def __init__(self, name):
        self.name = name
        self.trainer = Trainer(name)






if __name__ == "__main__":
    pass