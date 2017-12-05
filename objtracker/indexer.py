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

    def persist(self):
        if not self.trainer.is_trained:
            print("Trainer has not yet been executed. Cannot persist")
        else:
            # persist trained information in appropriate database
            model = self.trainer.load_model()
            # ...







if __name__ == "__main__":
    pass