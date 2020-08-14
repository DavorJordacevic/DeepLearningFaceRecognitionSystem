import time
import numpy as np
from utils import dbase
from n2 import HnswIndex

class RecognitionEngine:
    def __init__(self):
        self.recognizer = None

    def makeBase(self, descriptors: []) -> str:

        self.recognizer = HnswIndex(512, 'angular')

        for d in descriptors:
            self.recognizer.add_data(np.array(d))

        self.recognizer.build(m=5, max_m0=10, n_threads=4)
        self.recognizer.save('index.hnsw')
        return {'status': 'SUCCESS'}

    def identification(self, ids: [], descriptor: [], personids: []) -> dict:
        idx = self.recognizer.search_by_vector(np.array(descriptor).flatten(), 1, include_distances=True)

        if (idx[0][1] < 0.5):
            personID = personids[idx[0][0]]
        else:
            personID = None
        return {
            'status': 'SUCCESS',
            'personid': personID
        }