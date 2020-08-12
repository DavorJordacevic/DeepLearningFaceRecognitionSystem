import time
import numpy as np
from utils import dbase
from n2 import HnswIndex

class RecognitionEngine:
    def __init__(self):
        self.recognizer = HnswIndex(512, 'angular')

    def makeBase(self, descriptors: []) -> str:

        for d in descriptors:
            self.recognizer.add_data(np.array(d))

        self.recognizer.build(m=5, max_m0=10, n_threads=4)
        self.recognizer.save('index.hnsw')
        return 'SUCCESS'

    def identification(self, ids: [], descriptor: [], personids: []) -> dict:
        idx = self.recognizer.search_by_vector(np.array(descriptor).flatten(), 1)
        personID = personids[idx[0]]
        print(personID)
        return {
            'status'  : 'SUCCESS',
            'personid': personID
        }