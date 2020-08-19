import time
import numpy as np
from n2 import HnswIndex

class RecognitionEngine:
    def __init__(self, threshold):
        self.recognizer = None
        self.threshold = threshold

    def makeBase(self, descriptors: []) -> str:

        self.recognizer = HnswIndex(512, 'angular')

        for d in descriptors:
            self.recognizer.add_data(np.array(d))

        self.recognizer.build(m=5, max_m0=10, n_threads=4)
        self.recognizer.save('index.hnsw')
        return {'status': 'SUCCESS'}

    def identification(self, ids: [], descriptor: [], personids: []) -> dict:
        idx = self.recognizer.search_by_vector(np.array(descriptor).flatten(), 2, 1, include_distances=True)
        #print(idx[0][1])
        if (idx[0][1] < self.threshold):
            personID = personids[idx[0][0]]
        else:
            personID = None
        return {
            'status': 'SUCCESS',
            'personid': personID
        }