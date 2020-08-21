import numpy as np
from n2 import HnswIndex


class RecognitionEngine:
    def __init__(self, threshold):
        """
        __init__
        :param threshold: float
        """
        self.recognizer = None
        self.threshold = threshold

    def makeBase(self, descriptors: []) -> str:
        """
        makeBase
        :param threshold: float
        :return: img: numpy.array()
        """
        # initialize recognizer using empty ann and set angsular distance
        self.recognizer = HnswIndex(512, 'angular')

        # add vectors to the ann
        for d in descriptors:
            self.recognizer.add_data(np.array(d))

        # build ann
        self.recognizer.build(m=5, max_m0=10, n_threads=4)
        # dump to file (files is not used for now, but can be used to load the old one in case of some error)
        self.recognizer.save('index.hnsw')
        return {'status': 'SUCCESS'}

    def identification(self, descriptor: [], personids: []) -> dict:
        """
        makeBase
        :param threshold: float
        :return: img: numpy.array()
        """
        idx = self.recognizer.search_by_vector(np.array(descriptor).flatten(), 2, 1, include_distances=True)
        #print(idx[0][1])
        if (idx[0][1] < self.threshold):
            personID = personids[idx[0][0]]
        else:
            # personID = None
            personID = "Not recognized"
        return {
            'status': 'SUCCESS',
            'personid': personID
        }