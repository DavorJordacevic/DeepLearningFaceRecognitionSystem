import json
import numpy as np
from json import JSONEncoder

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

class NumpyArrayDecoder():
    def default(self, encodedNumpyData):
        # Deserialization
        decodedArrays = json.loads(encodedNumpyData)
        return np.asarray(decodedArrays["faces"])
