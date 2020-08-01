import json

def readConfig(path):
    with open(path, 'r') as config_file:
        cfg = json.load(config_file)
        return cfg