import os
import uuid
import time
import requests
import argparse


def main():
    # parse arguements
    parser = argparse.ArgumentParser(description='Process some arguments.')
    parser.add_argument('--path', type=str, help='the path to folder')

    args = parser.parse_args()
    rootdir = args.path

    url = 'http://localhost:5000/encodeAndInsert'

    folders = []
    for subdir, dirs, files in os.walk(rootdir):
        folders.append(subdir)

    folders.remove(folders[0])

    for folder in folders:
        for subdir, dirs, files in os.walk(folder):
            for file in files:
                file = os.path.join(folder, subdir, file)
                print(file)
                payload = {'name': str(uuid.uuid4())}
                files = [
                    ('images', open(file, 'rb'))
                ]
                headers = {}
                response = requests.request("POST", url, headers=headers, data=payload, files=files)
                #time.sleep(2)

if __name__ == '__main__':
    main()

