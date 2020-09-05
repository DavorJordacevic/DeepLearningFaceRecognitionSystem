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
        post_files = []
        for filename in os.listdir(folder):
            file = os.path.join(folder, filename)
            print(file)
            post_files.append(('images', open(file, 'rb')))
        payload = {'name': str(folder).split('/')[-1]}
        headers = {}
        _ = requests.request("POST", url, headers=headers, data=payload, files=post_files)


if __name__ == '__main__':
    main()

