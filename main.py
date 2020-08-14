import os
import cv2
import json
import mtcnn
import pyfiglet
import argparse
import numpy as np
import tensorflow as tf
from PIL import Image
from utils import dbase
from utils import config
from flask import Flask, request
from multiprocessing import Pool
from utils.imgProcessor import ImgProcessor
from utils.recognitionEngine import RecognitionEngine

# Create the application instance
app = Flask('FR APP')

def shutdown_server():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()



@app.route('/shutdown', methods=['POST'])
def shutdown():
    shutdown_server()
    return 'Server shutting down...'


# Create a URL route in our application for "/@app.route('/healtcheck')"
@app.route('/healtcheck')
def healtcheck():
    return {
        'DB_status': 'Active' if db is not None else 'Inactive',
        'opencv_version': cv2.__version__,
        'tensorflow_version': tf.__version__,
        'GPU_available': 'Yes' if gpus else 'No'
    }


# Create a URL route in our application for "/@app.route('/detect')"
# The purpose of this endpoint is to extract faces from the providied imagage
@app.route('/detect', methods=["POST"])
def detect():
    pil_image = Image.open(request.files['image']).convert('RGB')
    img = np.array(pil_image)
    faces = imgProcessor.detect(img)
    return {
        'Status':'SUCCESS'
    }



# Create a URL route in our application for "/@app.route('/isalive')"
# The purpose of this endpoint is to classify if the provided face is real or fake
@app.route('/isalive', methods=["POST"])
def isAlive():
    pil_image = Image.open(request.files['image']).convert('RGB')
    img = np.array(pil_image)
    return imgProcessor.isAlive(img)



# Create a URL route in our application for "/@app.route('/isalive')"
# The purpose of this endpoint is to classify if the provided face is real or fake
@app.route('/identification', methods=["POST"])
def predict_rest():
    pil_image = Image.open(request.files['image']).convert('RGB')
    img = np.array(pil_image)

    faces = imgProcessor.detect(img)
    response = []

    for face in faces:
        descriptor = imgProcessor.encode(face)

        personid = recEngine.identification(ids, descriptor, personsids)
        if (personid['personid'] is not None):
            person = dbase.findPersonByID(db, db_conn, personid['personid'])
            response.append(person)
        else:
            response.append(personid)

    return {
        'status': 'SUCCESS',
        'response': response
    }



# Create a URL route in our application for "/@app.route('/encode')"
# The purpose of this endpoint is to encode the face
@app.route('/encodeAndInsert', methods=["POST"])
def encodeAndInsert():
    name = request.form.get('name')
    #pil_image = Image.open(request.files['image']).convert('RGB')
    uploaded_files = request.files.getlist("files")
    embeds = []
    for image in uploaded_files:
        pil_image = Image.open(image).convert('RGB')
        img = np.array(pil_image)
        faces = imgProcessor.detect(img)
        embeds.append(imgProcessor.encode(np.array(faces[0])))

    result = dbase.receiveDescriptors(db, db_conn, name, embeds)
    if result['status'] != 'SUCCESS':
        return {'status': 'ERROR'}

    ids, descriptors, personsids = dbase.readDescriptors(db)
    return recEngine.makeBase(descriptors)



if __name__ == '__main__':

    # allow GPU memory grow
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    # Just some prints for healt check (can be deleted with any effect on the following code)
    print("OpenCV version: ", cv2.__version__)
    print("TensorFlow version: ", tf.__version__, "  GPU avaiable: ", tf.config.list_physical_devices('GPU'))

    # parse arguements
    parser = argparse.ArgumentParser(description='Process some arguments.')
    parser.add_argument('--cdp', type=str, help='the path to config file')

    args = parser.parse_args()
    config_path = args.cdp

    cfg = config.readConfig(config_path)
    db_conn, db = dbase.dbConnect(cfg["host"], cfg["port"], cfg["name"], cfg["user"], cfg["password"])

    ids, descriptors, personsids = dbase.readDescriptors(db)

    imgProcessor = ImgProcessor(cfg)
    recEngine = RecognitionEngine()

    recEngine.makeBase(np.array(descriptors))

    # Run the flask rest api
    # This can be updated to use multiple threads or processors
    # In addition, some type of queue should be used
    # print starting text
    ascii_banner = pyfiglet.figlet_format("F R     A P P", font="slant")
    print(ascii_banner)
    app.run(debug=True)