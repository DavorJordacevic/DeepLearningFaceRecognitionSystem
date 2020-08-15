import os
import cv2
import json
import time
import flask
import mtcnn
import base64
import logging
import pyfiglet
import argparse
import numpy as np
import tensorflow as tf
from PIL import Image
from utils import dbase
from utils import config
from flask import Flask, request
from utils.imgProcessor import ImgProcessor
from utils.recognitionEngine import RecognitionEngine


# Create the application instance
app = Flask('FR APP')


def shutdown_server():
    """
    shutdown_server
    The actual function for turning off the Flask server.
    """
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()



@app.route('/shutdown', methods=['POST'])
def shutdown() -> dict:
    """
    shutdown
    The endpoint for turning off the Flask server.
    :return: dict
    """
    shutdown_server()
    return {'status': 'Server shut down.'}


# Create a URL route in our application for "/@app.route('/healtcheck')"
@app.route('/healtcheck')
def healtcheck() -> dict:
    """
    healtcheck
    The function for performing the healtcheck of the whole system.
    :return: dict
    """
    return {
        'DB_status': 'Active' if db is not None else 'Inactive',
        'Flask_version': flask.__version__,
        'Opencv_version': cv2.__version__,
        'Tensorflow_version': tf.__version__,
        'GPU_available': 'Yes' if gpus else 'No'
    }


# Create a URL route in our application for "/@app.route('/detect')"
# The purpose of this endpoint is to extract faces from the providied imagage
@app.route('/detect', methods=["POST"])
def detect() -> dict:
    """
    detect
    The function for performing detection based on configuration parameters.
    Crops all faces from the image and returns the list containing all of them in numpy array format.
    :param img: numpy.array()
    :return: dict
    """
    pil_image = Image.open(request.files['image']).convert('RGB')
    img = np.array(pil_image)
    faces = imgProcessor.detect(img)
    faces_b64 = []
    for face in faces:
        # convert image to butes (base64 encoding)
        faces_b64.append(str(base64.b64encode(face)))
    return {
        'Status':'SUCCESS',
        # Serialization
        'faces': json.dumps(faces_b64)
    }



# Create a URL route in our application for "/@app.route('/isalive')"
# The purpose of this endpoint is to classify if the provided face is real or fake
@app.route('/isalive', methods=["POST"])
def isAlive() -> dict:
    """
    isAlive
    Performs the prediction if the face is real or fake (printed and video).
    :return: dict
    """
    pil_image = Image.open(request.files['image']).convert('RGB')
    img = np.array(pil_image)
    return imgProcessor.isAlive(img)



# Create a URL route in our application for "/@app.route('/isalive')"
# The purpose of this endpoint is to classify if the provided face is real or fake
@app.route('/identification', methods=["POST"])
def predict_rest() -> dict:
    """
    predict_rest
    The actual function for performing the recognition.
    Extract all faces from the image, check if faces are real,
    extract face descriptors and search the database using HNSW
    and angular distance metric.
    :return: dict
    """
    pil_image = Image.open(request.files['image']).convert('RGB')
    img = np.array(pil_image)

    # detect faces
    start = time.time()
    faces = imgProcessor.detect(img)
    response = []

    for face in faces:
        # encode face
        descriptor = imgProcessor.encode(face)

        # find persoin id in the database
        personid = recEngine.identification(ids, descriptor, personsids)
        if (personid['personid'] is not None):
            # find name in the database
            person = dbase.findPersonByID(db, db_conn, personid['personid'])
            response.append(person)
        else:
            response.append(personid)

    logging.info('Identification time: ' + str(time.time()-start))
    return {
        'status': 'SUCCESS',
        'response': response
    }



# Create a URL route in our application for "/@app.route('/encode')"
# The purpose of this endpoint is to encode the face
@app.route('/encodeAndInsert', methods=["POST"])
def encodeAndInsert() -> dict:
    """
    encodeAndInsert
    The actual function for addding a new person into database.
    After person is added successfully, make base is performed.
    :return: dict
    """
    name = request.form.get('name')
    #pil_image = Image.open(request.files['image']).convert('RGB')
    uploaded_files = request.files.getlist("files")
    embeds = []
    for image in uploaded_files:
        pil_image = Image.open(image).convert('RGB')
        img = np.array(pil_image)
        # detect faace
        faces = imgProcessor.detect(img)
        if len(faces) != 1:
            return {'status': 'ERROR'}
        embeds.append(imgProcessor.encode(np.array(faces[0])))

    result = dbase.receiveDescriptors(db, db_conn, name, embeds)
    if result['status'] != 'SUCCESS':
        return {'status': 'ERROR'}

    ids, descriptors, personsids = dbase.readDescriptors(db)
    return recEngine.makeBase(descriptors)



if __name__ == '__main__':

    # set environment variables
    os.environ['FLASK_ENV'] = 'development'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # remove old log file
    os.remove("FRAPP.log")

    # turn off tensorflow logger
    logger = tf.get_logger()
    logger.disabled = True
    logger.setLevel(logging.FATAL)

    # open file for logging
    logging.basicConfig(filename='FRAPP.log', level=logging.DEBUG, format='%(asctime)s %(levelname)-8s %(message)s')

    # allow GPU memory grow
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            logging.info(str(e))
            print(e)

    logging.info('Allow GPU memory grow successful.')

    # Just some prints for health check (can be deleted with any effect on the following code)
    # print("OpenCV version: ", cv2.__version__)
    # print("TensorFlow version: ", tf.__version__, "  GPU avaiable: ", tf.config.list_physical_devices('GPU'))

    # parse arguements
    parser = argparse.ArgumentParser(description='Process some arguments.')
    parser.add_argument('--cdp', type=str, help='the path to config file')
    logging.info('Parsing arguments successful.')

    args = parser.parse_args()
    config_path = args.cdp

    cfg = config.readConfig(config_path)
    db_conn, db = dbase.dbConnect(cfg["host"], cfg["port"], cfg["name"], cfg["user"], cfg["password"])

    ids, descriptors, personsids = dbase.readDescriptors(db)
    logging.info('Read descriptors successful.')

    imgProcessor = ImgProcessor(cfg)
    recEngine = RecognitionEngine(cfg['threshold'])
    logging.info('Detection, anti-spoofing and recognition models initialized.')

    recEngine.makeBase(np.array(descriptors))
    logging.info('Make base successful.')

    # Run the flask rest api
    # This can be updated to use multiple threads or processors
    # In addition, some type of queue should be used
    # print starting text
    ascii_banner = pyfiglet.figlet_format("F R     A P P", font="slant")
    print(ascii_banner)

    logging.info('FR APP IS RUNNING.')
    logging.info('---------------'*4)
    # threaded=False, processes=3
    app.run(debug=False)