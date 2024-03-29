import os
import cv2
import sys
import json
import time
import flask
import mtcnn
import base64
import logging
import pyfiglet
import argparse
import warnings
import numpy as np
import tensorflow as tf
from PIL import Image
from utils import dbase
from utils import config
from flask import Flask, request
from flask_cors import CORS, cross_origin
from utils.imgProcessor import ImgProcessor
from utils.recognitionEngine import RecognitionEngine


# disable warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

# Create the application instance
app = Flask('FR APP')
app.config['CORS_HEADERS'] = 'Content-Type'
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.png']
cors = CORS(app)

# initialize empty lists
ids, descriptors, persons_ids = [], [], []


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


'''
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
        # convert image to bytes (base64 encoding)
        faces_b64.append(str(base64.b64encode(face)))
    return {
        'Status': 'SUCCESS',
        # Serialization
        'faces': json.dumps(faces_b64)
    }



# Create a URL route in our application for "/@app.route('/isalive')"
# The purpose of this endpoint is to classify if the provided face is real or fake
@app.route('/isalive', methods=["POST"])
def is_alive() -> dict:
    """
    isAlive
    Performs the prediction if the face is real or fake.
    :return: dict
    """
    pil_image = Image.open(request.files['image']).convert('RGB')
    img = np.array(pil_image)
    return imgProcessor.is_alive(img)
'''


# Create a URL route in our application for "/@app.route('/isalive')"
# The purpose of this endpoint is to classify if the provided face is real or fake
@app.route('/identification', methods=["POST"])
@cross_origin()
def predict_rest() -> dict:
    """
    predict_rest
    The actual function for performing the recognition.
    Extract all faces from the image, check if faces are real,
    extract face descriptors and search the database using HNSW
    and angular distance metric.
    :return: dict
    """
    pil_image = Image.open(request.files['image'])
    img = np.float32(pil_image)

    # check image size (important for gpus with less vram)
    if img.shape[0]*img.shape[1] > 1920*1080:
        return{
            'status': 'ERROR',
            'response': 'Inappropriate image size (use smaller images).'
        }

    # detect faces
    faces = imgProcessor.detect(img)
    response = []

    global ids, descriptors, persons_ids

    start = time.time()
    if persons_ids:
        for face in faces:
            # encode face
            descriptor = imgProcessor.encode(face)
            # find personid in the database
            person_id = recEngine.identification(descriptor, persons_ids)
            if person_id['personid'] != 'Not recognized':
                # find name in the database
                person = dbase.find_person_by_id(db, person_id['personid'])
                response.append(person)
            else:
                response.append(person_id)
    else:
        return {
            'status': 'ERROR',
            'response': 'Empty database'
        }
    logging.info('Identification time: ' + str(time.time()-start))
    return {
        'status': 'SUCCESS',
        'response': response
    }


# Create a URL route in our application for "/@app.route('/encode')"
# The purpose of this endpoint is to encode the face
@app.route('/encodeAndInsert', methods=["POST"])
def encode_and_insert() -> dict:
    """
    encodeAndInsert
    The actual function for adding a new person into database.
    After person is added successfully, make base is performed.
    :return: dict
    """
    name = request.form.get('name')
    uploaded_files = request.files.getlist("images")

    embeds = []
    if not uploaded_files:
        return {"status": "ERROR"}
    for image in uploaded_files:
        pil_image = Image.open(image)
        img = np.float32(pil_image)

        # check image size (important for gpus with less vram)
        if img.shape[0] * img.shape[1] > 1920 * 1080:
            return {
                'status': 'ERROR',
                'response': 'Inappropriate image size (use smaller images).'
            }

        # detect face
        faces = imgProcessor.detect(img)

        if len(faces) != 1:
            return {
                'status': 'ERROR',
                'response': 'Images must contain only one face. Please try again.'
            }

        embeds.append(imgProcessor.encode(np.array(faces[0])))

    _ = dbase.receive_descriptors(db, db_conn, name, embeds)

    global ids, descriptors, persons_ids
    ids, descriptors, persons_ids = dbase.read_descriptors(db)
    return recEngine.make_base(descriptors)


if __name__ == '__main__':

    # remove old log file
    os.remove("FRAPP.log")

    # set environment variables
    os.environ['FLASK_ENV'] = 'development'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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

    # parse arguments
    parser = argparse.ArgumentParser(description='Process some arguments.')
    parser.add_argument('--cdp', type=str, help='the path to config file')
    logging.info('Parsing arguments successful.')

    args = parser.parse_args()
    config_path = args.cdp

    cfg = config.readConfig(config_path)
    db_conn, db = dbase.db_connect(cfg["host"], cfg["port"], cfg["name"], cfg["user"], cfg["password"])

    # initialize image processor (used for detection, anti-spoofing and vector extraction)
    imgProcessor = ImgProcessor(cfg)
    # initialize recognition engine
    recEngine = RecognitionEngine(cfg['threshold'])
    logging.info('All models initialized successfully.')

    try:
        # read ids, descriptors and person_ids from database
        ids, descriptors, persons_ids = dbase.read_descriptors(db)
        logging.info('Read descriptors successful.')
        recEngine.make_base(np.array(descriptors))
        logging.info('Make base successful.')
    except:
        logging.info('The database is empty.')

    # Run the flask rest api
    # This can be updated to use multiple threads or processors
    # In addition, some type of queue should be used
    # print starting text
    ascii_banner = pyfiglet.figlet_format("F R     A P P", font="slant")
    print(ascii_banner)

    logging.info('FR APP IS RUNNING.')
    logging.info('---------------' * 4)
    # threaded=False, processes=3
    app.run(debug=True, host='127.0.0.1', port=5000)
