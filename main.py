import os
import cv2
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


# Create a URL route in our application for "/@app.route('/healtcheck')"
@app.route('/detect', methods=["POST"])
def detect():
    pil_image = Image.open(request.files['image']).convert('RGB')
    img = np.array(pil_image)
    return imgProcessor.detect(img)


if __name__ == '__main__':
    ascii_banner = pyfiglet.figlet_format("F R     A P P", font="slant")
    print(ascii_banner)

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    print("OpenCV version: ", cv2.__version__)
    print("TensorFlow version: ", tf.__version__, "  GPU avaiable: ", tf.config.list_physical_devices('GPU'))

    parser = argparse.ArgumentParser(description='Process some arguments.')
    parser.add_argument('--cdp', type=str, help='the path to config file')
    parser.add_argument('--video_source', type=str, help='video source')

    args = parser.parse_args()
    config_path = args.cdp
    video_source = args.video_source if args.video_source!="0" else 0

    cfg = config.readConfig(config_path)
    db_conn, db = dbase.dbConnect(cfg["host"], cfg["port"], cfg["name"], cfg["user"], cfg["password"])

    # just a test for now
    dbase.readDescriptors(db)

    imgProcessor = ImgProcessor(cfg)

    '''
    video = VideoProcessor(video_source, cfg)

    if cfg["write_to_file"] == "false":
        video.capture()
    else:   
        video.capture_and_write()
    '''
    '''
    _pool = Pool(processes=12)  # this is important part- We
    try:
        app.run(use_reloader=False)
    except KeyboardInterrupt:
        _pool.close()
        _pool.join()
    '''
    app.run(debug=True)