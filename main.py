import os
import cv2
import mtcnn
import pyfiglet
import argparse
import tensorflow as tf
from utils import dbase
from utils import config
from utils.videoProcessor import VideoProcessor

if __name__ == '__main__':
    ascii_banner = pyfiglet.figlet_format("F R     A P P", font="slant")
    print(ascii_banner)

    #os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    #os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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

    dbase.readDescriptors(db)

    video = VideoProcessor(video_source, cfg)

    if cfg["write_to_file"] == "false":
        video.capture()
    else:
        video.capture_and_write()