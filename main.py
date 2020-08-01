import pyfiglet
import argparse
from utils import dbase
from utils import config
from utils.videoProcessor import VideoProcessor

if __name__ == '__main__':
    ascii_banner = pyfiglet.figlet_format("F R     A P P", font="slant")
    print(ascii_banner)

    parser = argparse.ArgumentParser(description='Process some arguments.')
    parser.add_argument('--cdp', type=str, help='the path to config file')

    args = parser.parse_args()
    config_path = args.cdp

    cfg = config.readConfig(config_path)
    db_conn, db = dbase.dbConnect(cfg["host"], cfg["port"], cfg["name"], cfg["user"], cfg["password"])

    dbase.readDescriptors(db)

    video = VideoProcessor(0, cfg["face_detector"], cfg["face_det_model_path"],
                           cfg["face_recognition"], cfg["face_reco_model_path"], cfg["threshold"],
                           cfg["tracker"], cfg["tracker_model_path"],
                           cfg["face_antispoofing"], cfg["face_antispoofing_model_path"],
                           )

    video.capture()