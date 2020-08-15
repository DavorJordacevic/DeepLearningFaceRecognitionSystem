import os
import cv2
import uuid
import time
import mtcnn
import numpy as np
import tensorflow as tf
from absl.flags import FLAGS
from absl import app, flags, logging
from scipy.special import softmax
from arcface_tf2.modules.models import ArcFaceModel
from arcface_tf2.modules.utils import set_memory_growth, load_yaml, l2_norm

class ImgProcessor:
    def __init__(self, cfg):

        self.detector_type = cfg["face_detector"]
        self.dnn_detector_path = cfg["face_det_model_path"]
        self.ssd_model_file = cfg["opencv_face_detector_uint8"]
        self.ssd_config_file = cfg["opencv_face_detector_pbtxt"]
        self.recognizer = cfg["face_recognition"]
        self.recognizer_path = cfg["face_reco_model_path"]
        self.threshold = cfg["threshold"]
        self.write = cfg["write_to_file"]
        self.experimental = True if cfg["experimental"]=="true" else False
        self.landmarks = []

        if self.detector_type == "MTCNN":
            if self.dnn_detector_path == "":
                print("[INFO] Loading MTCNN detection model...")
                self.detector = mtcnn.MTCNN()
                print("[INFO] MTCNN detection model loaded.")
                print("MTCNN version: ", mtcnn.__version__)

        if self.detector_type == "SSD":
            # load our serialized model from disk
            print("[INFO] Loading SSD detection model...")
            # Here we need to read our pre-trained neural net created using Tensorflow
            self.detector = cv2.dnn.readNetFromTensorflow(self.ssd_model_file, self.ssd_config_file)
            self.min_confidence = 0.5  # minimum probability to filter weak detections
            print("[INFO] SSD detection model loaded.")


        self.net1 = cv2.dnn.readNetFromONNX('antispoofing0.onnx')
        self.net2 = cv2.dnn.readNetFromONNX('antispoofing1.onnx')

        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'

        logger = tf.get_logger()
        logger.disabled = True
        logger.setLevel(logging.FATAL)
        set_memory_growth()

        self.cfg = load_yaml('./arcface_tf2/configs/arc_mbv2.yaml')

        self.model = ArcFaceModel(size=self.cfg['input_size'],
                             backbone_type=self.cfg['backbone_type'],
                             training=False)

        ckpt_path = tf.train.latest_checkpoint('./arcface_tf2/checkpoints/' + self.cfg['sub_name'])
        if ckpt_path is not None:
            print("[*] load ckpt from {}".format(ckpt_path))
            self.model.load_weights(ckpt_path)
        else:
            print("[*] Cannot find ckpt from {}.".format(ckpt_path))
            exit()

    def encode(self, img: np.array([])) -> np.array([]):
        img = cv2.resize(img, (self.cfg['input_size'], self.cfg['input_size']))
        img = img.astype(np.float32) / 255.
        if len(img.shape) == 3:
            img = np.expand_dims(img, 0)
        embeds = l2_norm(self.model(img))
        return embeds

    def rescale_img_percent(self, img, percent=50):
        """
        rescale_img_percent
        Function for rescaling the image by some percent
        :param img: numpy.array()
        :param percent: int
        :return: img: numpy.array()
        """

        width = int(img.shape[1] * percent / 100)
        height = int(img.shape[0] * percent / 100)
        dim = (width, height)
        return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    def rescale_img(self, img: np.array([]), width: int, height: int):
        """
        rescale_img
        Function for rescaling the image to the desired size
        :param img: numpy.array()
        :param width: int
        :param height: int
        :return: img: numpy.array()
        """

        return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

    def detect(self, img: np.array([])) -> np.array([]):
        """
        detect
        The actual function for performing detection based on configuration parameter.
        Crops all faces from the image and returns the numpy array containing all of them.
        :param img: numpy.array()
        :return: numpy.array()
        """

        faces_array = []
        if self.detector_type == "SSD":
            (h, w) = img.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
            self.detector.setInput(blob)
            # Runs forward pass to compute outputs of layers listed in outBlobNames.
            # start = time.time()
            detections = self.detector.forward()
            # print(time.time()-start)

            for i in range(0, detections.shape[2]):
                # extract the confidence (probability) associated with the prediction
                confidence = detections[0, 0, i, 2]

                # filter out weak detections by ensuring the `confidence` is
                # greater than the minimum confidence
                if confidence > self.min_confidence:
                    face = True
                    # compute the (x, y)-coordinates of the bounding box for the object
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    f = img[startX:endX, startY:endY]
                    faces_array = np.append(faces_array, f)

        if self.detector_type == "MTCNN":
            if self.dnn_detector_path == "":
                # start = time.time()
                faces = self.detector.detect_faces(img)
                # print(time.time()-start)
                for face in faces:
                    # get coordinates
                    x, y, width, height = face['box']
                    # what a disaster ...
                    # mtcnn predicts negative bounding boxes so we need to manually correct them
                    if x < 0:
                        x = 0
                    if y < 0:
                        y = 0
                    f = img[y:y+height, x:x+width]

                    if (self.experimental):
                        if (self.isAlive(f)['isAlive'] != True):
                            continue

                    right_eye_x, right_eye_y = face['keypoints']['right_eye'][0], face['keypoints']['right_eye'][1]
                    left_eye_x, left_eye_y = face['keypoints']['left_eye'][0], face['keypoints']['left_eye'][1]

                    delta_x = right_eye_x - left_eye_x
                    delta_y = right_eye_y - left_eye_y
                    angle = np.arctan(delta_y / delta_x)
                    angle = (angle * 180) / np.pi

                    h, w = f.shape[:2]
                    # Calculating a center point of the image
                    # Integer division "//"" ensures that we receive whole numbers
                    center = (w // 2, h // 2)
                    # Defining a matrix M and calling
                    # cv2.getRotationMatrix2D method
                    M = cv2.getRotationMatrix2D(center, (angle), 1.0)
                    # Applying the rotation to our image using the
                    aligned_face = cv2.warpAffine(f, M, (w, h))
                    #cv2.imshow('aligned_face', aligned_face)
                    #cv2.waitKey(0)
                    #cv2.destroyAllWindows()

                    faces_array.append(aligned_face)

        if self.write == "true":
            for face in faces_array:
                cv2.imwrite(str(uuid.uuid4())+'.jpg', face)

        return faces_array

    def isAlive(self, img: np.array([])) -> dict:
        """
        isAlive
        Function for performing prediction whether the face is real or fake.
        :param img: numpy.array()
        :return: dict
        """

        img = cv2.resize(img, (80, 80))

        # investigate why this wont work!
        #net1.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        #net2.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        #net1.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        #net2.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        #net1.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        #net2.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        blob1 = cv2.dnn.blobFromImage(img, 1, size=(80, 80), swapRB=False, crop=False)
        blob2 = cv2.dnn.blobFromImage(img, 1, size=(80, 80), swapRB=False, crop=False)

        # pass the blob through the network and obtain the predictions
        self.net1.setInput(blob1)
        self.net2.setInput(blob2)
        prediction = np.zeros((1, 3))

        # Runs forward pass to compute outputs of layers listed in outBlobNames.
        #start = time.time()
        prediction += softmax(self.net1.forward())
        prediction += softmax(self.net2.forward())
        #print(time.time()-start)

        label = np.argmax(prediction)
        value = prediction[0][label] / 2
        print(prediction / 2)
        if label == 1:
            return {"isAlive": True}
        else:
            return {"isAlive": False}
