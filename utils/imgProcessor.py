import cv2
import sys
import uuid
import mtcnn
import numpy as np
import tensorflow as tf
from scipy.special import softmax
sys.path.append('./retinaface_tf2/')
from arcface_tf2.modules.models import ArcFaceModel
from arcface_tf2.modules.utils import load_yaml, l2_norm
from retinaface_tf2.modules.models import RetinaFaceModel
from retinaface_tf2.modules.utils import pad_input_image, recover_pad_output


class ImgProcessor:
    def __init__(self, cfg):
        """
        __init__
        Initialize ImgProcessor with config file
        :param cfg: {}
        """
        self.detector_type = cfg["face_detector"]
        self.dnn_detector_path = cfg["face_det_model_path"]
        self.ssd_model_file = cfg["opencv_face_detector_uint8"]
        self.ssd_config_file = cfg["opencv_face_detector_pbtxt"]
        self.write = cfg["write_to_file"]
        self.experimental = True if cfg["experimental"] == "true" else False

        if self.detector_type == "MTCNN":
            if self.dnn_detector_path == "":
                self.detector = mtcnn.MTCNN()
                # print("[INFO] MTCNN detection model loaded.")

        if self.detector_type == "SSD":
            # load our serialized model from disk
            # Here we need to read our pre-trained neural net created using Tensorflow
            self.detector = cv2.dnn.readNetFromTensorflow(self.ssd_model_file, self.ssd_config_file)
            self.min_confidence = 0.5  # minimum probability to filter weak detections
            # print("[INFO] SSD detection model loaded.")

        if self.detector_type == "RetinaFace":
            # set config and checkpoints path
            self.face_det_cfg_path = load_yaml(cfg['face_det_cfg_path'])
            self.face_det_checkpoints_path = cfg['face_det_checkpoints_path']
            # load our serialized model from disk
            # Here we need to read our pre-trained neural net created using Tensorflow 2
            self.detector = RetinaFaceModel(self.face_det_cfg_path, training=False,
                                            iou_th=cfg["face_det_iou_th"], score_th=cfg["face_det_score_th"])

            self.face_det_down_scale_factor = cfg["face_det_down_scale_factor"]
            # load checkpoint
            checkpoint_dir = self.face_det_checkpoints_path + self.face_det_cfg_path['sub_name']
            checkpoint = tf.train.Checkpoint(model=self.detector)
            if tf.train.latest_checkpoint(checkpoint_dir):
                checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
                # print("[*] load ckpt from {}.".format(tf.train.latest_checkpoint(checkpoint_dir)))
            else:
                # print("[*] Cannot find ckpt from {}.".format(checkpoint_dir))
                exit()
            # print("[INFO] RetinaFace detection model loaded.")

        if self.experimental:
            self.net1 = cv2.dnn.readNetFromONNX('antispoofing0.onnx')
            self.net2 = cv2.dnn.readNetFromONNX('antispoofing1.onnx')

        # set config and checkpoints path
        self.face_reco_cfg_path = load_yaml(cfg['face_reco_cfg_path'])
        self.face_reco_checkpoints_path = cfg['face_reco_checkpoints_path']

        # initialize the ArcFace model
        self.model = ArcFaceModel(size=self.face_reco_cfg_path['input_size'],
                             backbone_type=self.face_reco_cfg_path['backbone_type'],
                             training=False)

        # load model weights
        ckpt_path = tf.train.latest_checkpoint(self.face_reco_checkpoints_path + self.face_reco_cfg_path['sub_name'])
        if ckpt_path is not None:
            # print("[INFO] Loading embeddings model.")
            self.model.load_weights(ckpt_path)
        else:
            # print("[ERROR] Cannot find embeddings model.")
            exit()

    def encode(self, img: np.array([])) -> np.array([]):
        """
        encode
        Function for extracting face embeddings using ArcFace model
        :param img: numpy.array()
        :return: img: numpy.array()
        """
        img = self.resize_img(np.array(img), self.face_reco_cfg_path['input_size'], self.face_reco_cfg_path['input_size'], 0, 0, cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.
        if len(img.shape) == 3:
            img = np.expand_dims(img, 0)
        # extract face embeddings and normalize
        embeds = l2_norm(self.model(img))
        return embeds

    def rescale_img_percent(self, img, percent=50) -> np.array([]):
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

    def resize_img(self, img: np.array([]), width: int, height: int, fx: int, fy: int, interpolation) -> np.array([]):
        """
        rescale_img
        Function for rescaling the image to the desired size
        :param img: numpy.array()
        :param width: int
        :param height: int
        :param fx: int
        :param fy: int
        :param interpolation: cv2.INTER_LINEAR
        :return: img: numpy.array()
        """
        return cv2.resize(img, (width, height), fx, fy, interpolation)

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
            detections = self.detector.forward()

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
            # we can also use other pre-trained model
            if self.dnn_detector_path == "":
                faces = self.detector.detect_faces(img)
                for face in faces:
                    # get coordinates
                    x, y, width, height = face['box']
                    # what a disaster ...
                    # mtcnn predicts negative bounding boxes so we need to manually correct them
                    x = 0 if x < 0 else x
                    y = 0 if y < 0 else y

                    f = img[y:y+height, x:x+width]
                    if self.experimental:
                        if not self.isAlive(f)['isAlive']:
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
                    m = cv2.getRotationMatrix2D(center, (angle), 1.0)
                    # Applying the rotation to our image using the
                    aligned_face = cv2.warpAffine(f, m, (w, h))
                    faces_array.append(aligned_face)

        if self.detector_type == "RetinaFace":
            img_height, img_width, _ = img.shape

            if self.face_det_down_scale_factor < 1.0:
                img = self.resize_img(img, 0, 0, self.face_det_down_scale_factor, self.face_det_down_scale_factor, cv2.INTER_LINEAR)

            # pad input image to avoid unmatched shape problem
            img, pad_params = pad_input_image(img, max_steps=max(self.face_det_cfg_path["steps"]))
            faces = self.detector(img[np.newaxis, ...]).numpy()
            # recover padding effect
            faces = recover_pad_output(faces, pad_params)

            for face in range(len(faces)):
                # get coordinates
                x1, y1, x2, y2 = int(faces[face][0] * img_width), int(faces[face][1] * img_height), \
                                 int(faces[face][2] * img_width), int(faces[face][3] * img_height)

                x1 = 0 if x1 < 0 else x1
                y1 = 0 if y1 < 0 else y1

                f = img[y1:y2, x1:x2]

                if self.experimental:
                    if not self.isAlive(f)['isAlive']:
                        continue

                # landmark
                if faces[face][14] > 0:
                    right_eye_x, right_eye_y = int(faces[face][4] * img_width), int(faces[face][5] * img_height)
                    left_eye_x, left_eye_y = int(faces[face][6] * img_width), int(faces[face][7] * img_height)

                    delta_x = right_eye_x - left_eye_x
                    delta_y = right_eye_y - left_eye_y
                    angle = np.arctan(delta_y / delta_x)
                    angle = (angle * 180) / np.pi

                    h, w = f.shape[:2]
                    # Calculating a center point of the image
                    # Integer division "//"" ensures that we receive whole numbers
                    center = (w // 2, h // 2)
                    # Defining a matrix M and calling
                    m = cv2.getRotationMatrix2D(center, (angle), 1.0)
                    # Applying the rotation to our image using the
                    aligned_face = cv2.warpAffine(f, m, (w, h))
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
        self.net1.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.net2.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        self.net1.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.net2.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        self.net1.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.net2.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        blob1 = cv2.dnn.blobFromImage(img, 1, size=(80, 80), swapRB=False, crop=False)
        blob2 = cv2.dnn.blobFromImage(img, 1, size=(80, 80), swapRB=False, crop=False)

        # pass the blob through the network and obtain the predictions
        self.net1.setInput(blob1)
        self.net2.setInput(blob2)
        prediction = np.zeros((1, 3))

        # Runs forward pass to compute outputs of layers listed in outBlobNames.
        prediction += softmax(self.net1.forward())
        prediction += softmax(self.net2.forward())

        label = np.argmax(prediction)
        value = prediction[0][label] / 2
        print(prediction / 2)
        if label == 1:
            return {"isAlive": True}
        else:
            return {"isAlive": False}
