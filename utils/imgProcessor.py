import cv2
import uuid
import time
import mtcnn
import numpy as np
from scipy.special import softmax

class ImgProcessor:
    def __init__(self, cfg):

        self.detector_type = cfg["face_detector"]
        self.dnn_detector_path = cfg["face_det_model_path"]
        self.haar_model_file = cfg["opencv_face_detector_uint8"]
        self.haar_config_file = cfg["opencv_face_detector_pbtxt"]
        self.recognizer = cfg["face_recognition"]
        self.recognizer_path = cfg["face_reco_model_path"]
        self.threshold = cfg["threshold"]
        self.write = cfg["write_to_file"]
        if self.detector_type == "MTCNN":
            if self.dnn_detector_path == "":
                print("[INFO] Loading MTCNN detection model...")
                self.detector = mtcnn.MTCNN()
                print("[INFO] MTCNN detection model loaded.")
                print("MTCNN version: ", mtcnn.__version__)

        if self.detector_type == "HAAR":
            # load our serialized model from disk
            print("[INFO] Loading Haar detection model...")
            # Here we need to read our pre-trained neural net created using Tensorflow
            self.detector = cv2.dnn.readNetFromTensorflow(self.haar_model_file, self.haar_config_file)
            self.min_confidence = 0.5  # minimum probability to filter weak detections
            print("[INFO] Haar detection model loaded.")

    def predict(self, img: np.array()) -> dict:
        pass

    def rescale_img_percent(self, img, percent=50):
        """ rescale_img_percent
        Function for rescaling the image by some percent
        :param img: numpy.array()
        :param percent: int
        :return: img: numpy.array()
        """

        width = int(img.shape[1] * percent / 100)
        height = int(img.shape[0] * percent / 100)
        dim = (width, height)
        return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    def rescale_img(self, img: np.array(), width: int, height: int):
        """ rescale_img
        Function for rescaling the image to the desired size
        :param img: numpy.array()
        :param width: int
        :param height: int
        :return: img: numpy.array()
        """

        return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

    def detect(self, img: np.array()) -> np.array():
        """ detect
        The actual function for performing detection based on configuration parameter.
        Crops all faces from the image and returns the numpy array containing all of them.
        :param img: numpy.array()
        :return: numpy.array()
        """

        faces_array = np.array([])
        if self.detector_type == "HAAR":
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
                    f = img[y:y+height, x:x+width]
                    faces_array = np.append(faces_array, f)

        if self.write == "false":
            for face in faces_array:
                cv2.imwrite(str(uuid.uuid4())+'.jpg', face)

        for face in faces_array:
            cv2.imshow('Face', face)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return faces_array

    def isAlive(self, img: np.array()) -> dict:
        """ isAlive
        Function for performing prediction whether the face is real or fake.
        :param img: numpy.array()
        :return: dict
        """

        img = cv2.resize(img, (80, 80))

        net1 = cv2.dnn.readNetFromONNX('antispoofing0.onnx')
        net2 = cv2.dnn.readNetFromONNX('antispoofing1.onnx')
        net3 = cv2.dnn.readNetFromONNX('feathernetB.onnx')

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
        net1.setInput(blob1)
        net2.setInput(blob2)
        prediction = np.zeros((1, 3))

        # Runs forward pass to compute outputs of layers listed in outBlobNames.
        #start = time.time()
        prediction += softmax(net1.forward())
        prediction += softmax(net2.forward())
        #print(time.time()-start)

        label = np.argmax(prediction)
        #value = prediction[0][label] / 2
        if label == 1:
            return '{"isAlive" : True}'
        else:
            return '{"isAlive": False}'
