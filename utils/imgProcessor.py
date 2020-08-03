import cv2
import mtcnn
import numpy as np

class ImgProcessor:
    def __init__(self, cfg):

        self.detector_type = cfg["face_detector"]
        self.dnn_detector_path = cfg["face_det_model_path"]
        self.haar_model_file = cfg["opencv_face_detector_uint8"]
        self.haar_config_file = cfg["opencv_face_detector_pbtxt"]
        self.recognizer = cfg["face_recognition"]
        self.recognizer_path = cfg["face_reco_model_path"]
        self.threshold = cfg["threshold"]

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

    def rescale_img(self, img, percent=50):
        width = int(img.shape[1] * percent / 100)
        height = int(img.shape[0] * percent / 100)
        dim = (width, height)
        return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    def detect(self, img):
        haveFaces = False
        faces_array = np.array([])
        if self.detector_type == "HAAR":
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
                    haveFaces = True
                    faces_array = np.append(faces_array, f)

        if self.detector_type == "MTCNN":
            if self.dnn_detector_path == "":
                faces = self.detector.detect_faces(img)
                for face in faces:
                    # get coordinates
                    x, y, width, height = face['box']
                    f = img[y:y+height, x:x+width]
                    haveFaces = True
                    faces_array = np.append(faces_array, f)

        for face in faces_array:
            cv2.imshow('Face', face)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return {"Detections": haveFaces}
        #return faces_array