import cv2
import time
import mtcnn
import numpy as np
from collections import defaultdict

class VideoProcessor:
    def __init__(self, video_source, cfg):

        self.video_source = video_source
        self.detector_type = cfg["face_detector"]
        self.dnn_detector_path = cfg["face_det_model_path"]
        self.haar_model_file = cfg["opencv_face_detector_uint8"]
        self.haar_config_file = cfg["opencv_face_detector_pbtxt"]
        self.recognizer = cfg["face_recognition"]
        self.recognizer_path = cfg["face_reco_model_path"]
        self.threshold = cfg["threshold"]
        self.tracker = cfg["tracker"]
        self.tracker_path = cfg["tracker_model_path"]
        self.antispoofing = cfg["face_antispoofing"]
        self.antispoofing_path = cfg["face_antispoofing_model_path"]

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

    def rescale_frame(self, frame, percent=75):
        width = int(frame.shape[1] * percent / 100)
        height = int(frame.shape[0] * percent / 100)
        dim = (width, height)
        return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

    def capture(self):
        # Create a VideoCapture object and read from input file
        # If the input is taken from the camera, pass 0 instead of the video file name.
        self.cap = cv2.VideoCapture(self.video_source)

        # Check if video capture object is openanned successfully
        if (self.cap.isOpened() == False):
            print("Error opening video stream or file")

        tracker = cv2.TrackerGOTURN_create()

        # Read until video is completed
        while (self.cap.isOpened()):
            # Capture frame-by-frame
            ret, frame = self.cap.read()
            #frame = self.rescale_frame(frame, 25)

            if self.detector_type == "HAAR":
                (h, w) = frame.shape[:2]
                blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
                self.detector.setInput(blob)
                # Runs forward pass to compute outputs of layers listed in outBlobNames.
                detections = self.detector.forward()

                for i in range(0, detections.shape[2]):
                    # extract the confidence (probability) associated with the prediction
                    confidence = detections[0, 0, i, 2]

                    # filter out weak detections by ensuring the `confidence` is
                    # greater than the minimum confidence
                    if confidence > self.min_confidence:
                        # compute the (x, y)-coordinates of the bounding box for the object
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")

                        # draw the bounding box of the face along with the associated
                        # probability
                        text = "{:.2f}%".format(confidence * 100)
                        y = startY - 10 if startY - 10 > 10 else startY + 10
                        cv2.rectangle(frame, (startX, startY), (endX, endY),
                                      (0, 0, 255), 2)
                        cv2.putText(frame, text, (startX, y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

            if self.detector_type == "MTCNN":
                if self.dnn_detector_path == "":
                    faces = self.detector.detect_faces(frame)
                    for face in faces:
                        # get coordinates
                        x, y, width, height = face['box']

                        cv2.rectangle(frame, (x, y), (x + width, y + height), color=(0, 0, 255), thickness=2)

                        for key, value in face['keypoints'].items():
                            # create and draw dots
                            cv2.circle(frame, value, radius=2, color=(0, 0, 255), thickness=1)

                        text = "{:.2f}%".format(face['confidence'] * 100)
                        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

            # Initialize tracker with first frame and bounding box
            ok = tracker.init(frame, tuple(box))

            if ret == True:
                # Display the resulting frame
                cv2.imshow('Video stream', frame)

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

            #fps = self.cap.get(cv2.CAP_PROP_FPS)
            #print(fps)

        # When everything done, release the video capture object
        self.cap.release()
        # Closes all the frames
        cv2.destroyAllWindows()

    def capture_and_write(self):
        # Create a VideoCapture object and read from input file
        # If the input is taken from the camera, pass 0 instead of the video file name.
        self.cap = cv2.VideoCapture(self.video_source)

        # Check if video capture object is openanned successfully
        if (self.cap.isOpened() == False):
            print("Error opening video stream or file")

        # Default resolutions of the frame are obtained.The default resolutions are system dependent.
        # We convert the resolutions from float to integer.
        frame_width = int(self.cap.get(3))
        frame_height = int(self.cap.get(4))

        # Define the codec and create VideoWriter object.The output is stored in 'output_py.avi' file.
        out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (frame_width, frame_height))

        # Read until video is completed
        while (self.cap.isOpened()):
            # Capture frame-by-frame
            ret, frame = self.cap.read()
            if ret == True:
                # Display the resulting frame
                cv2.imshow('Video stream', frame)
                # Write the frame into the file 'output.avi'
                out.write(frame)

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

            #fps = self.cap.get(cv2.CAP_PROP_FPS)
            #print(fps)

        # When everything done, release the video capture object
        self.cap.release()
        out.release()
        # Closes all the frames
        cv2.destroyAllWindows()

    def dnn_detect(self):
        pass

    def mtcnn_detect(self):
        pass

    def haar_detect(self):
        pass