import cv2
import matplotlib.pyplot as plt

class VideoProcessor:
  def __init__(self, path, detector, detector_path,
               recognizer, recognizer_path, threshold,
               tracker, tracker_path,
               antispoofing, antispoofing_path):

    self.path = path
    self.detector = detector
    self.detector_path = detector_path
    self.recognizer = recognizer
    self.recognizer_path = recognizer_path
    self.threshold = threshold
    self.tracker = tracker
    self.tracker_path = tracker_path
    self.antispoofing = antispoofing
    self.antispoofing_path = antispoofing_path

  def capture(self):
    # Create a VideoCapture object and read from input file
    # If the input is taken from the camera, pass 0 instead of the video file name.
    self.cap = cv2.VideoCapture(self.path)

    # Check if video capture object is openanned successfully
    if (self.cap.isOpened() == False):
        print("Error opening video stream or file")

    # Read until video is completed
    while (self.cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = self.cap.read()
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