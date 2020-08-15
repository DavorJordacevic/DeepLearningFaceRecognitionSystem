# Welcome! My name is Davor and this is my thesis about Face Recognition.

####Let's see what can Deep Learning achieve!

<hr>

#### Tech stack
![PYTHON](https://img.shields.io/badge/python-v3.7-blue)
![TENSORFLOW](https://img.shields.io/badge/TensorFlow-2-orange)
![OPENCV](https://img.shields.io/badge/OPENCV-4.4-green)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-12-blue)
![FLASK](https://img.shields.io/badge/Flask-1.1.2-lightgrey)
![N2](https://img.shields.io/badge/n2-0.1.6%20-grey)

<hr>

#### Docker setup


#### Manual setup

```
conda create --name tf_gpu tensorflow-gpu python=3.7
conda install -c conda-forge opencv=4.4.0
conda install -c anaconda psycopg2
conda install -c conda-forge pyfiglet
conda install -c conda-forge mtcnn
conda install -c anaconda pillow
conda install -c anaconda flask
pip install n2

wget https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/opencv_face_detector.pbtxt
wget https://github.com/spmallick/learnopencv/raw/master/AgeGender/opencv_face_detector_uint8.pb

git clone https://github.com/peteryuX/arcface-tf2.git
```

<hr>

#### License