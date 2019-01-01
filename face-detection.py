# -*- coding: utf-8 -*-
"""
Created on Sun Dec 30 17:17:43 2018

@author: T.Mikon_N
"""
from threading import Thread
import multiprocessing
import time
import numpy as np
from multiprocessing import Queue, Pool
import cv2
DNN = "Caffe"
if DNN == "Caffe":
    modelFile = "D:\ssd_iter_140000_fp16.caffemodel"
    configFile = "D:\deploy.prototxt"
else:
    modelFile = "D:\opencv_face_detector_uint8.pb"
    configFile = "D:\opencv_face_detector.pbtxt"
class WebcamVideoStream:
    def __init__(self, src=0):
        # initialize the video camera stream and read the first frame
        # from the stream
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        
        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False

    def start(self):
        # start the thread to read frames from the video stream
        Thread(target=self.update,daemon=True).start()
        return self
 
    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return
                    
            # otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()
 
    def read(self):
        # return the frame most recently read
        return self.grabbed, self.frame
 
    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True

    def getWidth(self):
        # Get the width of the frames
        return int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH))

    def getHeight(self):
        # Get the height of the frames
        return int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def getFPS(self):
        # Get the frame rate of the frames
        return int(self.stream.get(cv2.CAP_PROP_FPS))

    def isOpen(self):
        # Get the frame rate of the frames
        return self.stream.isOpened()
    
def detect_objects(frame,net):
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
    net.setInput(blob)
    detections = net.forward()
    for i in range(0,detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.9:
            box = detections[0, 0, i, 3:7] * np.array([300, 300, 300, 300])
            (startX, startY, endX, endY) = box.astype("int")
            cv2.rectangle(frame, (startX, startY), (endX, endY),(0, 0, 255), 2)
        return frame

def worker(input_q, output_q):
    if DNN == "Caffe":
        net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
    else:
        net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)
    while True:
        frame = input_q.get()
        output_q.put(detect_objects(frame,net))
    
def Camera():
    input_q = Queue(maxsize=10)
    output_q = Queue(maxsize=10)
    pool = Pool(2, worker, (input_q,output_q))
    vs = WebcamVideoStream(src=0).start()
    while True:
        ret, frame = vs.read()
        if ret:
            frame = cv2.resize(frame, (300, 300))
            input_q.put(frame)
            output = output_q.get()
            cv2.imshow("Face Detection ,press q to stop program", output)
        else:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    pool.terminate()
    vs.stop()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    Camera()
        