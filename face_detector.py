# -*- coding: utf-8 -*-
"""
DETAILS
author: gerianLL
date:
purpose:
summary:
requirements:
    1. opencv 4.7.0
"""

"""
ROADMAP
O1. Count number of faces
O2. Calculate distance of nearest face

1. access webcam
2. use opencv face detection
"""

import numpy as np
import cv2 as cv


def process_webcam(face_cascade):
    """
    webcam video capture code from opencv docs
    https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html
    """
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        
        # # Our operations on the frame come here
        # gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # # Display the resulting frame
        # cv.imshow('frame', gray)
        detectAndDisplay(frame, face_cascade)
        
        # how to exit
        if cv.waitKey(1) == ord('q'):
            break
    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()
    
    
def detectAndDisplay(frame, face_cascade):
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray)
    #-- Detect faces
    faces = face_cascade.detectMultiScale(frame_gray)
    for (x,y,w,h) in faces:
        center = (x + w//2, y + h//2)
        frame = cv.ellipse(frame, center, (w//2, h//2), 0, 0, 360, (255, 0, 255), 4)
        # faceROI = frame_gray[y:y+h,x:x+w]
        # #-- In each face, detect eyes
        # eyes = eyes_cascade.detectMultiScale(faceROI)
        # for (x2,y2,w2,h2) in eyes:
        #     eye_center = (x + x2 + w2//2, y + y2 + h2//2)
        #     radius = int(round((w2 + h2)*0.25))
        #     frame = cv.circle(frame, eye_center, radius, (255, 0, 0 ), 4)
    cv.imshow('Capture - Face detection', frame)
    
    
def main():
    # parser = argparse.ArgumentParser(description='Code for Cascade Classifier tutorial.')
    # parser.add_argument('--face_cascade', help='Path to face cascade.', default='data/haarcascades/haarcascade_frontalface_alt.xml')
    # parser.add_argument('--eyes_cascade', help='Path to eyes cascade.', default='data/haarcascades/haarcascade_eye_tree_eyeglasses.xml')
    # parser.add_argument('--camera', help='Camera divide number.', type=int, default=0)
    # args = parser.parse_args()
    # face_cascade_name = args.face_cascade
    # eyes_cascade_name = args.eyes_cascade
    face_cascade_name = 'Data/haarcascade_frontalface_alt.xml'
    face_cascade = cv.CascadeClassifier()
    face_cascade.load(face_cascade_name)
    # #-- 1. Load the cascades
    # if not face_cascade.load(cv.samples.findFile(face_cascade_name)):
    #     print('--(!)Error loading face cascade')
    #     exit(0)

    process_webcam(face_cascade)

if __name__ == "__main__":
    main()