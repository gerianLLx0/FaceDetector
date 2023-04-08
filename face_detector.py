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

def capture_video():
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
        # Our operations on the frame come here
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # Display the resulting frame
        cv.imshow('frame', gray)
        if cv.waitKey(1) == ord('q'):
            break
    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()
    
    
def main():
    capture_video()

if __name__ == "__main__":
    main()