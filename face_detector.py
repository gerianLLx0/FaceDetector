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
import matplotlib.pyplot as plt
# import matplotlib.animation as animation
import PySimpleGUI as sg
import time

class Face():
    def __init__(self):
        self.id = -1
        self.distance = -1
        self.x = 0
        self.y = 0
        self.w = 0
        self.h = 0
        self.centre = 0
    
def process_webcam(detector, tm):
    """
    webcam video capture code from opencv docs
    https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html
    """
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    
    # Declare needed variables
    num_faces_over_time = []
    # fig, ax = plt.subplots()
    
    # Start looping video feed
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        # If frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        # Perform operations on frame
        faces = process_frame(frame, detector, tm)
        # Gathering, calculating and plotting the data from detected faces
        # Get data from faces (# of faces and distance to faces)
        num_faces = get_data(faces)
        num_faces_over_time.append(num_faces)
        # Plot live data
        # plot_num_faces(num_faces_over_time)

        # How to exit
        if cv.waitKey(1) == ord('q'):
            break
    # When done, release the capture
    cap.release()
    cv.destroyAllWindows()

def process_frame(frame, detector, tm):
    """
    do all the face detection, data gathering and plotting
    """
    # Timer
    # starttime = time.time()
    # while True:
    #     print("tick")
    #     time.sleep(60.0 - ((time.time() - starttime) % 60.0))
    
    img_H = int(frame.shape[0])
    img_W = int(frame.shape[1])
    detector.setInputSize((img_W, img_H))
    # Get detections
    tm.start()
    faces = detector.detect(frame)
    tm.stop()
    # Draw results on the input image
    draw_on_frame(frame, faces, tm.getFPS())
    cv.imshow('Live', frame)
    return faces

def get_data(faces):
    if faces[1] is not None:
        num_faces = len(faces[1])
        return num_faces    
       
def draw_on_frame(frame, faces, fps, thickness=2):
    if faces[1] is not None:
        num_faces = len(faces[1])
        for idx, face in enumerate(faces[1]):
            # print('Face {}, top-left coordinates: ({:.0f}, {:.0f}), box width: {:.0f}, box height {:.0f}, score: {:.2f}'.format(idx, face[0], face[1], face[2], face[3], face[-1]))
            coords = face[:-1].astype(np.int32)
            cv.rectangle(frame, (coords[0], coords[1]), (coords[0]+coords[2], coords[1]+coords[3]), (0, 255, 0), thickness)
            # cv.circle(frame, (coords[4], coords[5]), 2, (255, 0, 0), thickness)
            # cv.circle(frame, (coords[6], coords[7]), 2, (0, 0, 255), thickness)
            # cv.circle(frame, (coords[8], coords[9]), 2, (0, 255, 0), thickness)
            # cv.circle(frame, (coords[10], coords[11]), 2, (255, 0, 255), thickness)
            # cv.circle(frame, (coords[12], coords[13]), 2, (0, 255, 255), thickness)
        cv.putText(frame, 'FPS: {:.2f}'.format(fps), (1, 16), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv.putText(frame, f'Number of Faces: {num_faces}', (1, 50), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

def plot_num_faces(data):
    plt.figure(1)
    plt.clf()
    plt.plot(data)
    plt.pause(0.05)
    plt.show()

def run_gui():
    # Define the window's contents
    layout = [[sg.Text("What's your name?")],
              [sg.Input(key='-INPUT-')],
              [sg.Text(size=(40,1), key='-OUTPUT-')],
              [sg.Button('Ok'), sg.Button('Quit')]]
    
    # Create the window
    window = sg.Window('Face Detector', layout)
    
    # Display and interact with the Window using an Event Loop
    while True:
        event, values = window.read()
        # See if user wants to quit or window was closed
        if event == sg.WINDOW_CLOSED or event == 'Quit':
            break
        # Output a message to the window
        window['-OUTPUT-'].update('Hello ' + values['-INPUT-'] + "! Thanks for trying PySimpleGUI")
    
    # Finish up by removing from the screen
    window.close()
    
def main():
    tm = cv.TickMeter()
    # Initialize detector with default values
    # (note: detector image size will be overwritten later)
    detector = cv.FaceDetectorYN.create(
        "Data/face_detection_yunet_2022mar.onnx", 
        "", 
        (0, 0),
        0.9,
        0.3,
        5000)
    run_gui()
    # process_webcam(detector, tm)

if __name__ == "__main__":
    main()