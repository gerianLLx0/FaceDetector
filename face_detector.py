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
# import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import PySimpleGUI as sg
# import time

class Face():
    def __init__(self):
        self.id = -1
        self.distance = -1
        self.x = 0
        self.y = 0
        self.w = 0
        self.h = 0
        self.centre = 0

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
    # cv.imshow('Live', frame)
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
        cv.putText(frame, 'FPS: {:.2f}'.format(fps), (1, 16), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv.putText(frame, f'Number of Faces: {num_faces}', (1, 50), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg

def run_gui_opencv(detector, tm):
    # define the window layout
    layout = [[sg.Text('Face Detector', size=(40, 1), justification='left', font='Helvetica 25')],
              [sg.Image(filename='', key='image'), sg.Canvas(size=(300, 300), key='-CANVAS-')],
              [sg.Text(size=(40,2), key='-OUTPUT_MODE_1-')],
              [sg.Button('Presence Mode', size=(12, 2), font='Helvetica 14'),
               sg.Button('Distance Mode', size=(12, 2), font='Helvetica 14'),
               sg.Button('Advanced Mode', size=(12, 2), font='Helvetica 14'),
               sg.Button('Exit', size=(12, 2), font='Helvetica 14')]]

    # create the window and show it without the plot
    window = sg.Window('Face Detector', layout, location=(400, 200), finalize=True)
    canvas_elem = window['-CANVAS-']
    canvas = canvas_elem.TKCanvas
    
    # ---===--- Event LOOP Read and display frames, operate the GUI --- #
    cap = cv.VideoCapture(0)
    
    # initialise variables
    recording = False
    mode = ''
    num_faces_over_time = []
    dummy_distance = []
    # draw the initial plot in the window
    fig = Figure()
    ax = fig.add_subplot(111)
    fig_agg = draw_figure(canvas, fig)
    # window.Element('-CANVAS-').Update(visible = False)
    
    while True:
        event, values = window.read(timeout=20)
        if event == 'Exit' or event == sg.WIN_CLOSED:
            break
        elif event == 'Presence Mode':
            recording = True
            mode = 'P'
        elif event == 'Distance Mode':
            recording = True
            mode = 'D'
        elif event == 'Advanced Mode':
            recording = True
            mode = 'A'

        if recording:
            ret, frame = cap.read()
            
            # Perform operations on frame
            faces = process_frame(frame, detector, tm)
            num_faces = get_data(faces)
            num_faces_over_time.append(num_faces)
            dummy_distance.append(num_faces+5)
            
            # Show frame on gui
            imgbytes = cv.imencode('.png', frame)[1].tobytes()
            window['image'].update(data=imgbytes)
            # window.Element('-CANVAS-').Update(visible = True)
            
            # Show text data on gui
            if mode == 'P':
                window['-OUTPUT_MODE_1-'].update(f'Number of faces: {num_faces}')
                ax.cla()
                ax.set_xlabel("Time")
                ax.set_ylabel("Number of Faces")
                ax.grid()
                ax.plot(num_faces_over_time)
                fig_agg.draw()
            elif mode == 'D':
                window['-OUTPUT_MODE_1-'].update(f'Distance to nearest face: {num_faces}')
                ax.cla()
                ax.set_xlabel("Time")
                ax.set_ylabel("Distance to Nearest Face")
                ax.grid()
                ax.plot(dummy_distance)
                fig_agg.draw()
            elif mode == 'A':
                window['-OUTPUT_MODE_1-'].update(f'Sorted distances: {num_faces}')
                
    # Finish up by removing from the screen
    cap.release()
    # cv.destroyAllWindows()
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
    run_gui_opencv(detector, tm)
    # process_webcam(detector, tm)

if __name__ == "__main__":
    main()