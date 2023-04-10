# -*- coding: utf-8 -*-
"""
DETAILS
author: gerianLL
date:
purpose:
summary:
requirements:
    1. opencv 4.7.0
    2. matplotlib 3.7.1
    3. PySimpleGUI 4.60.4
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
# from time import perf_counter, sleep

## Calibration Measurements
# w_mm, h_mm
FACE_SIZE = (130, 180)
# w_px, h_px, dist_mm
PT1 = (235, 300, 300)
PT2 = (125, 160, 600)

class Face():
    def __init__(self):
        self.id = -1
        self.distance = -1
        self.x = 0
        self.y = 0
        self.w = 0
        self.h = 0
        self.coeffs = self.get_coeffs()
    
    def show_data(self):
        print(f'Face {self.id}: x: {self.x} y: {self.y} w: {self.w}, h: {self.h}')
    
    def calc_dist(self):
        size = (self.w, self.h)
        dists = np.zeros(2)
        for i in range(2):
            A = self.coeffs[i, 0]
            B = self.coeffs[i, 1]
            dists[i] = A*(FACE_SIZE[i]/size[i]) - B
        self.distance = np.average(dists)

    def get_coeffs(self):
        # get constants from using w and h
        coeffs = np.eye(2)
        for i in range(2):
            h_1 = FACE_SIZE[i] / PT1[i] 
            h_2 = FACE_SIZE[i] / PT2[i]
            a = np.array([[h_1, -1], [h_2, -1]])
            b = np.array([PT1[2], PT2[2]])
            x = np.linalg.solve(a, b)
            check = np.allclose(np.dot(a, x), b)
            if check:
                coeffs[i,:] = x
            else:
                print('Could not solve matrix') 
        return coeffs

def get_faces(frame, detector, tm):
    """
    do all the face detection, data gathering and plotting
    DNN detector better than Haar cascade
    detector returns type tuple with (1, np.ndarray)
    for detector output format, 
    see link (https://docs.opencv.org/4.7.0/d0/dd4/tutorial_dnn_face.html)
    
    returns type: list
    """
    img_H = int(frame.shape[0])
    img_W = int(frame.shape[1])
    detector.setInputSize((img_W, img_H))
    # Get detections
    tm.start()
    faces = detector.detect(frame)
    # print(f'detector faces: {type(faces)}')
    tm.stop()
    # faces[1] contains the list of faces
    faces = faces[1]
    draw_on_frame(frame, faces, tm.getFPS())
    
    if faces is not None:
        # convert np.array to list
        faces = faces.tolist()
        return faces
    else:
        return []

def get_data(faces):
    face_data_list = []
    if faces:
        for idx, face in enumerate(faces):
            face_data = Face()
            face_data.x = round(face[0],0)
            face_data.y = round(face[1],0)
            face_data.w = round(face[2],0)
            face_data.h = round(face[3],0)
            # face_data.show_data()
            face_data_list.append(face_data)
            face_data.calc_dist()
        num_faces = len(faces)
        return num_faces, face_data_list          

def get_nearest_face(faces):
    """
    assume largest area = closest face
    """
    if faces:
        areas = []
        for face in faces:
            areas.append(face.w * face.h)
    else:
        print('No faces')
    

def draw_on_frame(frame, faces, fps, thickness=2):
    if faces is not None:
        for idx, face in enumerate(faces):
            # print('Face {}, top-left coordinates: ({:.0f}, {:.0f}), box width: {:.0f}, box height {:.0f}, score: {:.2f}'.format(idx, face[0], face[1], face[2], face[3], face[-1]))
            coords = face[:-1].astype(np.int32)
            cv.rectangle(frame, (coords[0], coords[1]), (coords[0]+coords[2], coords[1]+coords[3]), (0, 255, 0), thickness)
        cv.putText(frame, 'FPS: {:.2f}'.format(fps), (1, 16), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg

def run_gui_opencv(detector, tm):
    """
    Main GUI and event loop
    """
    # define the window layout
    layout = [[sg.Text('Face Detector', size=(40, 1), justification='left', font='Helvetica 25')],
              [sg.Text(size=(40,2), key='-OUTPUT_MODE_1-')],
              [sg.Button('Presence Mode', size=(12, 2), font='Helvetica 14'),
               sg.Button('Distance Mode', size=(12, 2), font='Helvetica 14'),
               sg.Button('Advanced Mode', size=(12, 2), font='Helvetica 14'),
               sg.Button('Exit', size=(12, 2), font='Helvetica 14')],
              [sg.Image(filename='', key='image'), sg.Canvas(size=(60, 60), key='-CANVAS-')]]

    # create the window and show it without the plot
    window = sg.Window('Face Detector', layout, location=(400, 200), finalize=True)
    canvas_elem = window['-CANVAS-']
    canvas = canvas_elem.TKCanvas
    
    ################################
    # Event Loop and GUI Operations
    ################################
    cap = cv.VideoCapture(0)
    
    # initialise variables
    recording = False
    mode = ''
    num_faces_over_time = []
    dummy_distance = []
    
    # draw and hide the initial plot in the window
    fig = Figure(figsize=(6,4))
    ax = fig.add_subplot(111)
    fig_agg = draw_figure(canvas, fig)
    canvas_elem.hide_row()
    
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
            faces = get_faces(frame, detector, tm)
            if faces:
                num_faces, face_data_list = get_data(faces)
                nearest = get_nearest_face(face_data_list)
                
                num_faces_over_time.append(num_faces)
            if num_faces:
                dummy_distance.append(num_faces+5)
            else:
                dummy_distance.append(0)
            # Show frame on gui
            imgbytes = cv.imencode('.png', frame)[1].tobytes()
            window['image'].update(data=imgbytes)
            
            # Show text data on gui
            canvas_elem.unhide_row()
            if mode == 'P':
                window['-OUTPUT_MODE_1-'].update(f'Number of faces: {num_faces}')
                ax.cla()
                ax.set_xlabel('Time')
                ax.set_ylabel('Number of Faces')
                ax.set_title('Number of Faces over Time')
                ax.grid()
                ax.plot(num_faces_over_time)
                fig_agg.draw()
            elif mode == 'D':
                window['-OUTPUT_MODE_1-'].update(f'Distance to nearest face: {num_faces}')
                ax.cla()
                ax.set_xlabel('Time')
                ax.set_ylabel('Distance')
                ax.set_title('Distance to Nearest Face over Time')
                ax.grid()
                ax.plot(dummy_distance)
                fig_agg.draw()
            elif mode == 'A':
                window['-OUTPUT_MODE_1-'].update(f'Sorted distances: {num_faces}')

    # Finish up by removing from the screen
    cap.release()
    window.close() 

def main():
    tm = cv.TickMeter()
    # Initialize detector with default values
    # (note: detector image size will be overwritten later)
    detector = cv.FaceDetectorYN.create(
        'Data/face_detection_yunet_2022mar.onnx', 
        '', 
        (0, 0),
        0.9,
        0.3,
        5000)
    run_gui_opencv(detector, tm)


if __name__ == "__main__":
    main()