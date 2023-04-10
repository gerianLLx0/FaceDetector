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

import numpy as np
import cv2 as cv
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import PySimpleGUI as sg

## Constants
PATH_DETECTION_MODEL = 'Data/face_detection_yunet_2022mar.onnx'

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
        self.size = 0
        self.coeffs = self.get_coeffs()
    
    def __repr__(self):
        return f'Face {self.id}: x: {self.x} y: {self.y} w: {self.w}, h: {self.h}\n'
    
    def show_data(self):
        print(f'Face {self.id}: x: {self.x} y: {self.y} w: {self.w}, h: {self.h}')
    
    def calc_size(self):
        self.size = self.w * self.h
        
    def calc_dist(self):
        size = (self.w, self.h)
        dists = np.zeros(2)
        for i in range(2):
            A = self.coeffs[i, 0]
            B = self.coeffs[i, 1]
            dists[i] = A*(FACE_SIZE[i]/size[i]) - B
        self.distance = int(np.average(dists))

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


def detect_faces(frame, detector, tm):
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

def convert_to_class_face(faces, counter):
    faces_data = []
    if faces:
        for idx, face in enumerate(faces):
            face_data = Face()
            face_data.id = counter
            face_data.x = round(face[0],0)
            face_data.y = round(face[1],0)
            face_data.w = round(face[2],0)
            face_data.h = round(face[3],0)
            # face_data.show_data()
            faces_data.append(face_data)
            face_data.calc_dist()
            counter += 1
        return faces_data, counter

def sort_faces(faces):
    """
    assume largest area = closest face
    """
    if faces:
        for face in faces:
            face.calc_size()
        faces.sort(key=lambda f: f.size, reverse=True)
        return faces
    else:
        print('No faces')
        return None 

def compare_to_existing_faces(existing_faces, seen_faces):
    new_faces = []
    if existing_faces:
        for s_face in seen_faces:
            # compare s_face to every existing face
            same_to_any = False
            for e_face in existing_faces:
                if is_same(e_face, s_face):
                    same_to_any = True
            # if new, add to new list
            if not same_to_any:
                new_faces.append(s_face)
    else:
        for s_face in seen_faces:
            new_faces.append(s_face)
    return new_faces

def is_same(e_face, n_face):
    """
    compare x, y, w & h if the same face within %
    """
    percent = 0.5
    same = np.zeros(4)
    if (e_face.x * (1-percent) <= n_face.x) and (n_face.x <= e_face.x * (1+percent)):
        same[0] = 1
    if (e_face.y * (1-percent) <= n_face.y) and (n_face.y <= e_face.y * (1+percent)):
        same[1] = 1  
    if (e_face.w * (1-percent) <= n_face.w) and (n_face.w <= e_face.w * (1+percent)):
        same[2] = 1
    if (e_face.h * (1-percent) <= n_face.h) and (n_face.h <= e_face.h * (1+percent)):
        same[3] = 1
    if same.all():
        return True
    else:
        return False

def draw_on_frame(frame, faces, fps, thickness=2):
    if faces is not None:
        for idx, face in enumerate(faces):
            # print('Face {}, top-left coordinates: ({:.0f}, {:.0f}), box width: {:.0f}, box height {:.0f}, score: {:.2f}'.format(idx, face[0], face[1], face[2], face[3], face[-1]))
            coords = face[:-1].astype(np.int32)
            cv.rectangle(frame, (coords[0], coords[1]), (coords[0]+coords[2], coords[1]+coords[3]), (0, 255, 0), thickness)
        cv.putText(frame, 'FPS: {:.2f}'.format(fps), (5, 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg

def init_gui():
    # define the window layout
    layout = [[sg.Text('Face Detector', size=(40, 1), font='Helvetica 25')],
              [sg.Button('Presence Mode', size=(12, 2), font='Helvetica 14'),
               sg.Button('Distance Mode', size=(12, 2), font='Helvetica 14'),
               sg.Button('Advanced Mode', size=(12, 2), font='Helvetica 14'),
               sg.Button('Exit', size=(12, 2), font='Helvetica 14'),
               sg.Multiline('', key='-MULTILINE-', size=(40, 5))],
              [sg.Canvas(key='-CANVAS-'),
               sg.Image(filename='', key='-IMAGE-')]]

    # create the window and show it without the plot
    window = sg.Window('Face Detector', layout, location=(200, 100), size=(1200,600), finalize=True)
    canvas_elem = window['-CANVAS-']
    canvas = canvas_elem.TKCanvas

    # draw initial plot in the window
    fig = Figure(figsize=(5,5))
    ax = fig.add_subplot(111)
    ax.set_axis_off()
    fig_agg = draw_figure(canvas, fig)
    
    return window, canvas_elem, ax, fig_agg

def get_detector():
    """
    Initialize detector with default values
    (note: detector image size will be overwritten later)
    for details, see link [https://docs.opencv.org/4.7.0/df/d20/classcv_1_1FaceDetectorYN.html]
    """
    detector = cv.FaceDetectorYN.create(
        PATH_DETECTION_MODEL, 
        '', 
        (0, 0),
        0.9,
        0.3,
        5000)
    return detector
    
def get_mode(mode, event):
    if event == 'Presence Mode':
        return 'P'
    elif event == 'Distance Mode':
        return 'D'
    elif event == 'Advanced Mode':
        return 'A'
    else:
        return mode
    
def activate_mode(mode, window, ax, fig_agg, num_faces_over_time, dists_over_time, faces):
    if mode == 'P':
        window['-MULTILINE-'].update(f'Number of faces: {num_faces_over_time[-1]}')
        ax.cla()
        ax.set_xlabel('Time')
        ax.set_ylabel('Number of Faces')
        ax.set_title('Number of Faces over Time')
        ax.grid()
        ax.plot(num_faces_over_time)
        fig_agg.draw()
    elif mode == 'D':
        window['-MULTILINE-'].update(f'Distance to nearest face: {dists_over_time[-1]} mm')
        ax.cla()
        ax.set_xlabel('Time')
        ax.set_ylabel('Distance (mm)')
        ax.set_title('Distance to Nearest Face over Time')
        ax.grid()
        ax.plot(dists_over_time)
        fig_agg.draw()
    elif mode == 'A':
        window['-MULTILINE-'].update('List of Faces:\n')
        ax.clear()
        ax.set_axis_off()
        fig_agg.draw()
        for face in faces:
            window['-MULTILINE-'].print(f'Face {face.id}: {face.distance}')

def run_gui():
    """
    Main GUI and event loop
    """
    window, canvas_elem, ax, fig_agg = init_gui()
    cap = cv.VideoCapture(0)
    detector = get_detector()
    tm = cv.TickMeter()
   
    # initialise variables
    num_faces_over_time = []
    dists_over_time = []
    mode = ''
    counter = 0
    existing_faces = []
    
    while True:
        event, values = window.read(timeout=20)
        if event == 'Exit' or event == sg.WIN_CLOSED:
            break
        mode = get_mode(mode, event)
        if mode:
            ret, frame = cap.read()
            
            # Perform operations on frame
            detected_faces = detect_faces(frame, detector, tm)
            if detected_faces:
                faces, counter = convert_to_class_face(detected_faces, counter)           
                faces = compare_to_existing_faces(existing_faces, faces)              
                sort_faces(faces)
                
                num_faces_over_time.append(len(faces))
                nearest_face = faces[0]
                dists_over_time.append(nearest_face.distance)
            else:
                num_faces_over_time.append(0)
                dists_over_time.append(0)
                
            # Show updates on gui
            imgbytes = cv.imencode('.png', frame)[1].tobytes()
            window['-IMAGE-'].update(data=imgbytes)
            activate_mode(mode, window, ax, fig_agg, num_faces_over_time, 
                          dists_over_time, faces)
            
    # Finish up by removing from the screen
    cap.release()
    window.close() 

def main():
    run_gui()

if __name__ == "__main__":
    main()