import matplotlib.pyplot as plt
import mediapipe as mp
import pandas as pd
import cv2
import numpy as np
import time
from threading import Thread
from processing_v2 import ParquetProcess


class WebcamStream:
    # initialization method
    def __init__(self, stream_id=0):
        self.stream_id = stream_id  # default is 0 for main camera

        # opening video capture stream
        self.vcap = cv2.VideoCapture(self.stream_id)
        if self.vcap.isOpened() is False:
            print("[Exiting]: Error accessing webcam stream.")
            exit(0)
        fps_input_stream = int(self.vcap.get(5))  # hardware fps
        print("FPS of input stream: {}".format(fps_input_stream))

        # reading a single frame from vcap stream for initializing
        self.grabbed, self.frame = self.vcap.read()
        if self.grabbed is False:
            print('[Exiting] No more frames to read')
            exit(0)
        # self.stopped is initialized to False
        self.stopped = True
        # thread instantiation
        self.t = Thread(target=self.update, args=())
        self.t.daemon = True  # daemon threads run in background

    # method to start thread
    def start(self):
        self.stopped = False
        self.t.start()

    # method passed to thread to read next available frame
    def update(self):
        while True:
            if self.stopped is True:
                break
            self.grabbed, self.frame = self.vcap.read()
            if self.grabbed is False:
                print('[Exiting] No more frames to read')
                self.stopped = True
                break
        self.vcap.release()

    # method to return latest read frame
    def read(self):
        return self.frame

    # method to stop reading frames
    def stop(self):
        self.stopped = True

class LiveFeed:
    def __init__(self, selected_landmark_indices):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_hands = mp.solutions.hands
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=2)
        self.selected_landmark_indices = selected_landmark_indices

    def extract_landmarks(self, image, frame_number=0):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image with the face_mesh and hands models
        face_results = self.face_mesh.process(image_rgb)
        hand_results = self.hands.process(image_rgb)

        # Initialize a list to store the data for DataFrame creation
        landmarks_list = []

        # Extract face landmarks if any faces are detected
        if face_results.multi_face_landmarks:
            for idx in self.selected_landmark_indices:  # Loop through your predefined selected indices
                landmark = face_results.multi_face_landmarks[0].landmark[idx]
                landmarks_list.append({
                    'frame': frame_number,
                    'row_id': f'{frame_number}-face-{idx}',
                    'type': 'face',
                    'landmark_index': idx,
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z
                })

        # Extract hand landmarks if hands are detected
        if hand_results.multi_hand_landmarks:
            for hand_no, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                for idx, landmark in enumerate(hand_landmarks.landmark):
                    landmarks_list.append({
                        'frame': frame_number,
                        'row_id': f'{frame_number}-hand{hand_no}-{idx}',
                        'type': 'right_hand',
                        'landmark_index': idx,
                        'x': landmark.x,
                        'y': landmark.y,
                        'z': landmark.z
                    })

        # Create a DataFrame from the list of landmarks
        landmarks_df = pd.DataFrame(landmarks_list)

        return landmarks_df

    def live_gesture(self, dataframe, frame_number):
        coordinates=parquet_proccessor.extract_landmarks(dataframe,frame_number,self.selected_landmark_indices)
        dist_matrix=parquet_proccessor.distance(coordinates)
        angle_marix=parquet_proccessor.angle_matrix(coordinates)
        trsh_matrix=parquet_proccessor.treshold_matrix(dist_matrix)
        rgb=parquet_proccessor.make_img(trsh_matrix,angle_marix,dist_matrix)
        plt.imshow(rgb)
        plt.show(block=False)
        plt.pause(0.01)  # Pause to display the current frame's matrix
        plt.clf()
    # initializing and starting multi-threaded webcam input stream
selected_landmark_indices = [33, 133, 159, 263, 46, 70, 4, 454, 234, 10, 338, 297, 332, 61, 291, 0, 78, 14, 317,
                             152, 155, 337, 299, 333, 69, 104, 68, 398]
parquet_proccessor=ParquetProcess()
live_feed=LiveFeed(selected_landmark_indices)
webcam_stream = WebcamStream(stream_id=0)  # 0 id for main camera
webcam_stream.start()
# processing frames in input stream
num_frames_processed = 0
start = time.time()
while True :
    if webcam_stream.stopped is True :
        break
    else :
        frame = webcam_stream.read()
        lnd=live_feed.extract_landmarks(frame, frame_number=num_frames_processed)
        print(lnd)
        live_feed.live_gesture(lnd,num_frames_processed)
    # adding a delay for simulating video processing time
    delay = 0.03 # delay value in seconds
    time.sleep(delay)
    num_frames_processed += 1
    # displaying frame
    #cv2.imshow('frame' , frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
end = time.time()
webcam_stream.stop() # stop the webcam stream