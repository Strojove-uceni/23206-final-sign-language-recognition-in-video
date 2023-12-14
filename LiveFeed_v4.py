import matplotlib.pyplot as plt
import mediapipe as mp
import pandas as pd
import cv2
import numpy as np
import time
from threading import Thread
from processing_v2 import ParquetProcess
from ParquetToMatrix import ParquetToMatrix
from matplotlib import animation


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
    def __init__(self, selected_landmark_indices, depth=140):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_hands = mp.solutions.hands
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=2)
        self.selected_landmark_indices = selected_landmark_indices
        self.depth = depth
        self.tensor = None
        self.initialized = False

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
                    'type': 'face',
                    'row_id': f'{frame_number}-face-{idx}',

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
                        'type': 'right_hand',
                        'row_id': f'{frame_number}-hand{hand_no}-{idx}',

                        'landmark_index': idx,
                        'x': landmark.x,
                        'y': landmark.y,
                        'z': landmark.z
                    })

        # Create a DataFrame from the list of landmarks
        landmarks_df = pd.DataFrame(landmarks_list)


        return landmarks_df

    def live_gesture(self, parquet_proccessor, dataframe, max_length):
        """
        This function performs processing on an input dataframe for live mode.
        It's supposed to be used for real-time or live video feeds.

        :param dataframe: The input DataFrame to perform the gesture analysis on.
        :param max_length: The maximum length of the output array.
        :return: The processed array with shape (max_length, len(dfs_result), 3)
        """

        # 1. Cleaning the DataFrame.
        # Here, we are not visualizing the DataFrame, so passing False.
        parquet_proccessor.df = dataframe
        # parquet_proccessor.clean_parquet(show_df=False)

        # 2. Now we create the matrix from the DataFrame for max_length.
        parquet_proccessor.tssi_preprocess(max_length)
        plt.imshow(parquet_proccessor.concatenated_matrix)
        plt.show(block=False)
        plt.pause(0.1)  # Pause to display the current frame's matrix
        plt.clf()


selected_landmark_indices = [46, 52, 53, 65, 7, 159, 155, 145, 0,
                             295, 283, 282, 276, 382, 385, 249, 374, 13, 324, 76, 14]
parquet_proccessor = ParquetToMatrix(None, selected_landmark_indices, max_length=96)
live_feed=LiveFeed(selected_landmark_indices,depth=96)
webcam_stream = WebcamStream(stream_id=0)  # 0 id for main camera
webcam_stream.start()
# processing frames in input stream
num_frames_processed = 0
empty_df = pd.DataFrame()
start = time.time()
while True :
    if webcam_stream.stopped is True :
        break
    else :
        if num_frames_processed <= 96:
            frame = webcam_stream.read()
            lnd = live_feed.extract_landmarks(frame, frame_number=num_frames_processed)
            extracted_data = pd.concat([empty_df, lnd])
            empty_df = extracted_data.copy()
        else:
            live_feed.live_gesture(parquet_proccessor, extracted_data, 96)

    num_frames_processed += 1
    # displaying frame
    #cv2.imshow('frame' , frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
end = time.time()
webcam_stream.stop() # stop the webcam stream
