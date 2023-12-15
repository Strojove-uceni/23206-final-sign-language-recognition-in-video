import matplotlib.pyplot as plt
import mediapipe as mp
import pandas as pd
import cv2
import torch
import numpy as np
import time
from threading import Thread
from processing_v2 import ParquetProcess
from ParquetToMatrix import ParquetToMatrix
from matplotlib import animation
import pytorch_lightning as pl
import torch.cuda
import time
import wandb
from pytorch_lightning.loggers import WandbLogger
from Models import AslLitModelMatrix, AslLitModelMatrix2
from pytorch_lightning.tuner import Tuner
from torch.optim.lr_scheduler import ReduceLROnPlateau
from Dataset import AslMatrixDataModule
from pytorch_lightning.callbacks import StochasticWeightAveraging
from ResNet_custom import ImageClassifier, ResNet50
from pytorch_lightning.callbacks import EarlyStopping
from test_live import  LiveFeedAI


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

        # Initialize a dictionary to store the data for DataFrame creation.
        # Prepopulate it with all potential landmarks as keys, with values set to NaN.
        total_landmarks = 468 + (21 * 2)  # 468 face landmarks, 42 hand landmarks (21 per hand)
        landmarks_dict = {idx: {
            'frame': frame_number,
            'type': np.nan,
            'row_id': np.nan,
            'landmark_index': idx,
            'x': np.nan,
            'y': np.nan,
            'z': np.nan
        } for idx in range(total_landmarks)}

        # Extract face landmarks if any faces are detected
        if face_results.multi_face_landmarks:
            for idx in self.selected_landmark_indices:  # Loop through predefined selected indices
                landmark = face_results.multi_face_landmarks[0].landmark[idx]
                landmarks_dict[idx] = {
                    'frame': frame_number,
                    'type': 'face',
                    'row_id': f'{frame_number}-face-{idx}',
                    'landmark_index': idx,
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z
                }

        # Extract hand landmarks if hands are detected
        if hand_results.multi_hand_landmarks:
            for hand_no, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                hand_type = 'left_hand' if hand_no == 0 else 'right_hand'
                for idx, landmark in enumerate(hand_landmarks.landmark):
                    row_id = f'{frame_number}-{hand_type}-{idx}'

                    # Offset idx by 468 for hand landmarks to avoid key collision with face landmarks
                    landmarks_dict[idx + 468] = {
                        'frame': frame_number,
                        'type': hand_type,
                        'landmark_index': idx,
                        'row_id': row_id,
                        'x': landmark.x,
                        'y': landmark.y,
                        'z': landmark.z
                    }

        landmarks_df = pd.DataFrame(landmarks_dict.values())
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

        return (parquet_proccessor.concatenated_matrix - parquet_proccessor.concatenated_matrix.min())/(parquet_proccessor.concatenated_matrix.max()-parquet_proccessor.concatenated_matrix.min())


# Load model
checkpoint_path = r'C:\Users\drend\Desktop\SU2\ResNet_custom_250cl.ckpt'
sign_list = ['finger', 'garbage', 'girl', 'goo', 'goose', 'yesterday', 'yourself', 'yucky', 'zebra', 'zipper']
model_path = r'C:\Users\drend\Desktop\SU2\good models\2DCNN_10_class_84_prct.ckpt'
model = AslLitModelMatrix2(10)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['state_dict'])
model.to(device)   # Load the model to the correct device
model.eval()

# Indexes of keypoints to use
selected_landmark_indices = [46, 52, 53, 65, 7, 159, 155, 145, 0,
                             295, 283, 282, 276, 382, 385, 249, 374, 13, 324, 76, 14]
parquet_proccessor = ParquetToMatrix(None, selected_landmark_indices, max_length=96)
live_feed=LiveFeed(selected_landmark_indices,depth=96)

webcam_stream = WebcamStream(stream_id=0)  # 0 for main camera
webcam_stream.start()

num_frames_processed = 0
empty_df = pd.DataFrame(columns=['frame', 'type', 'landmark_index', 'index', 'x', 'y', 'z'])

while num_frames_processed < 96:  # Collect only 96 frames
    frame = webcam_stream.read()
    cv2.imshow('Webcam Stream', frame)
    cv2.waitKey(1)
    lnd = live_feed.extract_landmarks(frame, frame_number=num_frames_processed)
    extracted_data = pd.concat([empty_df, lnd])
    empty_df = extracted_data.copy()
    num_frames_processed += 1

# Stop the webcam stream after collecting the frames
webcam_stream.stop()
cv2.destroyAllWindows()

# Process collected gesture
gesture = live_feed.live_gesture(parquet_proccessor, extracted_data, 96)
gesture = np.transpose(gesture, (2,0,1))
gesture = torch.from_numpy(gesture).float()
gesture = torch.unsqueeze(gesture, dim=0)
gesture = gesture.to(device)

# Get model prediction
with torch.no_grad():
    output = model(gesture)
    probabilities = torch.nn.functional.softmax(output, dim=1)
    _, top_pred = torch.topk(probabilities, 1)  # Get the top prediction

# Print the top prediction and corresponding probability
prediction = top_pred[0].item()
probability = probabilities[0][prediction].item()
print(f"Prediction: {sign_list[prediction]}, Probability: {probability}")
