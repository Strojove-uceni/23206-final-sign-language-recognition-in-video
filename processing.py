import pandas as pd
import numpy as np
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
import cv2


class ParquetProcess:

    def read_parquet(self, directory):
        print(f'Reading file from {directory}')
        dataframe = pd.read_parquet(directory)
        print(f'Successfully read file')
        return dataframe

    def clean_parquet(self, dataframe, show_df=True):
        print('DataFrame summary before cleaning:')
        print(dataframe.info())  # Summarize before cleaning
        print('\nCount of NaN values by column before cleaning:')
        print(dataframe.isna().sum())  # Count NaNs before cleaning
        cleaned_df = dataframe.dropna()
        print('DataFrame summary after cleaning:')
        print(cleaned_df.info())  # Summarize after cleaning
        print('\nCount of NaN values by column after cleaning:')
        print(cleaned_df.isna().sum())  # Count NaNs after cleaning
        unique_frames_after_cleaning = cleaned_df['frame'].unique()
        print("Unique frames after cleaning:", unique_frames_after_cleaning)
        if show_df == 1:
            print(f'Here is few lines from parquet file')
            print(cleaned_df.head())
        return cleaned_df

    def extract_landmarks(self, dataframe, frame_number, selected_landmark_indices):
        hand_rows = dataframe[(dataframe['frame'] == frame_number) & (
            ((dataframe['type'] == 'right_hand') | (dataframe['type'] == 'left_hand')))]
        face_rows = dataframe[(dataframe['frame'] == frame_number) & (dataframe['type'] == 'face')]
        hand_coordinates = hand_rows[['x', 'y', 'z']].values
        face_coordinates = face_rows[face_rows['landmark_index'].isin(selected_landmark_indices)][
            ['x', 'y', 'z']].values
        combined_coordinates = np.concatenate((hand_coordinates, face_coordinates), axis=0)
        return combined_coordinates

    def normalize_matrix(self, matrix):
        min_val = np.min(matrix)
        max_val = np.max(matrix)
        normalized_matrix = (matrix - min_val) / (max_val - min_val)
        return normalized_matrix

    def distance(self, coordinates):
        dist_norm = distance_matrix(coordinates, coordinates)
        distance_norm = self.normalize_matrix(dist_norm)
        return distance_norm

    def angle_between_vectors(self, v1, v2):
        # Calculate the dot product and the angle in radians
        dot_prod = np.dot(v1, v2)
        norms = np.linalg.norm(v1) * np.linalg.norm(v2)
        if norms == 0:
            angle = 0
        else:
            angle = np.arccos(np.clip(dot_prod / norms, -1.0, 1.0))
        angle_deg = np.degrees(angle)
        return angle_deg

    def angle_matrix(self, vectors):
        n = vectors.shape[0]
        angle_mat = np.zeros((n, n))

        for i in range(n):
            for j in range(i, n):
                angle = self.angle_between_vectors(vectors[i], vectors[j])
                angle_mat[i, j] = angle
                angle_mat[j, i] = angle
        angle_mat = np.nan_to_num(angle_mat, nan=0)
        angle_mat = self.normalize_matrix(angle_mat)
        return angle_mat

    def treshold_matrix(self, distance, treshold=0.05):
        proximity_matrix = (distance < treshold).astype(int)
        proximity_matrix = self.normalize_matrix(proximity_matrix)
        return proximity_matrix

    def make_img(self, R, G, B):
        rgb_img = cv2.merge([R, G, B])
        rgb_img = cv2.flip(rgb_img, 1)
        return rgb_img

    def animate_parquet(self, df, selected_landmark_indices):
        unique_frames = sorted(df['frame'].unique())
        for frame in unique_frames:
            hand_rows = df[(df['frame'] == frame) & ((df['type'] == 'right_hand') | (df['type'] == 'left_hand'))]
            if hand_rows.empty:  # Skip frames without hand landmarks
                continue
            pos = self.extract_landmarks(df, frame, selected_landmark_indices)
            dist = self.distance(pos)
            angle = self.angle_matrix(pos)
            trsh = self.treshold_matrix(dist)
            rgb_img = self.make_img(trsh, angle, dist)
            plt.imshow(rgb_img)
            plt.title(f'Frame {frame}')

            plt.show(block=False)
            plt.pause(0.01)  # Pause to display the current frame's matrix
            plt.clf()  # Clear the figure to display the next frame's matrix

        plt.close('all')


selected_landmark_indices = [33, 133, 159, 263, 46, 70, 4, 454, 234, 10, 338, 297, 332, 61, 291, 0, 78, 14, 317,
                             152, 155, 337, 299, 333, 69, 104, 68, 398]
parquet_processor = ParquetProcess()
df = parquet_processor.read_parquet(r'C:\Users\drend\Desktop\3574671853.parquet')
clean_df = parquet_processor.clean_parquet(df, show_df=False)
parquet_processor.animate_parquet(clean_df, selected_landmark_indices)
