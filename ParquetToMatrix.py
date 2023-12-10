import pandas as pd
import numpy as np
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
import cv2


class ParquetToMatrix():
    def __init__(self, path, landmark_id, max_length):
        self.concatenated_matrix = []
        self.max_length = max_length
        self.df = self.read_parquet(path)
        self.create_matrix(self.df, landmark_id)


    def read_parquet(self, directory):
        """
        Reads parquet file from directory.

        Parameters:
        directory (string): File directory.

        Returns:
        dataframe (DataFrame): Read data.
        """
        dataframe = pd.read_parquet(directory)
        #print('Successfully read file from: ', directory)
        return dataframe

    def clean_parquet(self, show_df=True):
        """
        Summarizes the information about the loaded, parquet dataframe and removes the NaN values.

        Parameters:
        dataframe (DataFrame): Parquet dataframe.
        show_df (Bool, optional): If True, prints the first lines of the dataframe after cleaning. Defaults to True.

        Returns:
        clean_df (DataFrame): Cleared dataframe.
        """
        self.clean_df = self.df.fillna(0)
        if show_df == 1:
            print(f'Here is few lines from parquet file')
            print(self.clean_df.head())

    def extract_landmarks(self, dataframe, frame_number, selected_landmark_indices):
        """
        Combines the coordinates of selected significant points from a single frame of video.

        Parameters:
        dataframe (DataFrame): Parquet dataframe.
        frame_number (int): Number of video frame.
        selected_landmark_indices (list): List of indexes of selected points.

        Returns:
        combined_coordinates (np.array): Matrix with combined coordinates.
        """
        hand_rows = dataframe[(dataframe['frame'] == frame_number) & (
            ((dataframe['type'] == 'right_hand') | (dataframe['type'] == 'left_hand')))]
        face_rows = dataframe[(dataframe['frame'] == frame_number) & (dataframe['type'] == 'face')]
        hand_coordinates = hand_rows[['x', 'y', 'z']].values
        face_coordinates = face_rows[face_rows['landmark_index'].isin(selected_landmark_indices)][
            ['x', 'y', 'z']].values
        combined_coordinates = np.concatenate((hand_coordinates, face_coordinates), axis=0)
        return combined_coordinates

    def create_matrix(self, dataframe, selected_landmark_indices):
        dataframe = dataframe.dropna()

        unique_frames = dataframe['frame'].unique()
        num_landmarks = len(selected_landmark_indices)
        coords_per_landmark = 3  # Each landmark has x, y, z coordinates
        expected_length = num_landmarks * coords_per_landmark

        frame_data = []

        for i, frame in enumerate(unique_frames):
            if i >= self.max_length:
                break

            frame_coordinates = self.extract_landmarks(dataframe, frame, selected_landmark_indices).flatten()

            # Adjust the frame data to the expected length
            if len(frame_coordinates) > expected_length:
                # Truncate if longer
                frame_coordinates = frame_coordinates[:expected_length]
            elif len(frame_coordinates) < expected_length:
                # Pad with zeros if shorter
                padded_coordinates = np.zeros(expected_length, dtype=np.float32)
                padded_coordinates[:len(frame_coordinates)] = frame_coordinates
                frame_coordinates = padded_coordinates

            frame_data.extend(frame_coordinates)

        # Convert the list to a NumPy array and reshape
        self.concatenated_matrix = np.array(frame_data, dtype=np.float32).reshape(-1, expected_length)

        # Normalize the matrix
        max_val = self.concatenated_matrix.max()
        min_val = self.concatenated_matrix.min()
        if max_val != min_val:
            self.concatenated_matrix = (self.concatenated_matrix - min_val) / (max_val - min_val)



selected_landmark_indices = [33, 133, 159, 263, 46, 70, 4, 454, 234, 10, 338, 297, 332, 61, 291, 0, 78, 14, 317,
                             152, 155, 337, 299, 333, 69, 104, 68, 398]
