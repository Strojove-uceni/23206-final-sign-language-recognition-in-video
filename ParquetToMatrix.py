import pandas as pd
import numpy as np
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
import cv2
from dfs_results import dfs_result
import torch
from torch.nn import functional as F


class ParquetToMatrix():
    def __init__(self, path, landmark_id, max_length):
        self.concatenated_matrix = []
        self.max_length = max_length
        if(not path is None):
            self.df = self.read_parquet(path)
            # self.create_matrix(self.df, landmark_id)
            self.tssi_preprocess(max_length)
            self.preprocess_succes = True


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

    def tssi_preprocess(self, max_length):
        frames = self.df.frame.unique()
        npy_preprocess = np.empty((max_length, len(dfs_result), 3))
        npy_preprocess[:] = np.nan

        self.df = self.df.sort_values(by=['frame'])

        right_hand_nan_count = 0
        left_hand_nan_count = 0
        landmark_eyebrow_id = []
        for dfs_landmark_id in range(len(dfs_result)):
            dfs_landmark_name = dfs_result[dfs_landmark_id]
            (landmark, body_part) = dfs_landmark_name.split("-")
            if(int(landmark) == 65 or int(landmark) == 295):
                # store id of eyebrows
                landmark_eyebrow_id += [dfs_landmark_id]
            df_landmark = self.df[(self.df.landmark_index == int(landmark)) & (self.df.type == body_part)]
            if(len(df_landmark) != 0):
                xyz_cord = np.array([df_landmark.x.values, df_landmark.y.values, df_landmark.z.values])
                if(np.isnan(np.min(xyz_cord))):
                    if(dfs_landmark_id < 21):
                        right_hand_nan_count += np.sum(np.isnan(xyz_cord))
                    elif(dfs_landmark_id > 41):
                        left_hand_nan_count += np.sum(np.isnan(xyz_cord))

                    nan_mask = np.isnan(xyz_cord)
                    rows_without_nan = ~nan_mask.any(axis=0)
                    xyz_without_nan = xyz_cord[:, rows_without_nan]
                    if(xyz_without_nan.shape[1] == 0):
                        xyz_interp = F.interpolate(torch.tensor(xyz_cord)[None, :], size=max_length, mode='linear').numpy()
                    else: 
                        xyz_interp = F.interpolate(torch.tensor(xyz_without_nan)[None, :], size=max_length, mode='linear').numpy()
                    xyz_transform = np.squeeze(xyz_interp, axis=0).T
                else:
                    xyz_interp = F.interpolate(torch.tensor(xyz_cord)[None, :], size=max_length, mode='linear').numpy()
                    xyz_transform = np.squeeze(xyz_interp, axis=0).T

                npy_preprocess[:, dfs_landmark_id, :] = np.copy(xyz_transform)

        # if(len(landmark_eyebrow_id) == 2):
        #     norm_point = (npy_preprocess[:, landmark_eyebrow_id[0], :] + npy_preprocess[:, landmark_eyebrow_id[1], :])/2
        # else:
        #     self.preprocess_succes = False

        # for dfs_landmark_id in range(len(dfs_result)):
        #     npy_preprocess[:, dfs_landmark_id, :] = npy_preprocess[:, dfs_landmark_id, :] - norm_point

        # if(left_hand_nan_count < right_hand_nan_count):
        #     npy_hand_removed = npy_preprocess[:, 21:64, :]
        # else:
        #     npy_hand_removed = npy_preprocess[:, 0:42, :]

        self.concatenated_matrix = np.nan_to_num(npy_preprocess, copy=True, nan=0)


selected_landmark_indices = [46, 52, 53, 65, 7, 159, 155, 145, 0,
                             295, 283, 282, 276, 382, 385, 249, 374, 13, 324, 76, 14]
