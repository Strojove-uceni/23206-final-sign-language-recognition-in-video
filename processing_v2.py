import pandas as pd
import numpy as np
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
import cv2
from mpl_toolkits.mplot3d import Axes3D


class ParquetProcess:

    def read_parquet(self, directory):
        """
        Reads parquet file from directory.

        Parameters:
        directory (string): File directory.

        Returns:
        dataframe (DataFrame): Read data.
        """
        print(f'Reading file from {directory}')
        dataframe = pd.read_parquet(directory)
        print(f'Successfully read file')
        return dataframe

    def clean_parquet(self, dataframe, show_df=True):
        """
        Summarizes the information about the loaded, parquet dataframe and removes the NaN values.

        Parameters:
        dataframe (DataFrame): Parquet dataframe.
        show_df (Bool, optional): If True, prints the first lines of the dataframe after cleaning. Defaults to True.

        Returns:
        cleaned_df (DataFrame): Cleared dataframe.
        """
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

    def normalize_matrix(self, matrix):
        """
        Normalizes the coordinate matrix of points using min-max normalization.

        Parameters:
        matrix (np.array): Coordinate matrix.

        Returns:
        normalized_matrix (np.array): Normalized coordinate matrix.
        """
        min_val = np.min(matrix)
        max_val = np.max(matrix)
        normalized_matrix = (matrix - min_val) / (max_val - min_val)
        return normalized_matrix

    def distance(self, coordinates):
        """
        Calculates the matrix of mutual distances of significant points and normalizes it.

        Parameters:
        coordinates (np.array): Coordinate array.

        Returns:
        distance_norm (np.array): Normalized distance matrix.
        """
        dist_norm = distance_matrix(coordinates, coordinates)
        distance_norm = self.normalize_matrix(dist_norm)
        return distance_norm

    def angle_between_vectors(self, v1, v2):
        """
        Calculates the angle between vectors.
        
        Parameters:
        v1 (np.array): 
        v2 (np.array): 

        Returns:
        angle_deg (float): 
        """
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
        """
        Calculates the matrix of angles between all vectors.
        
        Parameters:
        vectors (np.array): 

        Returns:
        angle_mat (np.array): 
        """
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
        """
        Replaces distances less than threshold with zero.
        
        Parameters:
        distance (np.array): 
        treshold (float, optional): 

        Returns:
        proximity_matrix (): 
        """
        proximity_matrix = (distance < treshold).astype(int)
        proximity_matrix = self.normalize_matrix(proximity_matrix)
        return proximity_matrix

    def make_img(self, R, G, B):
        """
        Creates an image.
        
        Parameters:
        R (np.array): 
        G (np.array): 
        B (np.array): 

        Returns:
        rgb_img (cv2.image): 
        """
        rgb_img = cv2.merge([R, G, B])
        rgb_img = cv2.flip(rgb_img, 1)
        return rgb_img

    def animate_parquet(self, df, selected_landmark_indices):
        """
        Plots the data representation animation for all video frames.

        Parameters:
        df (DataFrame): 
        selected_landmark_indices (list): 
        """
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

    def create_tensor(self, df, selected_landmark_indices):
        """
        Converts the data matrix to a tensor.
        
        Parameters:
        df (DataFrame): 
        selected_landmark_indices (list): 

        Returns:
        big_array (np.array): 
        """
        unique_frames = sorted(df['frame'].unique())
        stacked_images = []  # This list will store each individual image to be stacked

        for frame in unique_frames:
            hand_rows = df[(df['frame'] == frame) & ((df['type'] == 'right_hand') | (df['type'] == 'left_hand'))]
            if hand_rows.empty:  # Skip frames without hand landmarks
                continue
            pos = self.extract_landmarks(df, frame, selected_landmark_indices)
            dist = self.distance(pos)
            angle = self.angle_matrix(pos)
            trsh = self.treshold_matrix(dist)
            rgb_img = self.make_img(trsh, angle, dist)
            stacked_images.append(rgb_img)  # Append the current image to the list of images to be stacked

        # Only after all frames have been processed do we concatenate the images
        if stacked_images:  # Check if there are any images collected to stack
            # Concatenate all the collected images along the third dimension (channel dimension)
            big_array = np.concatenate(stacked_images, axis=2)
            return big_array
        else:
            # If no images were collected, return an empty array with the correct empty shape
            return np.empty((0, 0, 0))

    def plot_3d_cube_with_transparency(self, image_stack):
        """
        Plots a 3d projection of a matrix representation of all video frames.
        
        Parameters:
        image_stack (): 
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ny, nx, total_channels = image_stack.shape
        number_of_frames = total_channels // 3

        # Pre-calculate the grid which will be constant for all frames
        x_grid, y_grid = np.meshgrid(np.arange(nx), np.arange(ny))

        for frame in range(number_of_frames):
            # Extract the current frame's RGB values and calculate intensity
            current_frame = image_stack[:, :, frame * 3:(frame + 1) * 3]
            intensity = current_frame.mean(axis=2)
            normalized_intensity = intensity / intensity.max()  # Normalizing to the max intensity

            # Flatten the arrays for vectorized scatter plot
            x_vals = x_grid.flatten()
            y_vals = y_grid.flatten()
            z_vals = np.full(x_vals.shape, frame)
            color_vals = np.stack([np.zeros_like(normalized_intensity),
                                   normalized_intensity,
                                   np.zeros_like(normalized_intensity)], axis=2).reshape(-1, 3)

            # Plot all points at once for this frame
            ax.scatter(x_vals, y_vals, z_vals, c=color_vals, alpha=normalized_intensity.flatten())

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Frame')

        plt.show()





