import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
import cv2


class ParquetProcess:

    def euclidean_distance(p1, p2):
        return np.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

    def compute_angle(p1, p2, p3):
        # Vectors
        a = np.array([p1.x - p2.x, p1.y - p2.y])
        b = np.array([p3.x - p2.x, p3.y - p2.y])
        # Dot product and magnitudes
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        # Angle
        cos_theta = dot_product / (norm_a * norm_b)
        angle = np.arccos(np.clip(cos_theta, -1, 1))
        return angle

    def ReadParquet(self, file_directory):
        print(f"Reading the parquet file from {file_directory}...")
        dataframe = pd.read_parquet(file_directory)
        print("File read successfully!")
        return dataframe

    def CleanParquet(self, dataframe):
        print("Cleaning the dataframe by replacing NaN values with zeros...")
        cleaned_dataframe = dataframe.fillna(0)
        print("Here are the first few rows of the cleaned dataframe:")
        print(cleaned_dataframe.head())
        return cleaned_dataframe

    def CalculateDistanceMatrix(self, dataframe, frame_number, selected_landmark_indices):
        #Filtrace typů
        hand_rows = dataframe[(dataframe['frame'] == frame_number) & (((dataframe['type'] == 'right_hand') | (dataframe['type'] == 'left_hand')))]
        face_rows = dataframe[(dataframe['frame'] == frame_number) & (dataframe['type'] == 'face')]

        # Výběr landmarks
        hand_coordinates = hand_rows[['x', 'y', 'z']].values

        # Select landmark coordinates for faces based on specified indices
        face_coordinates = face_rows[face_rows['landmark_index'].isin(selected_landmark_indices)][
            ['x', 'y', 'z']].values

        combined_coordinates = np.concatenate((hand_coordinates, face_coordinates), axis=0)

        # Calculate the distance matrix using Euclidean distance
        distance_matrix = np.sqrt(
            np.sum((combined_coordinates[:, np.newaxis, :] - combined_coordinates[np.newaxis, :, :]) ** 2, axis=-1))

        # Normalize the distance matrix to [0,1]
        max_value = distance_matrix.max()
        normalized_matrix = distance_matrix / max_value if max_value > 0 else distance_matrix

        return normalized_matrix

    def CalculateAngleMatrix(self, landmarks):
        num_landmarks = len(landmarks)
        angle_matrix = np.zeros((num_landmarks, num_landmarks, num_landmarks))
        # Iterate over all unique combinations of three different landmarks
        for i, j, k in itertools.combinations(range(num_landmarks), 3):
            angle_matrix[i, j, k] = parquet_processor.compute_angle(landmarks[i], landmarks[j], landmarks[k])
            angle_matrix[i, k, j] = angle_matrix[i, j, k]  # Angle is the same for j-k and k-j order
        return angle_matrix

    def CalculateThresholdMatrix(self, distance_matrix, threshold=0.05):
        proximity_matrix = (distance_matrix < threshold).astype(int)
        return proximity_matrix

    def AnimateParquet(df):
        unique_frames = sorted(df['frame'].unique())
        for frame_number in unique_frames:
            distance_matrix = parquet_processor.CalculateDistanceMatrix(clean_df, frame_number,
                                                                        selected_landmark_indices)

            plt.imshow(distance_matrix, cmap='gray')
            plt.title(f'Frame {frame_number}')
            plt.colorbar()
            plt.show(block=False)
            plt.pause(0.08)  # Pause to display the current frame's matrix
            plt.clf()  # Clear the figure to display the next frame's matrix

        plt.close('all')  # Close the plot when done


selected_landmark_indices = [33, 133, 159, 263, 46, 70, 4, 454, 234, 10, 338, 297, 332, 61, 291, 0, 78, 14, 317,
                                 152, 155, 337, 299, 333, 69, 104, 68, 398]
parquet_processor = ParquetProcess()
df = parquet_processor.ReadParquet(r'C:\Users\drend\Desktop\3574671853.parquet')
clean_df = parquet_processor.CleanParquet(df)


