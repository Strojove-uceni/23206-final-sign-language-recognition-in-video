import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
import cv2


class ParquetProcess:

    def euclidean_distance(p1, p2):
        return np.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

    def compute_angle(self, p1, p2, p3):
        # Vectors
        a = np.array([p1[0] - p2[0], p1[1] - p2[1]])
        b = np.array([p3[0] - p2[0], p3[1] - p2[1]])
        # Dot product and magnitudes
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        # Prevent division by zero
        if norm_a == 0 or norm_b == 0:
            return 0
        # Angle
        cos_theta = dot_product / (norm_a * norm_b + 1e-10)  # Adding a small epsilon value
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

    def CalculateAngleMatrix(self, dataframe, frame_number, selected_landmark_indices):
        hand_rows = dataframe[(dataframe['frame'] == frame_number) & (((dataframe['type'] == 'right_hand') | (dataframe['type'] == 'left_hand')))]
        face_rows = dataframe[(dataframe['frame'] == frame_number) & (dataframe['type'] == 'face')]


        hand_coordinates = hand_rows[['x', 'y', 'z']].values

        # Select landmark coordinates for faces based on specified indices
        face_coordinates = face_rows[face_rows['landmark_index'].isin(selected_landmark_indices)][
            ['x', 'y', 'z']].values

        combined_coordinates = np.concatenate((hand_coordinates, face_coordinates), axis=0)
        num_landmarks=len(combined_coordinates)
        angle_matrix = np.zeros((num_landmarks, num_landmarks))
        for i in range(num_landmarks):
            for j in range(i + 1, num_landmarks - 1):  # modified range to avoid index error
                # Compute angle only if the next landmark is different, hence j + 2 check to ensure a triplet
                if (j + 2) < num_landmarks:
                    p1 = combined_coordinates[i]
                    p2 = combined_coordinates[j]
                    p3 = combined_coordinates[j+1]
                    angle_matrix[i, j] = self.compute_angle(p1, p2, p3)
        max_value = angle_matrix.max()
        normalized_matrix = angle_matrix / max_value if max_value > 0 else angle_matrix
        return normalized_matrix

    def CalculateThresholdMatrix(self, distance_matrix, threshold=0.05):
        proximity_matrix = (distance_matrix < threshold).astype(int)
        return proximity_matrix

    def AnimateParquet(self,df):
        unique_frames = sorted(df['frame'].unique())
        for frame_number in unique_frames:
            distance_matrix = parquet_processor.CalculateDistanceMatrix(clean_df, frame_number,
                                                                        selected_landmark_indices)
            angle_matrix=parquet_processor.CalculateAngleMatrix(clean_df,frame_number,selected_landmark_indices)
            treshold_matrix=parquet_processor.CalculateThresholdMatrix(distance_matrix)

            dist_img = (255 * (distance_matrix - np.min(distance_matrix)) / (
                    np.max(distance_matrix) - np.min(distance_matrix))).astype(np.uint8)
            angle_img = (255 * (angle_matrix - np.min(angle_matrix)) / (
                    np.max(angle_matrix) - np.min(angle_matrix))).astype(np.uint8)
            proximity_img = (255 * treshold_matrix).astype(np.uint8)

            rgb_img = cv2.merge([proximity_img, angle_img, dist_img])
            rgb_img=cv2.flip(rgb_img,1)
            plt.imshow(rgb_img)
            plt.title(f'Frame {frame_number}')

            plt.show(block=False)
            plt.pause(0.01)  # Pause to display the current frame's matrix
            plt.clf()  # Clear the figure to display the next frame's matrix

        plt.close('all')  # Close the plot when done



selected_landmark_indices = [33, 133, 159, 263, 46, 70, 4, 454, 234, 10, 338, 297, 332, 61, 291, 0, 78, 14, 317,
                                 152, 155, 337, 299, 333, 69, 104, 68, 398]
parquet_processor = ParquetProcess()
df = parquet_processor.ReadParquet(r'C:\Users\drend\Desktop\3574671853.parquet')
clean_df = parquet_processor.CleanParquet(df)

parquet_processor.AnimateParquet(clean_df)
