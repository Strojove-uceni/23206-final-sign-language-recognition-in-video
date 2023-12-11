from ParquetToMatrix import ParquetToMatrix
import pandas as pd
import numpy as np
import os
import shutil
from dfs_results import dfs_result


class ParquetData:
    def __init__(self):
        self.data = {}

    def preprocess_all(self, path, csv_list, landmark_id, sign_list, max_length=140):
        count = 0
        for participant_folder in os.listdir(path + "/train_landmark_files/"):
            path_folder = path + "/train_landmark_files/" + participant_folder + "/"
            path_matrix = path + "/matrix"
            for file_name in os.listdir(path_folder):
                count = count + 1

                file_num = file_name.split(".")[0]
                df_row = csv_list[(csv_list.participant_id == int(participant_folder)) & (csv_list.sequence_id == int(file_num))]
                file_path = path + "/train_landmark_files/" + participant_folder + "/" + file_num + ".parquet"
                df = pd.read_parquet(file_path)
                df = df.dropna()
                unique_frames = len(df['frame'].unique())
                if (unique_frames > max_length):
                    print('File was skipped:: ', file_path)
                    continue
                else:
                    sign_name = df_row.sign.values[0]
                    if not sign_list:
                        readed_data = ParquetToMatrix(file_path, landmark_id, max_length)
                        self.save_tensor(path_matrix, sign_name, readed_data.concatenated_matrix, count)
                    else:
                        if sign_name in sign_list:
                            readed_data = ParquetToMatrix(file_path, landmark_id, max_length)
                            self.save_tensor(path_matrix, sign_name, readed_data.concatenated_matrix, count)
            print("Data preprocess completed!")

    def preprocess(df):
        print(df)

    
    def save_tensor(self, path, sign, tensor, count):
        """

        :param path: Path to folder with data
        :param sign: Sign name
        :param tensor: Tensor we want to save as numpy array
        :param count: Just for the purpose of avoiding name conflicts
        :return: Saved tensor
        """

        sign_dir = os.path.join(path, sign)

        if not os.path.exists(sign_dir):
            os.makedirs(sign_dir)

        file_path = os.path.join(sign_dir, sign + "_" + str(count) + 'matrix.npy')

        np.save(file_path, tensor, allow_pickle=False)
        print(f"Tensor saved successfully at {file_path}")



# path = r"E:\asl-signs"
path = "C:/Skoda_Digital/Materials/Documents_FJFI/SU2/asl-signs"
selected_landmark_indices = [46, 52, 53, 65, 7, 159, 155, 145, 0,
                             295, 283, 282, 276, 382, 385, 249, 374, 13, 324, 76, 14]

signs=['airplane', 'alligator', 'any', 'apple', 'balloon', 'bath',
       'black', 'drink', 'drop', 'dry', 'duck', 'ear', 'empty', 'face', 'find', 'fine', 'finger', 'garbage', 'girl', 'goose']
df_train = pd.read_csv(path + "/train_mod.csv", sep=",")

data_load = ParquetData()
data_load.preprocess_all(path, df_train, selected_landmark_indices, signs, 140)
