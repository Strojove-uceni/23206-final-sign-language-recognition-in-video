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


df_line = df_train[df_train.index == 0]

path_file = df_line.path.values[0]

parquet_file = pd.read_parquet(path + "/" + path_file)

all_selected_landmark_indices = [int(dfs_landmark.split("-")[0]) for dfs_landmark in dfs_result]
selected_body_parts = ["face", "body", "left_hand", "right_hand"]

frames = parquet_file.frame.unique()
npy_preprocess = np.empty((len(frames),len(dfs_result),3))
npy_preprocess[:] = np.nan

for frame_id in range(len(frames)):
    frame = frames[frame_id]
    df_frame = parquet_file[parquet_file.frame == frame]

    # df_reduced = df_frame[df_frame.landmark_index.isin(all_selected_landmark_indices) & df_frame.type.isin(selected_body_parts)]
    # df_reduced = pd.DataFrame(columns=df_frame.columns)
    # for index, row in df_frame.iterrows():
    #     dfs_landmark = str(row.landmark_index) + "-" + row.type
    #     if dfs_landmark in dfs_result:
    #         df_reduced = df_reduced._append(row)

    for dfs_landmark_id in range(len(dfs_result)):
        dfs_landmark = dfs_result[dfs_landmark_id]
        (landmark, body_part) = dfs_landmark.split("-")
        df_selected_row = df_frame[(df_frame.landmark_index == int(landmark)) & (df_frame.type == body_part)]
        if(len(df_selected_row) == 0):
            df_selected_row = df_frame[(df_frame.landmark_index == 0) & (df_frame.type == body_part)]
            if(len(df_selected_row) == 0):
                df_selected_row = df_selected_row._append({"frame": frame_id, "row_id": 0, "type": body_part, "landmark_index": landmark, "x": 0, "y": 0, "z": 0}, ignore_index=True)

        if(df_selected_row.x.values[0] == np.nan or df_selected_row.y.values[0] == np.nan or df_selected_row.z.values[0] == np.nan):
            df_selected_row = df_frame[(df_frame.landmark_index == 0) & (df_frame.type == body_part)]
        
        df_selected_row = df_selected_row.fillna(0)
        
        # uložit data
        npy_preprocess[frame_id, dfs_landmark_id, :] = np.array([df_selected_row.x.values[0], df_selected_row.y.values[0], df_selected_row.z.values[0]])

# Odebrání nulových landmark
# matrix_to_del = []
# for dfs_landmark_id in range(len(dfs_result)):
#     if(np.sum(npy_preprocess[:, dfs_landmark_id, :]) == 0):
#         matrix_to_del
#         npy_preprocess = np.delete(npy_preprocess, dfs_landmark_id, 1)

print(npy_preprocess)
print(npy_preprocess.shape)
print("NaN? ", np.isnan(np.min(npy_preprocess)))
print("Zeros: ", np.count_nonzero(npy_preprocess==0))

# for dfs_landmark_id in range(len(dfs_result)):
#     print("Landmark: ", dfs_result[dfs_landmark_id])
#     print(npy_preprocess[:, dfs_landmark_id, :])


    # print(selected_rows)

    # df_frame.drop([1, 3])
    
    # for index, row in df_frame.iterrows():
    #     if row['A'] >= 5:
    #         filtered_df = filtered_df.append(row)

    # print(df_frame.landmark_index)



# print(df_train[df_train.index == 0])
# df_train[]

