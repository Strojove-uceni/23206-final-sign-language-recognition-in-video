from ParquetToMatrix import ParquetToMatrix
import pandas as pd
import numpy as np
import os
import shutil
from dfs_results import dfs_result
from matplotlib import pyplot as plt
import torch
from torch.nn import functional as F


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
                # df = df.dropna()
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
                            if(readed_data.preprocess_succes):
                                self.save_tensor(path_matrix, sign_name, readed_data.concatenated_matrix, count)
                            else:
                                print('File preprocessing failed:: ', file_path)
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

# signs=['airplane', 'alligator', 'any', 'apple', 'balloon', 'bath',
#        'black', 'drink', 'drop', 'dry', 'duck', 'ear', 'empty', 'face', 'find', 'fine', 'finger', 'garbage', 'girl', 'goose']

signs=['bad', 'fall', 'not', 'go', 'can', 'wet', 'grandma', 'grandpa', 'look', 'no']
df_train = pd.read_csv(path + "/train_mod.csv", sep=",")

data_load = ParquetData()
data_load.preprocess_all(path, df_train, selected_landmark_indices, signs, 96)

#######################################################################################
# df_line = df_train[df_train.index == 0]
# path_file = df_line.path.values[0]
# parquet_file = pd.read_parquet(path + "/" + path_file)

# final_len = 40

# frames = parquet_file.frame.unique()
# npy_preprocess = np.empty((final_len, len(dfs_result), 3))
# npy_preprocess[:] = np.nan

# parquet_file = parquet_file.sort_values(by=['frame'])

# print(parquet_file)

# right_hand_nan_count = 0
# left_hand_nan_count = 0
# landmark_eyebrow_id = []
# for dfs_landmark_id in range(len(dfs_result)):
#     dfs_landmark = dfs_result[dfs_landmark_id]
#     (landmark, body_part) = dfs_landmark.split("-")
#     if(int(landmark) == 65 or int(landmark) == 295):
#         # store id of eyebrows
#         landmark_eyebrow_id += [dfs_landmark_id]
#     df_landmark = parquet_file[(parquet_file.landmark_index == int(landmark)) & (parquet_file.type == body_part)]
#     xyz_cord = np.array([df_landmark.x.values, df_landmark.y.values, df_landmark.z.values])
#     if(np.isnan(np.min(xyz_cord))):
#         if(dfs_landmark_id < 21):
#             right_hand_nan_count += np.sum(np.isnan(xyz_cord))
#         elif(dfs_landmark_id > 41):
#             left_hand_nan_count += np.sum(np.isnan(xyz_cord))

#         nan_mask = np.isnan(xyz_cord)
#         rows_without_nan = ~nan_mask.any(axis=0)
#         xyz_without_nan = xyz_cord[:, rows_without_nan]
#         if(xyz_without_nan.shape[1] == 0):
#             xyz_interp = F.interpolate(torch.tensor(xyz_cord)[None, :], size=final_len, mode='nearest').numpy()
#         else: 
#             xyz_interp = F.interpolate(torch.tensor(xyz_without_nan)[None, :], size=final_len, mode='nearest').numpy()
#             print(xyz_interp)
#         xyz_transform = np.squeeze(xyz_interp, axis=0).T
#     else:
#         xyz_interp = F.interpolate(torch.tensor(xyz_cord)[None, :], size=final_len, mode='nearest').numpy()
#         xyz_transform = np.squeeze(xyz_interp, axis=0).T

#     npy_preprocess[:, dfs_landmark_id, :] = np.copy(xyz_transform)

# if(len(landmark_eyebrow_id) == 2):
#     # norm_point = np.diff(np.array([npy_preprocess[:, landmark_eyebrow_id[0], :], npy_preprocess[:, landmark_eyebrow_id[1], :]]), axis = 0).squeeze(xyz_interp, axis=0)
#     norm_point = (npy_preprocess[:, landmark_eyebrow_id[0], :] + npy_preprocess[:, landmark_eyebrow_id[1], :])/2
# else:
#     print("Normalization failed!")
# print(norm_point)

# for dfs_landmark_id in range(len(dfs_result)):
#     # print(npy_preprocess[:, dfs_landmark_id, :].shape)
#     npy_preprocess[:, dfs_landmark_id, :] = npy_preprocess[:, dfs_landmark_id, :] - norm_point

# if(left_hand_nan_count < right_hand_nan_count):
#     npy_hand_removed = npy_preprocess[:, 21:64, :]
# else:
#     npy_hand_removed = npy_preprocess[:, 0:42, :]

# # print(npy_preprocess)
# print(right_hand_nan_count)
# print(left_hand_nan_count)
# print(npy_hand_removed[0,:,:])

# plt.imshow((npy_hand_removed))
# plt.show()
# print(npy_preprocess.shape)
# print("NaN? ", %np.isnan(np.min(npy_preprocess)))
# print("Zeros: ", np.count_nonzero(npy_preprocess==0))


