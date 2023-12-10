from processing import ParquetProcess
import pandas as pd
import numpy as np
import os
import shutil

class ParquetData:
  def __init__(self):
    self.data = {}

  def read_all_parquet(self, path, df_list, landmark_id):
    for participant_folder in os.listdir(path + "/train_landmark_files/"):
      path_folder = path + "/train_landmark_files/" + participant_folder + "/"
      for file_name in os.listdir(path_folder):
        file_num = file_name.split(".")[0]
        df_row = df_list[(df_list.participant_id == int(participant_folder)) & (df_list.sequence_id == int(file_num))]
        file_path = path + "/train_landmark_files/" + participant_folder + "/" + file_num + ".parquet"
        sign_name = df_row.sign.values[0]
        sequence_name = df_row.sequence_id.values[0]
        if(not sign_name in list(self.data.keys())):
          self.data[sign_name] = {}
        if(not participant_folder in list(self.data[sign_name].keys())):
          self.data[sign_name][participant_folder] = {}
        self.data[sign_name][participant_folder][sequence_name] = ParquetProcess(file_path, landmark_id)
    print("Data read completed!")

  def read_all_tensor(self, path):
    for sign_name in os.listdir(path + "/tensors/"):
      path_sign = path + "/tensors/" + sign_name + "/"
      for file_name in os.listdir(path_sign):
        if(not sign_name in list(self.data.keys())):
          self.data[sign_name] = {}
        self.data[sign_name][file_name.split(".")[0]] = np.load(path_sign + file_name, allow_pickle=False)
    print("Data read completed!")

  def preprocess_all(self, path, df_list, landmark_id, sign_list):
    """
    Reads all parquet files in directory, shorter than max_length, optionally sorts and saves numpy tensors made of parquet.
    :param path: Path to data folder
    :param df_list: CSV file linked to data
    :param landmark_id: Selected landmark indices used for tensor creation
    :param max_length: Maximum length in frames
    :param SortSave: Optional, set True if sorting and saving tensor is desired
    :return: Loaded data in dictionary, optionally saved numpy tensors sorted, based on sign name.
    """
    count=0
    for participant_folder in os.listdir(path + "/train_landmark_files/"):
      path_folder = path + "/train_landmark_files/" + participant_folder + "/"
      path_tensor= path + "/tensors"
      for file_name in os.listdir(path_folder):
        count=count+1
        file_num = file_name.split(".")[0]
        df_row = df_list[(df_list.participant_id == int(participant_folder)) & (df_list.sequence_id == int(file_num))]
        file_path = path + "/train_landmark_files/" + participant_folder + "/" + file_num + ".parquet"
        sign_name = df_row.sign.values[0]
        if sign_name in sign_list:
          readed_data = ParquetProcess(file_path, landmark_id)
          self.save_tensor(path_tensor, sign_name, readed_data.tensor, count)
    print("Data preprocess completed!")

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

    file_path = os.path.join(sign_dir, sign + "_" + str(count) + '_tensor.npy')

    np.save(file_path, tensor, allow_pickle=False)
    print(f"Tensor saved successfully at {file_path}")



# path = r"E:\asl-signs"
path = "C:/Skoda_Digital/Materials/Documents_FJFI/SU2/asl-signs"
selected_landmark_indices = [33, 133, 159, 263, 46, 70, 4, 454, 234, 10, 338, 297, 332, 61, 291, 0, 78, 14, 317,
                             152, 155, 337, 299, 333, 69, 104, 68, 398]

signs=['man', 'sad', 'airplane', 'bad']
df_train = pd.read_csv(path + "/train_mod.csv", sep=",")
df_train.head()

data_load = ParquetData()
data_load.preprocess_all(path, df_train, selected_landmark_indices, signs)
# data_load.read_all_parquet(path, df_train, selected_landmark_indices, 140)


