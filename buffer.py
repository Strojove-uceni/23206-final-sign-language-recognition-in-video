from processing import ParquetProcess
import pandas as pd
import numpy as np
import os

class ParquetData:
  def __init__(self):
    self.data = {}

  def read_all(self, path, df_list, landmark_id, max_length = 537):
    for participant_folder in os.listdir(path + "/train_landmark_files/"):
      path_folder = path + "/train_landmark_files/" + participant_folder + "/"
      for file_name in os.listdir(path_folder):
        file_num = file_name.split(".")[0]
        df_row = df_list[(df_list.participant_id == int(participant_folder)) & (df_list.sequence_id == int(file_num))]
        file_path = path + "/train_landmark_files/" + participant_folder + "/" + file_num + ".parquet"
        sign_name = df_row.sign.values[0]
        if(not sign_name in list(self.data.keys())):
          self.data[sign_name] = {}
        self.data[sign_name][participant_folder] = ParquetProcess(file_path, landmark_id, max_length)
    print("Data read completed!")

path = "C:/Skoda_Digital/Materials/Documents_FJFI/SU2/asl-signs"
selected_landmark_indices = [33, 133, 159, 263, 46, 70, 4, 454, 234, 10, 338, 297, 332, 61, 291, 0, 78, 14, 317,
                             152, 155, 337, 299, 333, 69, 104, 68, 398]

df_train = pd.read_csv(path + "/train.csv", sep=",")
df_train.head()

data_load = ParquetData()
data_load.read_all(path, df_train, selected_landmark_indices)

print(data_load)
