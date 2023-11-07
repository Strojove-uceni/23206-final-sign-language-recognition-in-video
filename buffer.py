from processing import ParquetProcess
import pandas as pd
import numpy as np
import os

class ParquetData:
  def __init__(self):
    self.data = []

  def read_all(self, path, df_list):
    for participant_folder in os.listdir(path + "/train_landmark_files/"):
      path_folder = path + "/train_landmark_files/" + participant_folder + "/"
      for file_name in os.listdir(path_folder):
        file_num = file_name.split(".")[0]
        df_row = df_list[(df_list.participant_id == int(participant_folder)) & (df_list.sequence_id == int(file_num))]
        self.data.append(ParquetProcess(path + "/train_landmark_files/" + participant_folder + "/" + file_num + ".parquet", df_row.sign.values[0]))

    def clean_all(self):
        for parquet_processor in self.data:
            parquet_processor.clean_parquet(show_df=False)


path = "C:/Skoda_Digital/Materials/Documents_FJFI/SU2/asl-signs"

df_train = pd.read_csv(path + "/train.csv", sep=",")
df_train.head()

# Dictionary for data
data = ParquetData()
data.read_all(path, df_train)
data.clean_all()
