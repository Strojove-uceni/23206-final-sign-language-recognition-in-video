import pandas as pd
import os
import shutil


class SortParquets:

    def sort_parquet(self, path, destination_path, df_list, max_length):
        for participant_folder in os.listdir(path + "/train_landmark_files/"):
            path_folder = path + "/train_landmark_files/" + participant_folder + "/"
            for file_name in os.listdir(path_folder):
                file_num = file_name.split(".")[0]
                df_row = df_list[
                    (df_list.participant_id == int(participant_folder)) & (df_list.sequence_id == int(file_num))]
                file_path = path + "/train_landmark_files/" + participant_folder + "/" + file_num + ".parquet"
                if (df_row.length_frames.values[0] > max_length):
                    print('File was skipped:: ', file_path)
                    continue

                sign_name = df_row.sign.values[0]
                destination_folder = os.path.join(destination_path, sign_name)
                if not os.path.exists(destination_folder):
                    os.makedirs(destination_folder)

                # Copy the file
                shutil.copy(file_path, os.path.join(destination_folder, file_name))

                print('File copied to: ', os.path.join(destination_folder, file_name))

path=r'E:\asl-signs'
destination = r'E:\asl-signs\parquets_sorted'
df_train = pd.read_csv(path + "/train_mod.csv", sep=",")
sorting=SortParquets()
sorting.sort_parquet(path,destination, df_train,140)