import os
from os import listdir
from os.path import isfile, join, isdir
import shutil

import numpy as np
import math
def list_files(folder_path):
    return [join(folder_path, f) for f in listdir(folder_path) if isfile(join(folder_path, f))]

def list_dirs(folder_path):
	return [join(folder_path, f) for f in listdir(folder_path) if isdir(join(folder_path, f))]
output_dir='../../data_processed/test_corpus/'
entropy_folder_path = "../../data_processed/entropy_files/"
for folder_path in list_dirs(entropy_folder_path):
        all_files = list_files(folder_path)
        temp_dir=''
        print(folder_path)
        temp_dir=output_dir+folder_path.split('/')[-1]+'/'
        print(temp_dir)
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        num_files=len(all_files)

        test_num_files=0.2*(num_files)
        test_num_files=math.floor(test_num_files)
        print(test_num_files)
        rand_indices = np.random.randint(0,high=len(all_files),size=test_num_files)
        for rand_idx in rand_indices:
            file_path = all_files[rand_idx]

            if isfile(file_path):
                shutil.move(file_path, temp_dir+ file_path.split("/")[-1])
                #shutil.move(file_path,"../test_corpus/" + file_path.split("/")[-1])
