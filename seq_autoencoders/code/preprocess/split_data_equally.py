import numpy as np
from os import listdir
from os.path import isfile, join, isdir
import os

def split_arr_into_files(entropy_vals, file_id, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    len_entropy_vals = len(entropy_vals)
    split_index = 0
    arr_index = 0
    #count=0
    while True:
        split_index += 1


        with open(output_dir + str(file_id) + "_" + str(split_index) + ".entropy", "w")  as f:
            #new_index = arr_index + int(0.02 * len_entropy_vals * (1 + rand_val))
            #print(new_index)
            #i=0
            new_index=arr_index+200
            for temp_val in entropy_vals[arr_index:min(new_index, len_entropy_vals)]:
                f.write(temp_val + "\n")
                #count=count+1

                #print(arr_index)
                #print(new_index)
                #i=i+1

        arr_index = new_index
        #print(arr_index)
        if arr_index >= len_entropy_vals:
            #print('count')
            #print(count)
            break



def list_files(folder_path):
    return [join(folder_path, f) for f in listdir(folder_path) if isfile(join(folder_path, f))]

def get_arr(f_path):
    x = []
    for l in open(f_path):
        x.append(l.strip())
    return np.array(x, dtype=float)
'''
def get_data(file_path):
    data = get_arr(file_path)
    l=len(data)
    #train = int(split_percentage * l)
    #print(l)
    #X_train = data[:train]
    #X_test = data[train:]
    #print(str(X_train[0]))
    return (X_train,X_test)
'''
entropy_file_index=0

with open("../file_index_map.txt", "w") as f:

        for file_path in list_files("../../../../../100k/"):
            entropy_file_index =entropy_file_index +1
            #print(entropy_file_index)
            f.write(str(entropy_file_index) + "," + file_path.split("/")[-1] + "\n")
            #X_train,X_test=get_data(file_path)
            #print(entropy_file_index)
            arr = get_arr(file_path)
            #output_dir='../../data_processed/train/'

            arr = arr.astype(str)
            print(len(arr))
            split_arr_into_files(arr, "p" + str(entropy_file_index),
                                 "../../data_processed/entropy_files/p" + str(entropy_file_index) + "/")
            print(file_path, " : complete")

            '''
            
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            with open(output_dir+ 'train'+str( entropy_file_index)  +  ".entropy", "w")  as f1:
                for i in range(np.shape(X_train)[0]):
                    f1.write(str(X_train[i]) + "\n")
                #print(entropy_file_index)
                #print('shas')
            output_dir = '../../data_processed/test/'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
           #print('ko')
            with open(output_dir +'test' + str( entropy_file_index) + ".entropy", "w")  as f2:
                for i in range(len(X_test)):
                    f2.write(str(X_test[i]) + "\n")
                #print(entropy_file_index)



    #file_id=1
    #entropy_folder_path = "/media/epsilon90/Shasvat/MS Sem2/CMPSCI 696C/data/files"
    
            '''
