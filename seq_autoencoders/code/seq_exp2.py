from __future__ import print_function
from keras.layers import Input, LSTM, RepeatVector, Bidirectional
from keras.models import Model
# from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import h5py
from keras.models import model_from_json
from os import listdir
from os.path import isfile, join, isdir
from keras.models import load_model
import os

timesteps = 200
input_dim = 1
latent_dim = 256
def list_files(folder_path):
    return [join(folder_path, f) for f in listdir(folder_path) if isfile(join(folder_path, f))]

def list_dirs(folder_path):
	return [join(folder_path, f) for f in listdir(folder_path) if isdir(join(folder_path, f))]

def get_arr(f_path):
    x = []
    for l in open(f_path):
        x.append(l.strip())
    #return x
    return np.array(x, dtype=float)
def get_model():
    timesteps = 200
    input_dim = 1
    latent_dim = 256
    inputs = Input(shape=(timesteps, input_dim), batch_shape=(batch_size, timesteps, input_dim))
    encoded = LSTM(latent_dim, dropout=0.2,return_sequences=True, name='encoder1')(inputs)
    encoded = LSTM(latent_dim, dropout=0.2,return_sequences=False, name='encoder2')(encoded)
	
    #print('ghfsdk')
    decoded = RepeatVector(timesteps)(encoded)
    decoded = (LSTM(input_dim, return_sequences=True, name='decoder'))(decoded)

    sequence_autoencoder = Model(inputs, decoded)
    encoder = Model(inputs, encoded)
    return sequence_autoencoder



def getBatchData(X_train,num_samples,timesteps,batch_size=64):
    num_batches = int(num_samples / batch_size)
    # print("num_batches", num_batches)

    for i in range(num_batches):
        #X_batch = np.zeros([batch_size, timesteps, 1])
        #y_batch = np.zeros([batch_size, 1])
        batch_index = i * batch_size
        X_batch=X_train[batch_index:batch_index+batch_size]
            # print("X_samples ", X_samples[i])
            #y_batch[j] = series_dataframe_narray_norm[batch_index + split_length, 0]
        # print("xbatch", X_batch)
        # print("ybatch", y_batch)\
        X_batch=X_batch(...,np.newaxis())
        yield X_batch




batch_size=512
def train():
    entropy_folder_path = "../data_processed/entropy_files/"
    weight_path = '../weights/'
    count=1
    iter=0
    loss=0

    sequence_autoencoder = get_model()
    while iter<5000:

        for folder_path in list_dirs(entropy_folder_path):
            x=[]
            all_files=list_files(folder_path)
            rand_indices = np.random.randint(0, high=len(all_files), size=batch_size)
            for rand_idx in rand_indices:
                file_path = all_files[rand_idx]

                X_temp = get_arr(file_path)
                X_temp = np.reshape(X_temp, (1, X_temp.shape[0]))
                x.append(X_temp)
                #print(len(x))
                # file_id = file_id + 1
            X_train = np.concatenate(x, axis=0)

            #X_batch= getBatchData(X_train,nb_samples ,timesteps, batch_size)

            #X_train = np.reshape(X_train, [(len(X_train)), 1])

            #nb_samples = len(X_train) - timesteps - 1  # 1 for y_examples=1 in our case,;i.i hout=1


            #input_list = [np.expand_dims(np.atleast_2d(X_train[i:examples + i, :]), axis=0) for i in range(nb_samples)]

            #input_mat = np.concatenate(input_list, axis=0)

            '''
            if iter%100==0:

                if os.path.exists(weight_path):
                    sequence_autoencoder.load_weights('../weights/weight' + str(count) + '.h5', by_name=True)
                    #count = count + 1
            '''
            # X_train = np.array(x)
            #print('model loaded')
            #print(X_train.shape)
            # X_train = np.reshape(X_train, (1, -1))
            scaler = MinMaxScaler(feature_range=(0, 1))
            X_train = scaler.fit_transform(X_train)
            X_train = X_train[..., np.newaxis]
            # nb_samples = X_train.shape[0]
            sequence_autoencoder.compile(loss='mse', optimizer='Adam', metrics=['mse'])
            #b = 1
            print('iter: ')
            print(iter)
            history = sequence_autoencoder.fit(X_train, X_train,  batch_size=batch_size, shuffle=True)

            #sequence_autoencoder.summary()
        if iter%100==0:

            if not os.path.exists(weight_path):
                os.makedirs(weight_path)

            sequence_autoencoder.save_weights("../weights/weight" + str(count) + ".h5")
            print("Saved model to disk")
            count = count + 1
        #sequence_autoencoder.reset_states()
        iter=iter+1

train()
