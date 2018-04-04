from __future__ import print_function
from keras.layers import Input, LSTM, RepeatVector,Bidirectional
from keras.models import Model
#from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import h5py
from keras.models import model_from_json
from os import listdir
from os.path import isfile, join, isdir
from keras.models import load_model
import os

def list_files(folder_path):
    return [join(folder_path, f) for f in listdir(folder_path) if isfile(join(folder_path, f))]

def get_arr(f_path):
    x = []
    for l in open(f_path):
        x.append(l.strip())
    return np.array(x, dtype=float)

def train():
    


    file_id = 1
    count=1
    entropy_folder_path = "../data_processed/train/"
    for file_path in list_files(entropy_folder_path):

        X_train = get_arr(file_path)
        file_id = file_id + 1
        print('shas')
        timesteps=5
        input_dim=1
        latent_dim=128
        print('desai')
        inputs = Input(shape=(timesteps, input_dim),batch_shape=(1,timesteps, input_dim))
        encoded = LSTM(latent_dim,dropout=0.2,stateful=True,name='encoder')(inputs)

        print('ghfsdk')
        decoded = RepeatVector(timesteps)(encoded)
        decoded =(LSTM(input_dim,return_sequences=True,stateful=True,name='decoder'))(decoded)

        sequence_autoencoder = Model(inputs, decoded)
        encoder = Model(inputs, encoded)
        examples=timesteps
        print('jkf')

        X_train=np.array(X_train)

        X_train=np.reshape(X_train,(-1,1))
        scaler = MinMaxScaler(feature_range=(0, 1))
        X_train = scaler.fit_transform(X_train)




        print('fefr')
        X_train=np.reshape(X_train,[(len(X_train)),1])

        nb_samples=len(X_train)-timesteps-1#1 for y_examples=1 in our case,;i.i hout=1
        input_list = [np.expand_dims(np.atleast_2d(X_train[i:examples+i,:]), axis=0) for i in range(nb_samples)]
        input_mat = np.concatenate(input_list, axis=0)

        sequence_autoencoder.compile(loss='mse', optimizer='Adam',metrics=['mse'])
        b=1

        history=sequence_autoencoder.fit(input_mat,input_mat,nb_epoch=15,batch_size=b,shuffle=False)

        sequence_autoencoder.summary()
        sequence_autoencoder.save_weights("../weights/weight" + str(count) + ".h5")
        print("Saved model to disk")
        count=count+1
        sequence_autoencoder.reset_states()
train()