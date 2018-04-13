from __future__ import print_function
from keras.layers import Input, LSTM, RepeatVector, Bidirectional
from keras.models import Model


#from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import h5py
from keras.models import model_from_json
from os import listdir
from os.path import isfile, join, isdir
from keras.models import load_model

def list_files(folder_path):
    return [join(folder_path, f) for f in listdir(folder_path) if isfile(join(folder_path, f))]

def list_dirs(folder_path):
	return [join(folder_path, f) for f in listdir(folder_path) if isdir(join(folder_path, f))]
def get_arr(f_path):
    x = []
    for l in open(f_path):
        x.append(l.strip())
    return np.array(x, dtype=float)

count=0
batch_size=256
entropy_folder_path = "../data_processed/subset/test_corpus/"
result = np.zeros(256)
def test():
    weight_path='../weights/'
    #sequence_autoencoder_model
    for folder_path in list_dirs(entropy_folder_path):
        x=[]
        #all_files = list_files(folder_path)
        print('test')
        for file_path in list_files(folder_path):
            #break
            X_temp = get_arr(file_path)
            #file_id = file_id + 1


            X_temp = get_arr(file_path)
            if(X_temp.shape[0]!=200):
                continue

            X_temp = np.reshape(X_temp, (1, X_temp.shape[0]))
            x.append(X_temp)
            print(X_temp.shape)
            # print(len(x))
            # file_id = file_id + 1
        print(len(x))
        print(len(x[0][0]))
        print(len(x[1][0]))
        X_test = np.concatenate(x, axis=0)
        print(X_test.shape)
        batch_size=len(x)


        timesteps = 200
        input_dim = 1
        latent_dim = 256


        inputs = Input(shape=(timesteps, input_dim), batch_shape=(batch_size, timesteps, input_dim))
        encoded = LSTM(latent_dim, dropout=0.2, stateful=False, name='encoder')(inputs)
        encoder = Model(inputs, encoded)
        encoder.load_weights('../weights/weight'+str(100)+'.h5',by_name=True)


        X_test = np.array(X_test)
        #print( X_test.shape)
        scaler = MinMaxScaler(feature_range=(0, 1))
        X_test = scaler.fit_transform(X_test)
        X_test=X_test[...,np.newaxis]
        #X_test = np.reshape(X_test, [(len(X_test)), 1])


        encoder.compile(loss='mse', optimizer='Adam', metrics=['mse'])
        y=encoder.predict(X_test,batch_size=batch_size)

        y=np.mean(y,axis=0)
        #print(y.shape)

        result=np.vstack((result,y))
        count=count+1
        #encoder.reset_states()


    result=result[1:]
    #print(result.shape)
    np.save('../result/latent_dims',result)


    '''
    sequence_autoencoder.layers.pop()
    sequence_autoencoder.layers.pop()
    hidden=sequence_autoencoder.predict(input_mat,batch_size=b)
    
    print(len(sequence_autoencoder.layers))
    print(sequence_autoencoder.summary())
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.show()
    
    
    
    #encoded = LSTM(latent_dim,dropout=0.2,stateful=True)(inputs)
    #encoder = Model(inputs, encoded)
    #encoder.compile(loss='mse', optimizer='Adam',metrics=['mse'])
    #encoder.fit(input_mat, latent_dim, nb_epoch=10, batch_size=1)
    
    
    #hidden=encoder.reset_states()
    result.append(hidden)
    print('hidden')
    print(hidden.shape)
    '''
    # print_structure('/media/epsilon90/Shasvat/MS Sem2/CMPSCI 696C/seqtoseq/weights')
    # print(sequence_autoencoder.layers)
    '''
    model_json=sequence_autoencoder.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    '''

test()