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


def get_arr(f_path):
    x = []
    for l in open(f_path):
        x.append(l.strip())
    return np.array(x, dtype=float)
count=1
entropy_folder_path = "../data_processed/test"
result = np.zeros(128)
def test():

    print('test')
    for file_path in list_files(entropy_folder_path):
        #break
        X_test = get_arr(file_path)
        #file_id = file_id + 1





        timesteps = 10
        input_dim = 1
        latent_dim = 128


        inputs = Input(shape=(timesteps, input_dim), batch_shape=(1, timesteps, input_dim))
        encoded = LSTM(latent_dim, dropout=0.2, stateful=True, name='encoder')(inputs)
        encoder = Model(inputs, encoded)
        encoder.load_weights('../weights/weight'+str(count)+'.h5',by_name=True)


        X_test = np.array(X_test)
        #print( X_test.shape)
        scaler = MinMaxScaler(feature_range=(0, 1))
        X_test = scaler.fit_transform(X_test)

        X_test = np.reshape(X_test, [(len(X_test)), 1])

        examples = timesteps

        nb_samples = len(X_test) - timesteps - 1  # 1 for y_examples=1 in our case,;i.i hout=1
        input_list = [np.expand_dims(np.atleast_2d(X_test[i:examples + i, :]), axis=0) for i in range(nb_samples)]
        input_mat = np.concatenate(input_list, axis=0)

        encoder.compile(loss='mse', optimizer='Adam', metrics=['mse'])
        y=encoder.predict(input_mat,batch_size=1)
        
        y=np.mean(y,axis=0)
        #print(y.shape)

        result=np.vstack((result,y))
        count=count+1
        encoder.reset_states()

        
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

