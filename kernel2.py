
# coding: utf-8

# In[1]:


'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''
import tensorflow as tf
#from __future__ import print_function
from tensorflow.python.client import device_lib
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import os
import numpy as np
import random as rn
import tensorflow as tf
import pandas as pd
import time
import concurrent.futures
from keras.backend.tensorflow_backend import set_session


# In[2]:


def train(x_train,y_train,x_test, real_x_test, real_y_test,batch_size,num_classes,epochs,input_shape):
    # input image dimensions
#     import os
#     os.environ["CUDA_VISIBLE_DEVICES"]="0"    
    config = tf.ConfigProto(
        gpu_options=tf.GPUOptions(
            visible_device_list="0", # specify GPU number
            allow_growth=True
        )
    )
    set_session(tf.Session(config=config))
    # convert class vectors to binary class matrices
    if K.image_data_format() == 'channels_first':x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    else: x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    x_test = x_test.astype('float32')
    x_test /= 255
    
    y_train = keras.utils.to_categorical(y_train, num_classes)
    real_y_test = keras.utils.to_categorical(real_y_test, num_classes)
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=0,
              validation_data=(real_x_test, real_y_test))
    predict_score = model.predict(x_test, verbose=0)
    real_score = model.evaluate(real_x_test, real_y_test, verbose=0)
    model.save('./model.h5', include_optimizer=False)
    print('Test loss:', real_score[0])
    print('Test accuracy:', real_score[1])
    time.sleep(1)
    return predict_score,real_score


# In[1]:


def train2(x_train,y_train,x_test, real_x_test, real_y_test,batch_size,num_classes,epochs,input_shape):
    # input image dimensions
#     import os
#     os.environ["CUDA_VISIBLE_DEVICES"]="1"
    config = tf.ConfigProto(
        gpu_options=tf.GPUOptions(
            visible_device_list="0", # specify GPU number
            allow_growth=True
        )
    )
    set_session(tf.Session(config=config))
    #print(device_lib.list_local_devices()) 
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(0)
    rn.seed(0)
    
    # convert class vectors to binary class matrices
    if K.image_data_format() == 'channels_first':x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    else: x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    x_test = x_test.astype('float32')
    x_test /= 255
    
    y_train = keras.utils.to_categorical(y_train, num_classes)
    real_y_test = keras.utils.to_categorical(real_y_test, num_classes)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=0,
              validation_data=(real_x_test, real_y_test))
    predict_score = model.predict(x_test, verbose=0)
    real_score = model.evaluate(real_x_test, real_y_test, verbose=0)
    model.save('./model.h5', include_optimizer=False)
    print('Test loss:', real_score[0])
    print('Test accuracy:', real_score[1])
    time.sleep(1)
    return predict_score,real_score


# In[4]:


def active_learning(_min,x_train,y_train,c,batch_size,num_classes,epochs,input_shape):
    min_index = []
    for i in sorted(_min)[:c]:
        min_index.append(_min.index(i))
    x_train = x_train[:30000]
    y_train = y_train[:30000]
    for n in min_index: 
        list(x_train).append(x_train[n])
        list(x_train).append(y_train[n])
        
    predict_score,inflations_real_score =train2(x_train,y_train,x_train[30001:] , real_x_test, real_y_test,batch_size,num_classes,epochs,input_shape)
    return inflations_real_score

def create_data(predict):
    _min = []
    for i,l in enumerate(predict):
        sort_predict = sorted(l, reverse=True)
        _min.append(abs(sort_predict[0] - sort_predict[1]))
    return _min


# In[5]:


def inflations(x_train,y_train,real_x_test,real_y_test,batch_size,num_classes,epochs,input_shape):    
    for i,l in enumerate(y_train[30000:]):
        predict_score,inflations_real_score = train(x_train[:30000+i],y_train[:30000+i],x_train[30001:] , real_x_test, real_y_test,batch_size,num_classes,epochs,input_shape)
        i_score.append(inflations_real_score)
        iscoredf = pd.DataFrame({ 'inflation': i_score})
        iscoredf.to_csv('./iscoredf.csv')
        time.sleep(1)
def margins(predict_score,x_train,y_train,real_x_test,real_y_test,batch_size,num_classes,epochs,input_shape):
    for i,l in enumerate(y_train[30000:]):
        if i == 0:_min = create_data(predict_score)
        al_score.append(active_learning(_min,x_train,y_train,i,batch_size,num_classes,epochs,input_shape))
        alscoredf = pd.DataFrame({'al': al_score})
        alscoredf.to_csv('./alscoredf.csv')
        time.sleep(1)


# In[6]:
import threading
if __name__ == "__main__": 
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--batch_size", default = 20, type = int)
    args = parser.parse_args()
        # input image dimensions
    img_rows, img_cols = 28, 28

    batch_size = 128 * args.batch_size
    num_classes = 10
    epochs = 12

    # the data, split between train and test sets
    (x_train, y_train), (real_x_test, real_y_test) = mnist.load_data()


    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        real_x_test = real_x_test.reshape(real_x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        real_x_test = real_x_test.reshape(real_x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    real_x_test = real_x_test.astype('float32')
    x_train /= 255
    real_x_test /= 255
    
    i_score = []
    al_score = []
    predict_score,inflations_real_score = train(x_train[:30000],y_train[:30000],x_train[30001:] , real_x_test, real_y_test,batch_size,num_classes,epochs,input_shape)
#     thread_1 = threading.Thread(target=inflations(x_train,y_train,real_x_test,real_y_test))
#     thread_2 = threading.Thread(target=margins(predict_score,x_train,y_train,real_x_test,real_y_test))
    #executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
    #executor.submit(inflations(x_train,y_train,real_x_test,real_y_test, batch_size,num_classes,epochs,input_shape))
    #executor.submit(margins(predict_score,x_train,y_train,real_x_test,real_y_test, batch_size,num_classes,epochs,input_shape))
    margins(predict_score,x_train,y_train,real_x_test,real_y_test, batch_size,num_classes,epochs,input_shape)
