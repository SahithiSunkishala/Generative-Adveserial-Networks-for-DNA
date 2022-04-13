# -- coding: utf-8 --
"""
Created on Tue Jan 18 12:39:36 2022

@author: sahithis
"""

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
# example of training a gan on mnist
from numpy import expand_dims
from utils import *
from numpy import zeros
from numpy import ones
from numpy import vstack
from numpy.random import randn
from numpy.random import randint
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Convolution1D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.models import Model
import numpy
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy import asarray
from scipy.linalg import sqrtm
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.datasets.mnist import load_data
from skimage.transform import resize

from matplotlib import pyplot

# define the standalone discriminator model
from tensorflow.keras.layers import Embedding
import numpy as np


def define_discriminator(in_shape=(201,4),n_classes = 2):
    # label input
    in_label = Input(shape=(1,))
    n_nodes = in_shape[0] * in_shape[1]
    in_image = Input(shape=in_shape)
    fe = Conv1D(64, 5, padding='same')(in_image)
    fe = LeakyReLU(alpha=0.1)(fe)
    fe = Conv1D(64, 5, padding='same')(fe)
    fe = LeakyReLU(alpha=0.1)(fe)
    # flatten feature maps
    fe = Flatten()(fe)
    # dropout
    fe = Dropout(0.2)(fe)
    # output
    out_layer1 = Dense(1, activation='sigmoid')(fe)
    out_layer2 = Dense(1, activation='sigmoid')(fe)
    # define model
    model = Model(inputs = [in_image,in_label], outputs =  [out_layer1,out_layer2])
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss=['binary_crossentropy','binary_crossentropy'], optimizer=opt)
    return model

def define_generator(latent_dim, n_classes=2):
    # label input
    in_label = Input(shape=(1,))
    # embedding for categorical input
    li = Embedding(n_classes, latent_dim,trainable=True)(in_label)
    # linear multiplication
    li = Dense(128)(li)
    n_nodes = 201*4
    li = Dense(n_nodes)(li)
    li = Reshape((201,4))(li)
    in_lat = Input(shape=(latent_dim,))
    gen = Dense(n_nodes)(in_lat)
    gen = LeakyReLU(alpha=0.2)(gen)
    gen = Reshape((201,4))(gen)
    merge = Concatenate()([gen, li])
    out_layer = Conv1D(4, 2, activation='tanh', padding='same')(merge)
    model = Model([in_lat, in_label], out_layer)
    return model




# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model):
    # make weights in the discriminator not trainable
    d_model.trainable = False
    # get noise and label inputs from generator model
    gen_noise, gen_label = g_model.input
    # get image output from the generator model
    gen_output = g_model.output
    # connect image output and label input from generator as inputs to discriminator
    gan_output = d_model([gen_output, gen_label])
    # define gan model as taking noise and label and outputting a classification
    model = Model(inputs = [gen_noise, gen_label], outputs = gan_output)
    # compile model
    opt = Adam()
    model.compile(loss=['binary_crossentropy','binary_crossentropy'], optimizer=opt)
    return model


'''
randomly selected n_samples 
'''
# select real samples
def generate_real_samples(dataset, n_samples):
    # split into images and labels
    #dataset = (X_train, Y_train)
    xdata, labels = dataset
    # choose random instances
    ix = (randint(0,xdata.shape[0],n_samples))
    # select images and labels
    X, labels = xdata[ix], labels[ix]
    # generate class labels
    y = ones((n_samples, 1))
    return [X, labels], y


# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples, n_classes=10):
    # generate points in the latent space
    x_input = randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    z_input = x_input.reshape(n_samples, latent_dim)
    # generate labels
    labels = randint(0, n_classes, n_samples)
    return [z_input, labels]

# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n_samples,n_classes):
    # generate points in latent space
    z_input, labels_input = generate_latent_points(latent_dim, n_samples,n_classes)
    # predict outputs
    images = generator.predict([z_input, labels_input])
    images
    # create class labels
    y = zeros((n_samples, 1))
    return [images, labels_input], y

# example of calculating the frechet inception distance in Keras



# train the generator and discriminator
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=100, n_batch=100,n_classes=2):
    g_1_epoch = []
    g_2_epoch = []
    d_f1_epoch = []
    d_f2_epoch = []
    d_r1_epoch = []
    d_r2_epoch = []
    #bat_per_epo = int(dataset[0].shape[0] / n_batch)
	# calculate the number of training iterations
   # n_steps = bat_per_epo * n_epochs
    half_batch = int(n_batch / 2)

     # manually enumerate epochs
    for i in range(n_epochs):
            # get randomly selected 'real' samples
            [X_real, labels_real], y_real = generate_real_samples(dataset, half_batch)
            # update discriminator model weights
            _,d_r1,d_r2 = d_model.train_on_batch([X_real,labels_real] ,[y_real,labels_real])
            # generate 'fake' examples
            [X_fake, labels_fake], y_fake = generate_fake_samples(g_model, latent_dim, half_batch,n_classes)
            # update discriminator model weights
            _,d_f1,d_f2 = d_model.train_on_batch([X_fake,labels_fake], [y_fake,labels_fake])
            
            # prepare points in latent space as input for the generator
            [z_input, z_labels] = generate_latent_points(latent_dim, n_batch,n_classes)

            '''
            want the generator to create images to fool the discriminator 
            meaning increases discriminator loss on fake images with class labels 0
            same as decreasing discriminator loss on fake images with class labels 1
            Thats why create class labels 1 for the fake images (instead of 0)
            '''
            y_gan = ones((n_batch, 1))
            #if(i==0 and j==0):
            # (acc_real, acc_fake)=summarize_performance(i, g_model, d_model, dataset, latent_dim)
            _,g_1,g_2 = gan_model.train_on_batch([z_input, z_labels], [y_gan,z_labels])
            #print('>%d, dr[%.3f,%.3f], df[%.3f,%.3f], g[%.3f,%.3f]' % (i+1, d_r1,d_r2, d_f1,d_f2, g_1,g_2))
            g_1_epoch.append(g_1)
            g_2_epoch.append(g_2)
            d_f1_epoch.append(d_f1)
            d_f2_epoch.append(d_f2)
            d_r1_epoch.append(d_r1)
            d_r2_epoch.append(d_r2)
            #fid_epoch.append(fid) 
            #save the generator model every 10 epochs
            if (i+1) % 10 == 0:
                print('>%d, dr[%.3f,%.3f], df[%.3f,%.3f], g[%.3f,%.3f]' % (i+1, d_r1,d_r2, d_f1,d_f2, g_1,g_2))
                #print('>Loss %d,loss_real=%.3f, loss_fake=%.3f, loss_gan=%.3f, FID= %.3f'  % (i+1,d_loss1, d_loss2, g_loss,fid))   
                filename = 'cgan_model_%03d.h5' % (i + 1)
               
                g_model.save(filename)
    return (g_1_epoch,
            g_2_epoch,
            d_f1_epoch,
            d_f2_epoch,
            d_r1_epoch,
            d_r2_epoch)  


##### Reading data

X_train,X_val,X_test,Y_train,Y_val,Y_test = get_data("human_positive_seq.fa", "human_negative_seq.fa")


####training

#gan_model
d_model = define_discriminator((201,4))
print(d_model.summary())
latent_dim = 100
g_model = define_generator(latent_dim, n_classes = 2)
rollout = ROLLOUT(g_model, 0.8)
print(g_model.summary())
gan_model = define_gan(g_model, d_model)
print(gan_model.summary())


#train
(g_1_epoch,g_2_epoch,d_f1_epoch, d_f2_epoch, d_r1_epoch, d_r2_epoch) = train(g_model, d_model, gan_model, (positive_data), latent_dim,n_epochs=800, n_batch=10000,n_classes=2)

import pandas as pd
loss1 = pd.DataFrame({'fake' : d_f1_epoch, 'real' : d_r1_epoch})

loss2 = pd.DataFrame({'fake_class' : d_f2_epoch, 'real_class' : d_r2_epoch})

loss1.to_csv('Loss1.csv')
loss2.to_csv('Loss2.csv')
      


import matplotlib.pyplot as plt
plt.figure() 
loss1.plot()
plt.ylim([0, 1.5])
plt.axhline(y=0.693, color='k', linestyle='-')
plt.legend(loc='best')

plt.figure() 
loss2.plot()
plt.ylim([0, 1.5])
plt.axhline(y=0.693, color='k', linestyle='-')
plt.legend(loc='best')


import os,sys
import argparse
import h5py
import scipy.io
import numpy as np
from keras.optimizers import Adam
from keras.initializers import RandomUniform, RandomNormal, glorot_uniform, glorot_normal
from keras.models import Model
from keras.layers.core import  Dense, Dropout, Permute, Lambda
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras import regularizers
from keras.constraints import maxnorm
from keras.layers.recurrent import LSTM
from keras.layers import Bidirectional, Input
from keras.layers.merge import multiply
from keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow as tf
np.random.seed(12345)

'''Build the DeepLncCTCF model'''
def get_model(params):
    inputs = Input(shape = (201, 4,))
    cnn_out = Convolution1D(int(params['filter']), int(params['window_size']),
    	kernel_initializer=params['kernel_initializer'], 
    	kernel_regularizer=regularizers.l2(params['l2_reg']), 
    	activation="relu")(inputs)
    pooling_out = MaxPooling1D(pool_size=int(params['pool_size']), 
    	strides=int(params['pool_size']))(cnn_out)
    dropout1 = Dropout(params['drop_out_cnn'])(pooling_out)
    lstm_out = Bidirectional(LSTM(int(params['lstm_unit']), return_sequences=True, 
    	kernel_initializer=params['kernel_initializer'], 
    	kernel_regularizer=regularizers.l2(params['l2_reg'])), merge_mode = 'concat')(dropout1)
    a = Permute((2, 1))(lstm_out)
    a = Dense(lstm_out._keras_shape[1], activation='softmax')(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    attention_out = multiply([lstm_out, a_probs])
    attention_out = Lambda(lambda x: K.sum(x, axis=1))(attention_out)
    dropout2 = Dropout(params['drop_out_lstm'])(attention_out)
    dense_out = Dense(int(params['dense_unit']), activation='relu', 
    	kernel_initializer=params['kernel_initializer'], 
    	kernel_regularizer=regularizers.l2(params['l2_reg']))(dropout2)
    output = Dense(1, activation='sigmoid')(dense_out)
    model = Model(inputs=[inputs], outputs=output)
    adam = Adam(lr=params['learning_rate'],epsilon=10**-8)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=[roc_auc])
    return model


# load data model
Data = load_model('D:/Downloads/cgan_model_700.h5')

# generate fake data 20000
X,y = generate_fake_samples(Data, 100,30000,2)# generate 10 example images
X[0]=np.where(X[0] > 0,1,0)
#Create new X_train1 and y_train1
X_train1= np.r_[X_train,X[0]]
X_train1.shape

#Y_train = Y_train.reshape(72696,1)
y_train1 = np.r_[Y_train,X[1]]
y_train1.shape


best = {'batch_size': 4.0, 'dense_unit': 80.0, 'drop_out_cnn': 0.2738070724985381, 'drop_out_lstm': 0.16261503928101084, 'filter': 128.0, 'kernel_initializer': 'random_uniform', 'l2_reg': 1.0960198460047699e-05, 'learning_rate': 0.00028511592517082153, 'lstm_unit': 624.0, 'pool_size': 3.0, 'window_size': 9.0}
dnn_model = get_model(best)
filepa = "bestmodelnew_humanAC.hdf5"
checkpointer = ModelCheckpoint(filepath=filepa, verbose=1, save_best_only=True)
earlystopper = EarlyStopping(monitor='val_loss', patience=3, verbose=1)
dnn_model.fit(X_train1, y_train1, batch_size=2**int(best['batch_size']), epochs=100,shuffle=True, validation_data=(X_val,Y_val), callbacks=[checkpointer,earlystopper])
predictions = dnn_model.predict(X_test)
rounded = [round(x[0]) for x in predictions]
pred_train_prob = predictions
metrics(Y_test, rounded, pred_train_prob)
