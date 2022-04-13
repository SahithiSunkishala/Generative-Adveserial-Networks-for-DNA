#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 12:39:36 2022

@author: sahithis
"""

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
# example of training a gan on mnist
from numpy import expand_dims
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
    # embedding for categorical input
    li = Embedding(input_dim=n_classes,output_dim=50,trainable=True)(in_label)
    # scale up to image dimensions with linear activation
    n_nodes = in_shape[0] * in_shape[1]
    li = Dense(n_nodes)(li)
    # reshape to additional channel
    li = Reshape((in_shape[0], in_shape[1]))(li)
    #input
    in_image = Input(shape=in_shape)
    # concat label as a channel
    merge = Concatenate()([in_image, li])
    fe = Conv1D(64, 2, padding='same')(merge)
    fe = LeakyReLU(alpha=0.1)(fe)
    fe = Conv1D(64, 2, padding='same')(fe)
    fe = LeakyReLU(alpha=0.1)(fe)
    # flatten feature maps
    fe = Flatten()(fe)
    # dropout
    fe = Dropout(0.2)(fe)
    # output
    out_layer = Dense(1, activation='sigmoid')(fe)
    # define model
    model = Model([in_image, in_label], out_layer)
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

def define_generator(latent_dim, n_classes=10):
    # label input
    in_label = Input(shape=(1,))
    # embedding for categorical input
    li = Embedding(n_classes, latent_dim,trainable=True)(in_label)
   
    # linear multiplication
    n_nodes = 201*4
    li = Dense(n_nodes)(li)
    li = Reshape((201,4))(li)
    in_lat = Input(shape=(latent_dim,))
    gen = Dense(n_nodes)(in_lat)
    gen = LeakyReLU(alpha=0.2)(gen)
    gen = Dense(n_nodes)(gen)
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
    model = Model([gen_noise, gen_label], gan_output)
    # compile model
    opt = Adam()
    model.compile(loss='binary_crossentropy', optimizer=opt)
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
    ix = np.sort(randint(0,xdata.shape[0],n_samples))
    # select images and labels
    (X, labels) = (xdata[ix], labels[ix])
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


# scale an array of images to a new size
def scale_images(images, new_shape):
	images_list = list()
	for image in images:
		# resize with nearest neighbor interpolation
		new_image = resize(image, new_shape, 0)
		# store
		images_list.append(new_image)
	return asarray(images_list)

# calculate frechet inception distance
def calculate_fid(model, images1, images2):
    images1 = images1.astype('float32')
    images2 = images2.astype('float32')
    images1 = scale_images(images1, (299,299,3))
    images2 = scale_images(images2, (299,299,3))
    #print('Scaled', images1.shape, images1.shape)
    # pre-process images
    images1 = preprocess_input(images1)
    images2 = preprocess_input(images2)
	# calculate activations
    act1 = model.predict(images1)
    act2 = model.predict(images2)
	# calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
	# calculate sum squared difference between means
    ssdiff = numpy.sum((mu1 - mu2)**2.0)
	# calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
	# check and correct imaginary numbers from sqrt
    if iscomplexobj(covmean):
        covmean = covmean.real
	# calculate score
    fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid
# prepare the inception v3 model
model = InceptionV3(include_top=False, pooling='avg', input_shape=(299,299,3))


# train the generator and discriminator
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=100, n_batch=100,n_classes=2):
    
    #bat_per_epo = int(dataset.shape[0] / n_batch)
    half_batch = int(n_batch / 2)
    d_real_loss_epoch=[]
    d_fake_loss_epoch=[]
    g_loss_epoch=[]
    fid_epoch = []
    
    half_batch = int(n_batch / 2)
    # manually enumerate epochs
    for i in range(n_epochs):
            # get randomly selected 'real' samples
            [X_real, labels_real], y_real = generate_real_samples(dataset, half_batch)
            # update discriminator model weights
            d_loss1, _ = d_model.train_on_batch([X_real, labels_real], y_real)
            # generate 'fake' examples
            [X_fake, labels], y_fake = generate_fake_samples(g_model, latent_dim, half_batch,n_classes)
            # update discriminator model weights
            d_loss2, _ = d_model.train_on_batch([X_fake, labels], y_fake)
            
            # fid between images1 and images1
            #fid = calculate_fid(model, X_real, X_fake)
            #print(fid)
            # prepare points in latent space as input for the generator
            [z_input, labels_input] = generate_latent_points(latent_dim, n_batch,n_classes)

            '''
            want the generator to create images to fool the discriminator 
            meaning increases discriminator loss on fake images with class labels 0
            same as decreasing discriminator loss on fake images with class labels 1
            Thats why create class labels 1 for the fake images (instead of 0)
            '''
            y_gan = ones((n_batch, 1))
            #if(i==0 and j==0):
            # (acc_real, acc_fake)=summarize_performance(i, g_model, d_model, dataset, latent_dim)
            g_loss = gan_model.train_on_batch([z_input, labels_input], y_gan)
            g_loss_epoch.append(g_loss)
            d_real_loss_epoch.append(d_loss1)
            d_fake_loss_epoch.append(d_loss2)
            #fid_epoch.append(fid) 
            #save the generator model every 10 epochs
            if (i+1) % 10 == 0:
                print('>Loss %d,loss_real=%.3f, loss_fake=%.3f, loss_gan=%.3f'  % (i+1,d_loss1, d_loss2, g_loss))   
                filename = 'cgan_model_%03d.h5' % (i + 1)
                g_model.save(filename)
    return (g_loss_epoch,d_real_loss_epoch,d_fake_loss_epoch)    





##### Reading data

X_train,X_val,X_test,Y_train,Y_val,Y_test = get_data("human_positive_seq.fa", "human_negative_seq.fa")


####training

#gan_model
d_model = define_discriminator((201,4))
print(d_model.summary())
latent_dim = 100
g_model = define_generator(latent_dim, n_classes = 2)
print(g_model.summary())
gan_model = define_gan(g_model, d_model)
print(gan_model.summary())


#train
#train
(g_loss_epoch,d_real_loss_epoch,d_fake_loss_epoch) = train(g_model, d_model, gan_model, (X_train,Y_train), latent_dim,n_epochs=700, n_batch=10000,n_classes=2)

import pandas as pd
loss1 = pd.DataFrame({'fake' : d_fake_loss_epoch, 'real' : d_real_loss_epoch})




headerList = ['fakeloss_x' ,'fakeloss_y' , 'realloss_x', 'realloss_y']

loss1.to_csv('Loss1.csv')

      


import matplotlib.pyplot as plt
plt.figure() 
loss1.plot()
plt.ylim([0, 1.5])
plt.axhline(y=0.693, color='k', linestyle='-')
plt.legend(loc='best')

