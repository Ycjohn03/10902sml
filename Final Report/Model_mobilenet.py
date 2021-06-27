import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator,array_to_img,img_to_array,load_img
from tensorflow.keras.layers import Lambda, Dropout,Input, Activation, Dense, GlobalAveragePooling2D, Conv2D, Add, BatchNormalization,DepthwiseConv2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import optimizers,regularizers
from tensorflow.keras.initializers import he_normal
from tensorflow.keras.callbacks import Callback,LearningRateScheduler, TensorBoard, ModelCheckpoint, EarlyStopping,ReduceLROnPlateau

def depthwise_separable(x,params):
    # f1/f2 filter size, s1 stride of conv
    (s1,f2) = params
    alpha = 0.75
    x = DepthwiseConv2D((3,3),strides=(s1[0],s1[0]), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(int(alpha*f2[0]), (1,1), strides=(1,1), padding='same')(x)
    x = BatchNormalization()(x)    
    x = Activation('relu')(x)
    return x
	
#建立 MobileNet model

def MobileNet(img_input,shallow=False, classes=10):
    """Instantiates the MobileNet.Network has two hyper-parameters
        which are the width of network (controlled by alpha)
        and input size.
        # Arguments
            alpha: optional parameter of the network to change the 
                width of model.
            shallow: optional parameter for making network smaller.
            classes: optional number of classes to classify images
                into.
    """
    alpha = 0.75
    x = Conv2D(int(alpha*32), (3,3), strides=(2,2), padding='same')(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = depthwise_separable(x,params=[(1,),(64,)])
    x = depthwise_separable(x,params=[(2,),(128,)])
    x = depthwise_separable(x,params=[(1,),(128,)])
    x = depthwise_separable(x,params=[(2,),(256,)])
    x = depthwise_separable(x,params=[(1,),(256,)])
    x = depthwise_separable(x,params=[(2,),(512,)])
    
    if not shallow:
        for _ in range(5):
            x = depthwise_separable(x,params=[(1,),(512,)])
            
    x = depthwise_separable(x,params=[(2,),(1024,)])
    x = depthwise_separable(x,params=[(1,),(1024,)])

    x = GlobalAveragePooling2D()(x)
    out = Dense(classes, activation='softmax')(x)
    return out


def MobileNet_drop(img_input,shallow=False, classes=10,dropout_set=0.5):
    alpha = 0.75
    """Instantiates the MobileNet.Network has two hyper-parameters
        which are the width of network (controlled by alpha)
        and input size.
        # Arguments
            alpha: optional parameter of the network to change the 
                width of model.
            shallow: optional parameter for making network smaller.
            classes: optional number of classes to classify images
                into.
    """
    x = Conv2D(int(alpha*32), (3,3), strides=(2,2), padding='same')(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = depthwise_separable(x,params=[(1,),(64,)])
    x = depthwise_separable(x,params=[(2,),(128,)])
    x = depthwise_separable(x,params=[(1,),(128,)])
    x = depthwise_separable(x,params=[(2,),(256,)])
    x = depthwise_separable(x,params=[(1,),(256,)])
    x = depthwise_separable(x,params=[(2,),(512,)])
    
    if not shallow:
        for _ in range(5):
            x = depthwise_separable(x,params=[(1,),(512,)])
            
    x = depthwise_separable(x,params=[(2,),(1024,)])
    x = depthwise_separable(x,params=[(1,),(1024,)])

    x = GlobalAveragePooling2D()(x)
    x = Dropout(dropout_set)(x)
    out = Dense(classes, activation='softmax')(x)
    return out