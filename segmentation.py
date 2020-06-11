import tensorflow
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Activation, Dense, BatchNormalization, Dropout, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, Input, Reshape
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam, SGD
import tensorflow as tf
import numpy as np
import pandas as pd
import glob
import PIL
from PIL import Image
import matplotlib.pyplot as plt
import os
import cv2

class Segmentation:

    def segnet(self):
        # Encoding layer
        img_input = Input(shape= (192, 256, 3))
        x = Conv2D(64, (3, 3), padding='same', name='conv1',strides= (1,1))(img_input)
        x = BatchNormalization(name='bn1')(x)
        x = Activation('relu')(x)
        x = Conv2D(64, (3, 3), padding='same', name='conv2')(x)
        x = BatchNormalization(name='bn2')(x)
        x = Activation('relu')(x)
        x = MaxPooling2D()(x)
        
        x = Conv2D(128, (3, 3), padding='same', name='conv3')(x)
        x = BatchNormalization(name='bn3')(x)
        x = Activation('relu')(x)
        x = Conv2D(128, (3, 3), padding='same', name='conv4')(x)
        x = BatchNormalization(name='bn4')(x)
        x = Activation('relu')(x)
        x = MaxPooling2D()(x)

        x = Conv2D(256, (3, 3), padding='same', name='conv5')(x)
        x = BatchNormalization(name='bn5')(x)
        x = Activation('relu')(x)
        x = Conv2D(256, (3, 3), padding='same', name='conv6')(x)
        x = BatchNormalization(name='bn6')(x)
        x = Activation('relu')(x)
        x = Conv2D(256, (3, 3), padding='same', name='conv7')(x)
        x = BatchNormalization(name='bn7')(x)
        x = Activation('relu')(x)
        x = MaxPooling2D()(x)

        x = Conv2D(512, (3, 3), padding='same', name='conv8')(x)
        x = BatchNormalization(name='bn8')(x)
        x = Activation('relu')(x)
        x = Conv2D(512, (3, 3), padding='same', name='conv9')(x)
        x = BatchNormalization(name='bn9')(x)
        x = Activation('relu')(x)
        x = Conv2D(512, (3, 3), padding='same', name='conv10')(x)
        x = BatchNormalization(name='bn10')(x)
        x = Activation('relu')(x)
        x = MaxPooling2D()(x)
        
        x = Conv2D(512, (3, 3), padding='same', name='conv11')(x)
        x = BatchNormalization(name='bn11')(x)
        x = Activation('relu')(x)
        x = Conv2D(512, (3, 3), padding='same', name='conv12')(x)
        x = BatchNormalization(name='bn12')(x)
        x = Activation('relu')(x)
        x = Conv2D(512, (3, 3), padding='same', name='conv13')(x)
        x = BatchNormalization(name='bn13')(x)
        x = Activation('relu')(x)
        x = MaxPooling2D()(x)

        x = Dense(1024, activation = 'relu', name='fc1')(x)
        x = Dense(1024, activation = 'relu', name='fc2')(x)
        # Decoding Layer 
        x = UpSampling2D()(x)
        x = Conv2DTranspose(512, (3, 3), padding='same', name='deconv1')(x)
        x = BatchNormalization(name='bn14')(x)
        x = Activation('relu')(x)
        x = Conv2DTranspose(512, (3, 3), padding='same', name='deconv2')(x)
        x = BatchNormalization(name='bn15')(x)
        x = Activation('relu')(x)
        x = Conv2DTranspose(512, (3, 3), padding='same', name='deconv3')(x)
        x = BatchNormalization(name='bn16')(x)
        x = Activation('relu')(x)
        
        x = UpSampling2D()(x)
        x = Conv2DTranspose(512, (3, 3), padding='same', name='deconv4')(x)
        x = BatchNormalization(name='bn17')(x)
        x = Activation('relu')(x)
        x = Conv2DTranspose(512, (3, 3), padding='same', name='deconv5')(x)
        x = BatchNormalization(name='bn18')(x)
        x = Activation('relu')(x)
        x = Conv2DTranspose(256, (3, 3), padding='same', name='deconv6')(x)
        x = BatchNormalization(name='bn19')(x)
        x = Activation('relu')(x)

        x = UpSampling2D()(x)
        x = Conv2DTranspose(256, (3, 3), padding='same', name='deconv7')(x)
        x = BatchNormalization(name='bn20')(x)
        x = Activation('relu')(x)
        x = Conv2DTranspose(256, (3, 3), padding='same', name='deconv8')(x)
        x = BatchNormalization(name='bn21')(x)
        x = Activation('relu')(x)
        x = Conv2DTranspose(128, (3, 3), padding='same', name='deconv9')(x)
        x = BatchNormalization(name='bn22')(x)
        x = Activation('relu')(x)

        x = UpSampling2D()(x)
        x = Conv2DTranspose(128, (3, 3), padding='same', name='deconv10')(x)
        x = BatchNormalization(name='bn23')(x)
        x = Activation('relu')(x)
        x = Conv2DTranspose(64, (3, 3), padding='same', name='deconv11')(x)
        x = BatchNormalization(name='bn24')(x)
        x = Activation('relu')(x)
        
        x = UpSampling2D()(x)
        x = Conv2DTranspose(64, (3, 3), padding='same', name='deconv12')(x)
        x = BatchNormalization(name='bn25')(x)
        x = Activation('relu')(x)
        x = Conv2DTranspose(1, (3, 3), padding='same', name='deconv13')(x)
        x = BatchNormalization(name='bn26')(x)
        x = Activation('sigmoid')(x)
        pred = Reshape((192,256))(x)
        
        model = Model(inputs=img_input, outputs=pred)
        
        model.compile(optimizer= SGD(lr=0.001, momentum=0.9, decay=0.0005, nesterov=False), loss= ["binary_crossentropy"])
        
        return model

    def getPreProcImg(self, cv2_image):
        
        img = cv2.imdecode(cv2_image, cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h,w,_ = img.shape
        mask = np.zeros((h,w), np.uint8)
        cv2.circle(mask, (300,220), 325, 255, -1)
        img = cv2.bitwise_and(img, img, mask= mask)

        img = cv2.resize(img, (256, 192))

        return img
    
    def enhance(self, img, model_1):
        sub = (model_1.predict(img.reshape(1,192,256,3))).flatten()

        for i in range(len(sub)):
            if sub[i] > 0.5:
                sub[i] = 1
            else:
                sub[i] = 0
        return sub

    def getCropped(self, input_image):
        model_0 = self.segnet()
        model_0.compile(optimizer= SGD(lr=0.001, momentum=0.9, decay=0.0005, nesterov=False), loss= ["binary_crossentropy"])
        model_0.load_weights('.\weights\segnet.h5')

        cv2_image = np.fromstring(input_image, np.uint8)
        img = self.getPreProcImg(cv2_image)
        img_array = np.array(img).astype(np.float32)
        pred = self.enhance(img_array, model_0)
        pred = pred.reshape(192, 256)


        result = img.copy()
        result[pred != 1] = 0
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        return result

    def clean(self):
        if 'seg_img.jpg' in os.listdir('./static/proc_imgs'):
            os.remove('./static/proc_imgs/seg_img.jpg')
    



    

