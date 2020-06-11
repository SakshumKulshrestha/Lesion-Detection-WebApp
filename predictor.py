# Import the libraries
import numpy as np
import tensorflow
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.python.keras.layers.core import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import cv2
import os

class Predictor():

    
    def clean(self):
        file = open('./static/proc_imgs/output.txt', 'w')
        file.close()

    def getPrediction(self, input_image):
        #Define all the class strings
        pred_classes = [
            'Actinic Keratoses(akiecc)',
            'Basal Cell Carcinoma(bcc)',
            'Seborrheic Keratosis(bkl)',
            'Dermatofibroma(df)',
            'Malignant Melanoma(mel)',
            'Melanocytic Nevi(nv)',
            'Vascular Lesions(vasc)'
        ]

        # Create a basic network
        mobile = keras.applications.mobilenet.MobileNet()
        x = mobile.layers[-6].output
        x = Dropout(0.25)(x)
        #The softmax layer to get the output
        predictions = Dense(7, activation='softmax')(x)
        model = Model(inputs=mobile.input, outputs=predictions)

        #load trained weights
        model.load_weights('.\weights\model.h5')
        #resize image for input
        img = self.getProcessedImg(input_image)

        #return predictions
        prediction = model.predict(img)[0]
        output_str = pred_classes[np.argmax(prediction)]
        output_certainty = prediction[np.argmax(prediction)]

        if output_certainty < 0.5:
            output_str = 'None of the above / Unsure'

        self.writeToFile(output_str, output_certainty)
        return output_str, output_certainty
    
    #reshape the image for input
    def getProcessedImg(self, input_image):

        cv2_image = np.fromstring(input_image, np.uint8)
        img = cv2.imdecode(cv2_image, cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = img[None, :]
        return img

    def writeToFile(self, o_str, o_certainty):
        file = open('./static/proc_imgs/output.txt', 'w+')
        lines = [str(o_str) + '\n', 'Certainty: ' + str(o_certainty) + '\n']
        file.writelines(lines)
        file.close()

