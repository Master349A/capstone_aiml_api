# -*- coding: utf-8 -*-


from matplotlib import pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
import numpy as np # linear algebra
import keras
import pydicom as dicom
import cv2   


def get_predictions(dcm_path = 'input/sample.dcm'):
    # read the pre-trained model 
    PATH = 'api1'
    cnn = keras.models.load_model(PATH)
    print(cnn.summary())
    '''
    ds = dicom.dcmread(dcm_path)
    pixel_array_numpy = ds.pixel_array
    image_path = 'temp/img.jpg'

    cv2.imwrite(image_path, pixel_array_numpy)
    '''
    return 0


get_predictions()