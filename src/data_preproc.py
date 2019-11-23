import pandas as pd
import numpy as np
from scipy.io import loadmat
import tensorflow as tf
import skimage
import cv2


class DataLoader:
    def __init__(self, data='svhn'):
        self.data_name = data
        func_data = getattr(self, f'load_{data}_data')
        self.train_x, self.train_y, self.test_x, self.test_y = func_data()
        return

    def to_grayscale(self, imgs_color):
        gray_images = [cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY) for img_color in imgs_color]
        return np.array(gray_images)

    def load_svhn_data(self):
        train = loadmat('./src/data/train_32x32.mat')
        test = loadmat('./src/data/test_32x32.mat')
        d1_train, d2_train, d3_train, d4_train = train['X'].shape
        d1_test, d2_test, d3_test, d4_test = test['X'].shape
        train_x = np.reshape(train['X'], (d4_train, d1_train, d2_train, d3_train))
        test_x = np.reshape(test['X'], (d4_test, d1_test, d2_test, d3_test))
        gray_scale_x_train = self.to_grayscale(train_x)
        gray_scale_x_test = self.to_grayscale(test_x)
        x_data_reshaped = gray_scale_x_train.reshape((d4_train, d1_train * d2_train))
        x_test_reshaped = gray_scale_x_test.reshape((d4_test, d1_test * d2_test))
        x_data_reshaped = x_data_reshaped/255
        x_test_reshaped = x_test_reshaped/255
        y_train = tf.keras.utils.to_categorical(train['y'])
        y_test = tf.keras.utils.to_categorical(test['y'])
        return x_data_reshaped, \
                y_train,\
                x_test_reshaped, y_test


