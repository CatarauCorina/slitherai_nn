import pandas as pd
import numpy as np
from scipy.io import loadmat
import tensorflow as tf
import skimage
import scipy.io
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

    def transform_label_10_to_0(self, data):
        out = np.array(data).flatten()
        out = np.where(out == 10, 0, out)
        return out

    def normalize(self, imgs_color):
        normalized_images = [(img - img.mean(axis=0))/img.std(axis=0) for img in imgs_color]
        return np.array(normalized_images)

    def load_svhn_data(self):
        train = scipy.io.loadmat('./data/train_32x32.mat')
        test = scipy.io.loadmat('./data/test_32x32.mat')
        train_x = np.moveaxis(train['X'], -1, 0)
        test_x = np.moveaxis(test['X'], -1, 0)
        d1_train, d2_train, d3_train, d4_train = train_x.shape
        d1_test, d2_test, d3_test, d4_test = test_x.shape

        nr_samples_train = d1_train
        nr_samples_test = d1_test

        flatten_train = d2_train*d3_train*d4_train
        flatten_test = d2_test*d3_test*d4_test

        x_data_reshaped = train_x.reshape(nr_samples_train, flatten_train)
        x_test_reshaped = test_x.reshape(nr_samples_test, flatten_test)

        normalized_x_train = self.normalize(x_data_reshaped)
        normalized_x_test = self.normalize(x_test_reshaped)

        y_0_10_train = self.transform_label_10_to_0(train['y'])
        y_0_10_test = self.transform_label_10_to_0(test['y'])

        y_train = tf.keras.utils.to_categorical(y_0_10_train)
        y_test = tf.keras.utils.to_categorical(y_0_10_test)
        return normalized_x_train, \
                y_train,\
                normalized_x_test, y_test


