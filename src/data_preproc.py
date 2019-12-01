import pandas as pd
import os
import numpy as np
from scipy.io import loadmat
import tensorflow as tf
import skimage
import scipy.io
import cv2
from pathlib import Path
from shutil import copyfile, copy2

import io
import torch
import random
import torchvision
import matplotlib.pyplot as plt
import pandas as pd
import PIL
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from operator import itemgetter
import pickle
from skimage.io import imread

DATA_DIR_CK = "../../../../../Downloads/"


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

    def load_ck_data(self):
        ck_prep = CKPrepare()
        ck_prep.copy_all_files_to_destionation()
        data_path = f'{DATA_DIR_CK}/emotion_split/'
        transform = transforms.Compose([transforms.ToTensor()])
        train_dataset = FolderWithPaths(
            root=data_path,
            transform=transform
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=64,
            num_workers=0,
            shuffle=True
        )
        return train_dataset, train_loader

    def load_svhn_data(self):
        train = scipy.io.loadmat('./src/data/train_32x32.mat')
        test = scipy.io.loadmat('./src/data/test_32x32.mat')
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


class CKPrepare:

    def __init__(self):
        self.df = pd.DataFrame()
        self.df_dict = []
        self.init_df()
        return

    def init_df(self):
        all_paths = self.process_paths()
        all_paths_df = pd.DataFrame(all_paths)
        all_paths_df['emotion'] = 0
        all_paths_df['file_name'] = ""
        all_paths_df['image_name'] = ""
        paths_imgs_dict = list(all_paths_df.T.to_dict().values())
        self.df = all_paths_df
        self.df_dict = paths_imgs_dict
        return all_paths_df, paths_imgs_dict

    def find_min_nr_imgs(self):
        min_imgs = 100
        max_imgs = 0
        folder_min = ""
        folder_max = ""
        root_dir_image = 'cohn-kanade-images/'
        all_paths = []
        ignore_folders = ['.DS_Store']
        emotion_folders = os.listdir(root_dir_image)
        for entry in emotion_folders:
            if entry not in ignore_folders:
                subject_folder = os.listdir(f'{root_dir_image}{entry}')
                for emotion in subject_folder:
                    if emotion not in ignore_folders:
                        emotion_folder = os.listdir(f'{root_dir_image}{entry}/{emotion}')
                        if len(emotion_folder) < min_imgs:
                            min_imgs = len(emotion_folder)
                            folder_min = f'{root_dir_image}{entry}/{emotion}'
                        if len(emotion_folder) > max_imgs:
                            max_imgs = len(emotion_folder)
                            folder_max = f'{root_dir_image}{entry}/{emotion}'
        return min_imgs, folder_min, max_imgs, folder_max

    def process_paths(self):
        root_dir = f'{DATA_DIR_CK}/Emotion/'
        root_dir_image = f'{DATA_DIR_CK}/cohn-kanade-images/'
        all_paths = []
        ignore_folders = ['.DS_Store']
        emotion_labels = os.listdir(root_dir)
        for entry in emotion_labels:
            if entry not in ignore_folders:
                subject_folder = os.listdir(f'{root_dir}{entry}')
                new_paths = [{'path': f'{root_dir}{entry}/{sub}', 'path_image': f'{root_dir_image}{entry}/{sub}'}
                             for sub in subject_folder if sub not in ignore_folders]
                all_paths = all_paths + new_paths
        return all_paths

    def get_neutral_and_multiple_of_same_expression(self, path_image, labeled_img=None, emotion=None):
        imgs = np.array(os.listdir(path_image))
        middle_ground = int(len(imgs) / 2)
        neutral_names = list(range(1, middle_ground))
        emotion_names = list(range(middle_ground + 1, len(imgs)))

        for img in imgs:
            try:
                name_img = int(img.rsplit('_')[2].replace('.png', ''))
                if name_img in neutral_names:
                    df_neutral = {"path": "", "path_image": path_image, "emotion": 0, "file_name": "",
                                  "image_name": img}
                    self.df_dict.append(df_neutral)

                elif labeled_img is not None and name_img in emotion_names:
                    df_emotion = {"path": "", "path_image": path_image, "emotion": emotion, "file_name": "",
                                  "image_name": img}
                    self.df_dict.append(df_emotion)
            except:
                print(img)

        return

    def get_label_files(self):
        all_image_names = []
        for index, row in enumerate(self.df_dict):
            print(row['path'])
            if row["path"] != "":
                entries = Path(row['path'])
                labeled_img = None
                labeled_emotion = None
                for entry in entries.iterdir():
                    print(entry.name)
                    file_txt = open(f'{row["path"]}/{entry.name}', "r+")
                    contents = file_txt.read()
                    row['emotion'] = float(contents.rstrip())
                    # all_path_df.loc[index, 'emotion'] = float(contents.rstrip())
                    img_name = entry.name.rsplit('_', 1)[0]
                    # all_path_df.loc[index, 'file_name'] = entry.name
                    row['file_name'] = entry.name
                    # all_path_df.loc[index, 'image_name'] = f'{img_name}.png'
                    row['image_name'] = f'{img_name}.png'
                    all_image_names.append(f'{img_name}.png')
                    labeled_img = f'{img_name}.png'
                    labeled_emotion = float(contents.rstrip())
                    print(contents)

                self.get_neutral_and_multiple_of_same_expression(
                    row["path_image"],
                    labeled_img,
                    labeled_emotion
                )

        return

    def copy_all_files_to_destionation(self):
        self.get_label_files()
        all_paths_df = pd.DataFrame(self.df_dict)
        unique_emotions = list(all_paths_df['emotion'].unique())
        for emotion in unique_emotions:
            emotion_files = all_paths_df[all_paths_df['emotion'] == emotion]
            for index, row in emotion_files.iterrows():
                if row["image_name"] != "":
                    copy2(f'{row["path_image"]}/{row["image_name"]}', f'{DATA_DIR_CK}/emotion_split/{int(emotion)}/')
        return


class FolderWithPaths(torchvision.datasets.ImageFolder):

    def __getitem__(self, index):
        original_tuple = super(FolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]

        tuple_with_path = {
            'image': original_tuple[0],
            'target': original_tuple[1],
            'path': (path.rsplit('/', 1)[-1],)
        }
        list_img = [original_tuple[0], original_tuple[1], path.rsplit('/', 1)[-1]]
        tuple_with_path = original_tuple + (path.rsplit('/', 1)[-1],)
        return list_img


def main():
    ck_prep = CKPrepare()
    ck_prep.copy_all_files_to_destionation()
    return

if __name__ == '__main__':
    main()



