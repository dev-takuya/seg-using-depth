
import os
import argparse

import numpy as np
import cv2

import h5py
import time
import glob

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pylab as plt

import keras.backend as K
from keras.utils import generic_utils
from keras.optimizers import Adam, SGD
from keras.models import load_model

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto(
    gpu_options=tf.GPUOptions(
        visible_device_list="2", # specify GPU number
        allow_growth=True
    )
)
set_session(tf.Session(config=config))

import models

def my_normalization(X):
    return X / 127.5 - 1
def my_inverse_normalization(X):
    return (X + 1.) / 2.

def predict(X_raw, generator_model, output_dir, filenames):
    X_gen = generator_model.predict(X_raw)
    X_gen = my_inverse_normalization(X_gen)

    for i in range(len(filenames)):
        cv2.imwrite(output_dir + os.path.basename(filenames[i]), X_gen[i] * 255.0)

def my_load_data(datasetpath):
    with h5py.File(datasetpath, "r") as hf:
        X_sketch_test = hf["test_data_raw"][:].astype(np.float32)
        X_sketch_test = my_normalization(X_sketch_test)
        return X_sketch_test

def my_predict(args):
    # create output finder
    if not os.path.exists(os.path.expanduser(args.datasetpath)):
        os.mkdir(findername)
    # create figures
    if not os.path.exists('./figures'):
        os.mkdir('./figures')

    # load data
    test_rgb = my_load_data(args.datasetpath)

    # load model and weight
    generator_model = load_model(args.weight_path)
    
    # predict img
    filePaths = glob.glob('./images/'+args.scene_number+'/test/rgb/*')
    filePaths.sort()
    predict(test_rgb, generator_model, './results/', filePaths)

def main():
    parser = argparse.ArgumentParser(description='Train Font GAN')
    parser.add_argument('--datasetpath', '-d', type=str, default='datasetimages.hdf5')
    parser.add_argument('--weight_path', '-wp', type=str, default='model_weight')
    parser.add_argument('--scene_number', '-s', type = str, default='1')
    args = parser.parse_args()

    K.set_image_data_format("channels_last")

    my_predict(args)


if __name__ == '__main__':
    main()
