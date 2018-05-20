########################################################################################################################
# mnist_cnn1.py
# Author: Zach Harris @jzharris
#
# Dependencies:
#   Python 3.6.0
#   TensorFlow 1.4.x
#   Keras 2.x
#
# Python packages:
#   os
#   argparse
#   matplotlib
#   scipy
#
# Datasets in use:
#   MNIST
#
# Notes:
#   Determine version of your TF: python3 -c 'import tensorflow as tf; print(tf.__version__)'

import os
import os.path as path
import argparse

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
from keras import backend as K

import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib

import matplotlib.pyplot as plt
from scipy.misc import imsave

########################################################################################################################
# Set ArgumentParser so you can set all variables from terminal

parser = argparse.ArgumentParser(description='Test Arguments')

# model params
parser.add_argument('--model_name', default='mnist_cnn1')       # the name of the saved model

# dataset params
parser.add_argument('--export_images', default=False)           # instead of training a model, save MNIST to .png's
parser.add_argument('--export_number', default=10)              # number of MNIST images to export (if enabled)
parser.add_argument('--plot_images', default=False)             # instead of training a model, display MNIST images

# training params
parser.add_argument('--epochs', default=5)                      # number of epochs to train model for
parser.add_argument('--batch_size', default=128)                # batch size to use for training

args = parser.parse_args()

########################################################################################################################
# Global Vars

model_name = args.model_name
export_images = args.export_images
export_number = args.export_number
plot_images = args.plot_images
epochs = args.epochs
batch_size = args.batch_size

########################################################################################################################
# Load data: load the MNIST train/validation sets, and export/display MNIST images

def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if export_images:

        # save images from MNIST to files
        for i in range(export_number):
            if not path.exists('export'):
                os.mkdir('export')

            imsave('export/mnist_train_{}.png'.format(i), x_train[i])
            imsave('export/mnist_test_{}.png'.format(i), x_test[i])

        exit(0)

    if plot_images:
        # plot 4 images as gray scale
        plt.subplot(221)
        plt.imshow(x_train[0], cmap=plt.get_cmap('gray'))
        plt.subplot(222)
        plt.imshow(x_train[1], cmap=plt.get_cmap('gray'))
        plt.subplot(223)
        plt.imshow(x_train[2], cmap=plt.get_cmap('gray'))
        plt.subplot(224)
        plt.imshow(x_train[3], cmap=plt.get_cmap('gray'))
        plt.show()

        exit(0)

    # if not exporting or plotting, return processed dataset
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    return x_train, y_train, x_test, y_test

########################################################################################################################
# Build the basic CNN model:

def build_model():
    model = Sequential()
    model.add(Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu', input_shape=[28, 28, 1]))
    # model is now 28*28*64
    model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))
    # model is now 14*14*64

    model.add(Conv2D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu'))
    # model is now 14*14*128
    model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))
    # model is now 7*7*128

    model.add(Conv2D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu'))
    # model is now 7*7*256
    model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))
    # model is now 4*4*256

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(10, activation='softmax'))
    # model is now 10

    return model

########################################################################################################################
# Train the model using Adadelta optimizer from Keras

def train(model, x_train, y_train, x_test, y_test):
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=True, validation_data=(x_test, y_test))

########################################################################################################################
# Export the frozen graph for later use in Unity

def export_model(saver, model, input_node_names, output_node_name):
    if not path.exists('out'):
        os.mkdir('out')

    tf.train.write_graph(K.get_session().graph_def, 'out', model_name + '_graph.pbtxt')

    saver.save(K.get_session(), 'out/' + model_name + '.chkp')

    freeze_graph.freeze_graph('out/' + model_name + '_graph.pbtxt', None, False,
                              'out/' + model_name + '.chkp', output_node_name,
                              "save/restore_all", "save/Const:0",
                              'out/frozen_' + model_name + '.bytes', True, "")

    input_graph_def = tf.GraphDef()
    with tf.gfile.Open('out/frozen_' + model_name + '.bytes', "rb") as f:
        input_graph_def.ParseFromString(f.read())

    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
            input_graph_def, input_node_names, [output_node_name],
            tf.float32.as_datatype_enum)

    with tf.gfile.FastGFile('out/opt_' + model_name + '.bytes', "wb") as f:
        f.write(output_graph_def.SerializeToString())

    print("graph saved!")

########################################################################################################################
# Main program

def main():

    # 1. load dataset
    x_train, y_train, x_test, y_test = load_data()

    # 2. build model
    model = build_model()

    # 3. train model
    train(model, x_train, y_train, x_test, y_test)

    # 4. export model to file for Unity
    export_model(tf.train.Saver(), model, ["conv2d_1_input"], "dense_2/Softmax")


if __name__ == '__main__':
    main()
