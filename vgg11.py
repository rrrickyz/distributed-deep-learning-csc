# VGG11 model based on the dog-or-cat database: https://www.kaggle.com/competitions/dogs-vs-cats/data?select=sampleSubmission.csv

import os, datetime
import random
import pathlib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from distutils.version import LooseVersion as LV
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import TensorBoard

print('Using Tensorflow version: {}, and Keras version: {}.'.format(tf.__version__, tf.keras.__version__))
assert(LV(tf.__version__) >= LV("2.0.0"))

# Data
# -- dogs-vs-cats
# ---- test1
# ---- train

if 'DATAIR' in os.environ:
    DATAIR = os.environ['DATAIR']
else:
    DATAIR = '/scratch/project_2006142/'

datapath = os.path.join(DATAIR, 'dogs-vs-cats/')
assert os.path.exists(datapath), "Data not found at "+datapath

def get_paths(dataset):
    data_root = pathlib.Path(datapath+dataset)
    image_paths = list(data_root.glob('*.jpg'))
    image_paths = [str(path) for path in image_paths]
    image_count = len(image_paths)
    print("Found {} images.".format(image_count))
    return image_paths

# store the path of each image
# image_paths = {'train':[...], 'test':[...]}
image_paths = dict()
image_paths['train'] = sorted(get_paths('train/'))
# image_paths['test'] = get_paths('test1/')

train_size=len(image_paths['train'])

label_names = {'cat','dog'}
label_to_index = dict((name, index) for index,name in enumerate(label_names))

# store the label of each image
# i.e. /.../cat.10055.jpg --> cat
def get_labels(dataset):
    return [label_to_index[path.split('/')[5].split('.')[0]]
           for path in image_paths[dataset]]

# {'train': [0,1,0,1,...]}
image_labels = dict()
image_labels['train'] = sorted(get_labels('train'))

def load_image(path, label):
    print(path)
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    return tf.image.resize(image, INPUT_IMAGE_SIZE), label

INPUT_IMAGE_SIZE = [224,224]

#split the training dataset into training set and validation set
from sklearn.model_selection import train_test_split
train_paths, validation_paths, train_labels, validation_labels = train_test_split(image_paths['train'],image_labels['train'],test_size=0.2)
train_dataset = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
validation_dataset = tf.data.Dataset.from_tensor_slices((validation_paths, validation_labels))

BATCH_SIZE = 256

# convert lists to tensors
train_dataset = train_dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.shuffle(25000).batch(BATCH_SIZE, drop_remainder=True)
train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

validation_dataset = validation_dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
validation_dataset = validation_dataset.shuffle(25000).batch(BATCH_SIZE, drop_remainder=True)
validation_dataset = validation_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

epochs = 100

# model:
inputs = keras.Input(shape=INPUT_IMAGE_SIZE+[3])

# pre-processing
x = layers.Rescaling(scale=1./255)(inputs)
x = layers.RandomCrop(224,224)(x)
x = layers.RandomFlip(mode="horizontal")(x)

# layers
x = layers.Conv2D(64, (3,3), activation='relu')(x)
x = layers.MaxPooling2D(pool_size=(2,2), strides=2)(x)
x = layers.Conv2D(128, (3,3), activation='relu')(x)
x = layers.MaxPooling2D(pool_size=(2,2), strides=2)(x)
x = layers.Conv2D(256, (3,3), activation='relu')(x)
x = layers.Conv2D(256, (3,3), activation='relu')(x)
x = layers.MaxPooling2D(pool_size=(2,2), strides=2)(x)
x = layers.Conv2D(512, (3,3), activation='relu')(x)
x = layers.Conv2D(512, (3,3), activation='relu')(x)
x = layers.MaxPooling2D(pool_size=(2,2), strides=2)(x)
x = layers.Conv2D(512, (3,3), activation='relu')(x)
x = layers.Conv2D(512, (3,3), activation='relu')(x)
x = layers.MaxPooling2D(pool_size=(2,2), strides=2)(x)

x = layers.Flatten()(x)
l2 = keras.regularizers.L2(l2=0.0005)
x = layers.Dense(4096, activation='relu', kernel_regularizer='l2')(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(4096, activation='relu', kernel_regularizer='l2')(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation='sigmoid')(x)

# the learning rate should stop decreasing when the accuracy of validation set stops improving
initial_learning_rate=0.01
opt = keras.optimizers.SGD(learning_rate=initial_learning_rate, momentum=0.9)

model = keras.Model(inputs=inputs, outputs=outputs, name="VGG11_model")
model.compile(loss='binary_crossentropy', optimizer=opt,metrics=['accuracy'])
print(model.summary())
vgg11_name = model.name

callbacks = [keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1,patience=5, min_lr=0.00001)]

# log messages
logdir = os.path.join(os.getcwd(), "logs", "cats-or-dogs"+vgg11_name+"-reuse-"+
                      datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
print('TensorBoard log directory: ', logdir)
os.makedirs(logdir)
history = model.fit(train_dataset, epochs=epochs,
                   validation_data=validation_dataset,
                   callbacks=callbacks, verbose=2)
fname = 'cat-or-dog-' + vgg11_name + '-reuse.h5'
print('Saving model to ', fname)
model.save(fname)
