# Distributed VGG16 model based on the dog-or-cat database: https://www.kaggle.com/competitions/dogs-vs-cats/data?select=sampleSubmission.csv

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

# Horovod: import and initialize
import horovod.tensorflow.keras as hvd
hvd.init()

if hvd.rank() == 0:
    print('Using Tensorflow version:', tf.__version__,
          'Keras version:', tf.keras.__version__,
          'backend:', tf.keras.backend.backend())
    print('Using Horovod with', hvd.size(), 'workers')

# Horovod: pin GPU to be used to process local rank (one GPU per process)
# return a list of physical GPUs visible to the host runtime
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    # the runtime initialization will not allocate all memory on the device
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

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

# pretained model
pt_model = keras.models.load_model('/scratch/project_2006142/cat-or-dog-VGG11_model-reuse.h5')
pt_name = pt_model.name
print('Using "{}" pre-trained model with {} layers'.format(pt_name, len(pt_model.layers)))

# model:
inputs = keras.Input(shape=INPUT_IMAGE_SIZE+[3])

# pre-processing
x = layers.Rescaling(scale=1./255)(inputs)
x = layers.RandomCrop(224,224)(x)
x = layers.RandomFlip(mode="horizontal")(x)

# layers
x = layers.Conv2D(64, (3,3), activation='relu')(x)
x = layers.Conv2D(64, (3,3), activation='relu')(x)
x = layers.MaxPooling2D(pool_size=(2,2), strides=2)(x)
x = layers.Conv2D(128, (3,3), activation='relu')(x)
x = layers.Conv2D(128, (3,3), activation='relu')(x)
x = layers.MaxPooling2D(pool_size=(2,2), strides=2)(x)
x = layers.Conv2D(256, (3,3), activation='relu')(x)
x = layers.Conv2D(256, (3,3), activation='relu')(x)
x = layers.Conv2D(256, (1,1), activation='relu')(x)
x = layers.MaxPooling2D(pool_size=(2,2), strides=2)(x)
x = layers.Conv2D(512, (3,3), activation='relu')(x)
x = layers.Conv2D(512, (3,3), activation='relu')(x)
x = layers.Conv2D(512, (1,1), activation='relu')(x)
x = layers.MaxPooling2D(pool_size=(2,2), strides=2)(x)
x = layers.Conv2D(512, (3,3), activation='relu')(x)
x = layers.Conv2D(512, (3,3), activation='relu')(x)
x = layers.Conv2D(512, (1,1), activation='relu')(x)
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

# Horovod: add Horovod DistributedOptimizer.
opt = hvd.DistributedOptimizer(opt)

# Horovod: Specify 'experimental_run_tf_function=False' to ensure
# TensorFlow uses hvd.DistributedOptimizer() to compute gradients.
model = keras.Model(inputs=inputs, outputs=outputs, name="VGG16_model_hvd")

# fine-tuning
model.layers[4].set_weights(pt_model.layers[4].get_weights())
model.layers[7].set_weights(pt_model.layers[6].get_weights())
model.layers[10].set_weights(pt_model.layers[8].get_weights())
model.layers[11].set_weights(pt_model.layers[9].get_weights())
model.layers[23].set_weights(pt_model.layers[18].get_weights())
model.layers[25].set_weights(pt_model.layers[20].get_weights())
model.layers[27].set_weights(pt_model.layers[22].get_weights())

model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
if hvd.rank() == 0:
    print(model.summary())
    vgg16_hvd_name = model.name

# learning
callbacks = [keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1,patience=5, min_lr=0.00001),
             keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=10),
             # Horovod: broadcast initial variable states from rank 0 to all other processes.
             # ensure consistent initialization of all workers when training is started with
             # random weights or restored from a checkpoint.
             hvd.callbacks.BroadcastGlobalVariablesCallback(0),

             # Horovod: average metrics among workers at the end of every epoch.
             hvd.callbacks.MetricAverageCallback(),

             # Scale the learning rate 'lr = 1.0' --> 'lr = 1.0 * hvd.size()' during the first
             # three epochs.
             hvd.callbacks.LearningRateWarmupCallback(initial_learning_rate, warmup_epochs=3,verbose=1)
             ]

# log messages
if hvd.rank() == 0:
    logdir = os.path.join(os.getcwd(), "logs", "cats-or-dogs"+vgg16_hvd_name+"-reuse-"+
                      datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    print('TensorBoard log directory: ', logdir)
    os.makedirs(logdir)
    callbacks.append(TensorBoard(log_dir=logdir))

verbose = 2 if hvd.rank() == 0 else 0

history = model.fit(train_dataset, epochs=epochs,
                   validation_data=validation_dataset,
                   callbacks=callbacks, verbose=verbose)

if hvd.rank() == 0:
    fname = 'cat-or-dog-' + vgg16_hvd_name + '-reuse.h5'
    print('Saving model to ', fname)
    model.save(fname)

print('All done for rank ', hvd.rank())