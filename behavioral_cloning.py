from numpy.lib.type_check import imag
from google.colab import files
from operator import index
import os
from sys import path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import keras
import cv2
import pandas as pd
import random
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from imgaug import augmenters as iaa
import ntpath

from tensorflow.python.keras.layers.convolutional import Conv2D

datadir = 'track'
columns = ['center', 'left', 'right',
           'steering', 'throttle', 'reverse', 'speed']
data = pd.read_csv(os.path.join(datadir, 'driving_log.csv'), names=columns)
pd.set_option('display.max_colwidth', -1)
data.head()


def path_leaf():
    head, tail = ntpath.split(path)
    return tail


data['center'] = data['center'].apply(path_leaf)
data['right'] = data['right'].apply(path_leaf)
data['left'] = data['left'].apply(path_leaf)
data.head()

num_bins = 25
sample_per_bin = 200
hist, bins = np.histogram(data['steering'], num_bins)
center = (bins[:, -1] + bins[1:])*0.5
plt.bar(center, hist, width=0.05)
plt.plot(np.min(data['steering']), np.max(
    data['steering']), (sample_per_bin, sample_per_bin))

print("total data:", len(data))
remove_list = []
for j in range(num_bins):
    list_ = []
    for i in range(len(data['steering'])):
        if data['steering'][i] >= bins[j] and data['steering'][i] <= bins[j+1]:
            list_.append(i)
        list_ = shuffle(list_)
        list_ = list_[sample_per_bin:]
        remove_list.extend(list_)
print("removed:", len(remove_list))
data.drop(data.index[remove_list], inplace=True)
print('remaining', len(data))

hist, _ = np.histogram(data['steering'], (num_bins))
plt.bar(center, hist, width=0.05)
plt.plot(np.min(data['steering']), np.max(
    data['steering']), (sample_per_bin, sample_per_bin))


def load_img_steering(datadir, df):
    image_path = []
    steering = []
    for i in range(len(data)):
        indexed_data = data.iloc[i]
        center, left, right = indexed_data[0], indexed_data[1], indexed_data[2]
        image_path.append(os.path.join(datadir, center.strip()))
        steering.append(float(indexed_data[3]))
    image_path = np.asarray(image_path)
    steering = np.asarray(steering)
    return image_path, steering


image_path, steering = load_img_steering(datadir+"/IMG", data)
X_train, X_valid, y_train, y_valid = train_test_split(
    image_path, steering, test_size=0.2, random_state=6)
print('Training samples:{}\nValid samples:{}'.format(len(X_train), len(X_valid)))

fig, axs = plt.subplot(1, 2, figsize=(12, 4))
axs[0].hist(y_train, bins=num_bins, width=0.05, color='blue')
axs[0].set_title('Training set')
axs[1].hist(y_valid, bins=num_bins, width=0.05, color='red')
axs[1].set_title('Validation set')


def img_preprocess(img):
    img = cv2.imread(img)
    img = img[60:135, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img/255
    return img


image = image_path[100]
original_image = mpimg.imread(image)
preprocessed_image = img_preprocess(image)
fig, axs = plt.subplots(1, 2, figsize=(15, 10))
fig.tight_layout()
axs[0].imshow(original_image)
axs[0].set_title('original image')
axs[0].imshow(preprocessed_image)
axs[0].set_title('preprocessed image')

X_train = np.array(list(map(img_preprocess, X_train)))
X_valid = np.array(list(map(img_preprocess, X_valid)))
plt.imshow(X_train[random.randint(0, len(X_train)-1)])
plt.axis('off')
print(X_train.shape)


def nvidia_model():
    model = Sequential()
    model.add(Conv2D(24, (5, 5), subsample=(2, 2),
                     input_shape=(66, 200, 3), activation='elu'))
    model.add(Conv2D(36, (5, 5), subsample=(2, 2), activation='elu'))
    model.add(Conv2D(48, (5, 5), subsample=(2, 2), activation='elu'))
    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(Conv2D(64, (3, 3), activation='elu'))
    # model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    # model.add(Dropout(0.5))
    model.add(Dense(50, activation='elu'))
    # model.add(Dropout(0.5))
    model.add(Dense(10, activation='elu'))
    # model.add(Dropout(0.5))
    model.add(Dense(1, activation='elu'))
    model.compile(Adam(lr=1e-4), loss='mse')
    return model


def batch_generator(image_path, steering_ang, batch_size, istraining):
    # 54
    while True:
        batch_img = []
        batch_steering = []
        for i in range(batch_size):
            random_index = random.randint(0, len(image_path)-1)
            if istraining:
                im, steering = random_augment(image_path(
                    random_index, steering_ang(random_index)))
            else:
                im = mpimg.imread(image_path(random_index))
                steering = steering_ang(random_index)
            img_preprocess(im)
            batch_img.append(im)
            batch_steering.append(steering)
        yield(np.array(batch_img)), (np.array(batch_steering))


model = nvidia_model()
print(model.summary())
history = model.fit_generator(batch_generator(X_train, y_train, 100,  1),
                              steps_per_epoch=300,
                              epochs=10,
                              validation_data=batch_generator(
                                  X_valid, y_valid, 100, 0),
                              validation_steps=200,
                              verbose=1,
                              shuffle=1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(('Training', 'Validation'))
plt.title('Loss')
plt.xlabel('Epoch')

# save the model
model.save('model.h5')
files.download('model.h5')

# 52


def zoom(image):
    zoom = iaa.Affine(scale=(1, 1.3))
    image = zoom.augment_image(image)
    return image


image = image_path[random.randint(0, 1000)]
original_image = mpimg.imread(image)
zoomed_image = zoom(original_image)
fig, axs = plt.subplots(1, 2, figsize=(15, 10))
fig.tight_layout()
axs[0].imshow(original_image)
axs[0].set_title('original image')
axs[1].imshow(zoomed_image)
axs[1].set_title('zoomed image')


def pan(image):
    pan = iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)})
    image = pan.augment_image(image)
    return image


image = image_path[random.randint(0, 1000)]
original_image = mpimg.imread(image)
panned_image = pan(original_image)
fig, axs = plt.subplots(1, 2, figsize=(15, 10))
fig.tight_layout()
axs[0].imshow(original_image)
axs[0].set_title('original image')
axs[1].imshow(panned_image)
axs[1].set_title('panned image')


def img_random_brightness(image):
    brightness = iaa.Multiply(0.2, 1.2)
    image = brightness.augment_image(image)
    return image


image = image_path[random.randint(0, 1000)]
original_image = mpimg.imread(image)
brightness_altered_image = img_random_brightness(original_image)
fig, axs = plt.subplots(1, 2, figsize=(15, 10))
fig.tight_layout()
axs[0].imshow(original_image)
axs[0].set_title('original image')
axs[1].imshow(brightness_altered_image)
axs[1].set_title('brightness altered image')


def img_random_flip(image, steering_angle):
    image = cv2.flip(image, 1)
    steering_angle = -steering_angle
    return image, steering_angle


random_index = random.randint(0, 1000)
image = image_path[random_index]
steering_angle = steering[random_index]
original_image = mpimg.imread(image)
flipped_image, flipped_steering_angle = img_random_flip(
    original_image, steering_angle)
fig, axs = plt.subplots(1, 2, figsize=(15, 10))
fig.tight_layout()
axs[0].imshow(original_image)
axs[0].set_title('original image'+'steering angle:'+str(steering_angle))
axs[1].imshow(flipped_image)
axs[1].set_title('flipped image'+'steering angel'+str(flipped_steering_angle))


def random_augment(image, steering_angle):
    image = mpimg.imread(image)
    if np.random.rand() < 0.5:
        image = pan(image)
    if np.random.rand() < 0.5:
        image = zoom(image)
    if np.random.rand() < 0.5:
        image = img_random_brightness(image)
    if np.random.rand() < 0.5:
        image, steering_angle = img_random_flip(
            image, steering_angle)
    return image, steering_angle


ncol = 2
nrow = 10
fig, axs = plt.subplots(nrow, ncol, figsize=(15, 10))
fig.tight_layout()
for i in range(10):
    randnum = random.randint(0, len(image_path)-1)
    random_steering = steering[randnum]
    random_image = image_path[randnum]
    original_image = mpimg.imread(random_image)
    augmented_image, steering = random_augment(random_image, random_steering)
    axs[i][0].imshow(original_image)
    axs[i][0].set_title('original image')
    axs[i][1].imshow(augmented_image)
    axs[i][1].set_title('augmented image')


x_train_gen, y_train_gen = next(batch_generator(X_train, y_train, 1, 1))
x_valid_gen, y_valid_gen = next(batch_generator(X_valid, y_valid, 1, 0))
fig, axs = plt.subplots(1, 2, figsize=(15, 10))
fig.tight_layout()
axs[0].imshow(x_train_gen[0])
axs[0].set_title('training image')
axs[1].imshow(x_valid_gen[0])
axs[1].set_title('validation image')
