from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import requests
import numpy as np
import matplotlib.pyplot as plt
import keras
import random
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.layers import Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
import pickle
import pandas as pd
import cv2

from numpy.core.fromnumeric import shape

np.random.seed(0)

with open('german-traffic-signs/train.p', 'rb') as f:
    train_data = pickle.load(f)

with open('german-traffic-signs/valid.p', 'rb') as f:
    valid_data = pickle.load(f)

with open('german-traffic-signs/test.p', 'rb') as f:
    test_data = pickle.load(f)

X_train, y_train = train_data['features'], train_data['labels']
X_valid, y_valid = valid_data['features'], valid_data['labels']
X_test, y_test = test_data['features'], test_data['labels']
print(X_train.shape)
print(X_valid.shape)
print(X_test.shape)

assert(X_train.shape[0] == y_train.shape[0]
       ), 'the number of images is not equal to the number of labels'
assert(X_test.shape[0] == y_test.shape[0]
       ), 'the number of images is not equal to the number of labels'
assert(X_valid.shape[0] == y_valid.shape[0]
       ), 'the number of images is not equal to the number of labels'
assert(X_train.shape[1:] == (32, 32, 3)
       ), 'the dimension of images are not 32*32*3'

data = pd.read_csv('german-traffic-signs/signnames.csv')

num_of_samples = []

cols = 5
num_classes = 43

fig, axs = plt.subplots(nrows=num_classes, ncols=cols, figsize=(5, 50))
fig.tight_layout()
for i in range(cols):
    for j, row in data.iterrows():
        x_selected = X_train[y_train == j]
        axs[j][i].imshow(x_selected[random.randint(
            0, len(x_selected - 1)), :, :], cmap=plt.get_cmap("gray"))
        axs[j][i].axis("off")
        if i == 2:
            axs[j][i].set_title(str(j)+"-"+row['SignName'])
            num_of_samples.append(len(x_selected))

plt.show(X_train[1000])
plt.axis('off')


def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img


img = grayscale(X_train[1000])
plt.imshow(img)
plt.axis('off')
print(img.shape)


def equalize(img):
    cv2.equalizeHist(img)
    return img


img = equalize(img)
plt.imshow(img)
plt.axis('off')
print(img.shape)


def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img/255
    return img


X_train = np.array(list(map(preprocessing, X_train)))
X_test = np.array(list(map(preprocessing, X_test)))
X_valid = np.array(list(map(preprocessing, X_valid)))
plt.imshow(X_train[random.randint(0, len(X_train)-1)])
plt.axis('off')
print(X_train.shape)

X_train = X_train.reshape(34799, 32, 32, 1)
X_test = X_test.reshape(12630, 32, 32, 1)
X_valid = X_valid.reshape(4410, 32, 32, 1)

# 44
datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1,
                             zoom_range=0.2, shear_range=0.1, rotation_range=10)
datagen.fit(X_train)
# will be commented
# batches = datagen.flow(X_train, y_train, batch_size=20)
# X_batch, y_batch = next(batches)
# fig, axs = plt.subplots(1, 15, figsize=(20, 5))
# fig.tight_layout()
# for i in range(15):
#     axs[i].imshow(X_batch[i].reshape(32, 32))
#     axs[i].axis('off')

# 42
y_train = to_categorical(y_train, 43)
y_test = to_categorical(y_test, 43)
y_valid = to_categorical(y_valid, 43)

print(num_of_samples)
plt.figure(figsize=(12, 4))
plt.bar(range(0, num_classes), num_of_samples)
plt.title("Distribution of the training dataset")
plt.xlabel("Class number")
plt.ylabel("Number of images")

# 43


def modified_model():
    model = Sequential()
    model.add(Conv2D(60, (5, 5), input_shape=(32, 32, 1), activation='relu'))
    model.add(Conv2D(60, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(30, (3, 3), activation='relu'))
    model.add(Conv2D(30, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(43, activation='softmax'))
    model.compile(Adam(lr=0.001), loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


model = modified_model()
print(model.summary())
history = model.fit_generator(datagen.flow(X_train, y_train, batch_size=50),
                              steps_per_epoch=2000, epochs=10,
                              validation_data=(X_valid, y_valid), shuffle=1)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('loss')
plt.xlabel('epoch')

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'validation'])
plt.title('accuracy')
plt.xlabel('epoch')

score = model.evaluate(X_test, y_test, verbose=0)
print("Test score", score[0])
print("Test Accuracy", score[1])

# predict internet number
url = 'https://c8.alamy.com/comp/A0RX23/cars-and-automobiles-must-turn-left-ahead-sign-A0RX23.jpg'
r = requests.get(url, stream=True)
img = Image.open(r.raw)
plt.imshow(img, cmap=plt.get_cmap('gray'))

img = np.asarray(img)
img = cv2.resize(img, (32, 32))
img = preprocessing(img)
plt.imshow(img, cmap=plt.get_cmap('gray'))
print(img.shape)
img = img.reshape(1, 32, 32, 1)

print("predicted sign: " + str(model.predict_classes(img)))
