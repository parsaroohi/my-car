import requests
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
import random
import cv2
from tensorflow.python.keras.callbacks import History

np.random.seed(0)
(X_train, y_train), (X_test, y_test) = mnist.load_data
assert(X_train.shape[0] == y_train.shape[0]
       ), "The number of images is not equal to the number of labels"
assert(X_test.shape[0] == y_test.shape[0]
       ), "The number of images is not equal to the number of labels"
assert(X_train[1:] == (28, 28)), "The dimensions of the images are not 28*28"
assert(X_test[1:] == (28, 28)), "The dimensions of the images are not 28*28"

num_of_samples = []
cols = 5
num_calsses = 10
fig, axs = plt.subplot(nrows=num_calsses, ncols=cols, figsize=(10, 10))
fig.tight_layout()

for i in range(cols):
    for j in range(num_calsses):
        x_selected = X_train[y_train == j]
        axs[j][i].imshow(
            x_selected[random.randint(0, len(x_selected-1)), :, :], cmap=plt.get_cmap("gray"))
        axs[j][i].axis('off')
        if i == 2:
            axs[j][i].set_title(str(j))
            num_of_samples.append(len(x_selected))

plt.figure(figsize=(12, 4))
plt.bar(range(0, num_calsses), num_of_samples)
plt.title('distribution of the training datasets')
plt.xlabel('class number')
plt.ylabel('number of images')

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
X_train = X_train/255
X_test = X_test/255
num_pixels = 784
X_train = X_train.reshape(X_train.shape[0], num_pixels)
X_test = X_test.reshape(X_test.shape[0], num_pixels)

# 36


def create_model():
    model = Sequential()
    model.add(Dense(10, input_dim=num_pixels, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(num_calsses, activation='softmax'))
    model.compile(Adam(lr=0.01), loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


model = create_model()
print(model.summary())
history = model.fit(X_train, y_train, validation_split=0.1,
                    epochs=10, batch_size=200, verbose=1, shuffle=1)
# history
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['loss', 'val_loss'])
plt.title('loss')
plt.xlabel('epoch')
# accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['accuracy', 'val_accuracy'])
plt.title('accuracy')
plt.xlabel('epoch')

# work with image
url = 'https://www.researchgate.net/profile/Jose_Sempere/publication/221258631/figure/fig1/AS:305526891139075@1449854695342/Handwritten-digit-2.png'
response = requests.get(url, stream=True)
img = Image.open(response.raw)
plt.imshow(img)
img_array = np.array(img)
resized = cv2.resize(img_array, (28, 28))
gray_scale = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
image = cv2.bitwise_not(gray_scale)
plt.imshow(image, cmap=plt.get_cmap("gray"))
image = image/255
image = image.reshape(1, 784)

prediction = model.predict_classes(image)
print("predicted digit:", str(prediction))
