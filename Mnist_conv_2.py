from tensorflow.python.util.nest import _yield_value
from traffic_sign import X_valid
import numpy as np
import matplotlib.pyplot as plt
import keras
import cv2
from pandas.core.algorithms import mode
import requests
from PIL import Image
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
import random
from keras.layers import Flatten
# from tensorflow.python.keras.backend import categorical_crossentropy
# from tensorflow.python.keras.layers.pooling import MaxPool2D
# from tensorflow.python.util.nest import flatten
# from tensorflow.python.keras.layers.pooling import MaxPool2D
# from tensorflow.python.keras.backend import conv2d
from keras.layers.convolutional import Conv2D
from keras.models import Model
from keras.layers.convolutional import MaxPooling2D
from tensorflow.python.keras.backend import conv2d, flatten
from tensorflow.python.keras.optimizer_v2 import adam


np.random.seed(0)
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(X_train.shape)
print(X_test.shape)
assert(X_train.shape[0] == y_train.shape[0]
       ), "The number of images is not equal to the number of labels."
assert(X_train.shape[1:] == (28, 28)
       ), "The dimensions of the images are not 28 x 28."
assert(X_test.shape[0] == y_test.shape[0]
       ), "The number of images is not equal to the number of labels."
assert(X_test.shape[1:] == (28, 28)
       ), "The dimensions of the images are not 28 x 28."

num_of_samples = []
cols = 5
num_classes = 10

fig, axs = plt.subplots(nrows=num_classes, ncols=cols, figsize=(5, 10))
fig.tight_layout()

for i in range(cols):
    for j in range(num_classes):
        x_selected = X_train[y_train == j]
        axs[j][i].imshow(x_selected[random.randint(
            0, (len(x_selected) - 1)), :, :], cmap=plt.get_cmap('gray'))
        axs[j][i].axis("off")
        if i == 2:
            axs[j][i].set_title(str(j))
            num_of_samples.append(len(x_selected))


print(num_of_samples)
plt.figure(figsize=(12, 4))
plt.bar(range(0, num_classes), num_of_samples)
plt.title("Distribution of the train dataset")
plt.xlabel("Class number")
plt.ylabel("Number of images")
plt.show()


X_train = X_train.reshape(60000, 28, 28, 1)
X_test = X_test.reshape(10000, 28, 28, 1)


y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

X_train = X_train/255
X_test = X_test/255

# define the LeNet_model function


def LeNet_model():
    model = Sequential()
    model.add(Conv2D(30, (5, 5), input_shape=(
        28, 28, 1), activation_function='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), stride=2))
    model.add(Conv2D(15, (3, 3), activation_function='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(Adam(lr='0.01'), loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


model = LeNet_model()
print(model.summary())

history = model.fit(X_train, y_train, epochs=10, validation_split=0.1,
                    batch_size=400, verbose=1, shuffle=1)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'validation'])
plt.title('Accuracy')
plt.xlabel('epoch')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('Loss')
plt.xlabel('epoch')

# work with image
url = 'https://www.researchgate.net/profile/Jose_Sempere/publication/221258631/figure/fig1/AS:305526891139075@1449854695342/Handwritten-digit-2.png'
response = requests.get(url, stream=True)
img = Image.open(response.raw)
plt.imshow(img, cmap=plt.get_cmap('gray'))

img_array = np.array(img)
resized = cv2.resize(img_array, (28, 28))
gray_scale = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
image = cv2.bitwise_not(gray_scale)
plt.imshow(image, cmap=plt.get_cmap("gray"))
image = image/255
image = image.reshape(1, 28, 28, 1)

prediction = model.predict_classes(image)
print("predicted digit:", str(prediction))

layer1 = Model(inputs=model.layers[0].input, outputs=model.layers[0].output)
layer2 = Model(inputs=model.layers[0].input, outputs=model.layers[2].output)
visual_layer1, visual_layer2 = layer1.predict(img), layer2.predict(img)


plt.figure(figsize=(10, 6))
for i in range(3):
    plt.subplot(6, 5, i+1)
    plt.imshow(visual_layer1[0, :, :, i], cmap=plt.get_cmap('jet'))
    plt.axis('off')

plt.figure(figsize=(10, 6))
for i in range(15):
    plt.subplot(3, 5, i+1)
    plt.imshow(visual_layer2[0, :, :, i], cmap=plt.get_cmap('jet'))
    plt.axis('off')
