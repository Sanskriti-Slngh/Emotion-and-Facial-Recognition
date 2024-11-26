import numpy as np
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
from matplotlib import pyplot as plt
import pickle


np.random.seed(123)
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print (X_train.shape)

plt.imshow(X_train[0])
plt.show()

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# convert 1-dimensional class arrays to 10-D class matrices
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)
print (y_train.shape)

if False:
    model = Sequential()
    model.add(Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)))
    model.add(Conv2D(32, (3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10,activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, batch_size=32, epochs=1, verbose=1)

    print(model.output_shape)
    model_json = model.to_json()
    with open("cnn_mnist.model", "w") as fout:
        fout.write(model_json)
    model.save_weights("cnn_mnist.h5")
    print ("Saved model to disk")

else:
    with open('cnn_mnist.model', 'r') as fin:
        model = model_from_json(fin.read())
    model.load_weights("cnn_mnist.h5")
    print("Loaded model from disk")
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

score = model.evaluate(X_test, y_test, verbose=0)
print (score)