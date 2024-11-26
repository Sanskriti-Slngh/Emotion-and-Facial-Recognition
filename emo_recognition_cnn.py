import numpy as np
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten, Merge
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Concatenate
from keras.utils import np_utils
import pickle
from matplotlib import pyplot as plt

# parameters to control the program
model_id = '1h'

loadModel = False
doTraining = True
data_file_name = 'data/emotion_48x48_aug_split90.dat'
model_file_name = 'models/cnn_aug' + str(model_id) + '.model'

#data_file_name = 'emotion_48x48_split90.dat'
#model_file_name = 'cnn_' + str(model_id) + '.model'

# Load the data
with open(data_file_name, 'rb') as fin:
    dataTrain, dataTest, emotion = pickle.load(fin)

X_train = np.transpose(dataTrain['input'])
y_train = np.transpose(dataTrain['target'])

X_test = np.transpose(dataTest['input'])
y_test = np.transpose(dataTest['target'])

X_train = X_train.reshape(X_train.shape[0], 48, 48,1)
X_test = X_test.reshape(X_test.shape[0], 48, 48,1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# 6. Preprocess class labels
Y_train = np_utils.to_categorical(y_train, 7)
Y_test = np_utils.to_categorical(y_test, 7)

# 7. Define model architecture

if doTraining:
    model = Sequential()

    if model_id == 1:
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(32, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(7, activation='softmax'))

    if model_id == 2:
        model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(48, 48, 1)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(32, activation='relu'))
        model.add(Dense(64, activation='relu'))
        # model.add(Dropout(0.5))
        model.add(Dense(7, activation='softmax'))

    if model_id == '1b':
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(32, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(7, activation='softmax'))

    if model_id == '1c':
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # 23x23
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # 10x10
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # 4x4
        model.add(Flatten())
        model.add(Dense(32, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(7, activation='softmax'))

    if model_id == '1d':
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # 23x23
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # 10x10
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # 4x4
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # 1x1
        model.add(Flatten())
        model.add(Dense(32, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(7, activation='softmax'))

    if model_id == '1f':
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # 23x23
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # 10x10
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # 4x4
        model.add(Flatten())
        model.add(Dense(32, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(7, activation='softmax'))

    if model_id == '1g':
        model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(48, 48, 1)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # 23x23
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # 10x10
        model.add(Conv2D(256, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # 4x4
        model.add(Flatten())
        model.add(Dense(32, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(7, activation='softmax'))

    if model_id == '1h':
        model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(48, 48, 1)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # 23x23
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # 10x10
        model.add(Conv2D(256, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # 4x4
        model.add(Flatten())
        model.add(Dense(32, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(7, activation='softmax'))

    if model_id == 3:
        model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(48, 48, 1)))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(32, activation='relu'))
        model.add(Dense(64, activation='relu'))
        # model.add(Dropout(0.5))
        model.add(Dense(7, activation='softmax'))

    if model_id == 4:
        model.add(Conv2D(10, (5, 5), activation='relu', input_shape=(48, 48, 1)))
        model.add(AveragePooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))
        model.add(Conv2D(10, (5, 5), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(1,1), padding='same'))
        model.add(Dropout(0.5))
        model.add(Conv2D(10, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(32, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(7, activation='softmax'))

    if model_id == 5:
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)))
        # resolution 46
        #model.add(AveragePooling2D(pool_size=(2, 2)))
        #model.add(Dropout(0.5))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        # resolution 44
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # resolution 22
        #model.add(Dropout(0.5))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        # resolution 20
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # resolution 10
        model.add(Conv2D(32, (3, 3), activation='relu'))
        # resolution 8
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # resolution 4
        model.add(Conv2D(64, (3, 3), activation='relu'))
        # resolution 2
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        #model.add(Dropout(0.5))
        model.add(Dense(64, activation='relu'))
        #model.add(Dropout(0.5))
        model.add(Dense(7, activation='softmax'))

    if model_id == 6:
        model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(48, 48, 1)))
        # resolution 44
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # resolution 22
        #model.add(Dropout(0.5))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        # resolution 20
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # resolution 10
        #model.add(Dropout(0.5))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        # resolution 8
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # resolution 4
        model.add(Conv2D(32, (3, 3), activation='relu'))
        # resolution 2
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # resolution 1
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        #model.add(Dropout(0.5))
        model.add(Dense(64, activation='relu'))
        #model.add(Dropout(0.5))
        model.add(Dense(32, activation='relu'))
        # model.add(Dropout(0.5))
        model.add(Dense(7, activation='softmax'))

    if model_id == 7:
        model.add(Conv2D(64, (5, 5), activation='relu', input_shape=(48, 48, 1)))
        # resolution 44
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # resolution 22
        #model.add(Dropout(0.5))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        # resolution 20
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # resolution 10
        #model.add(Dropout(0.5))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        # resolution 8
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # resolution 4
        model.add(Conv2D(32, (3, 3), activation='relu'))
        # resolution 2
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # resolution 1
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        #model.add(Dropout(0.5))
        model.add(Dense(64, activation='relu'))
        #model.add(Dropout(0.5))
        model.add(Dense(32, activation='relu'))
        # model.add(Dropout(0.5))
        model.add(Dense(7, activation='softmax'))

    if model_id == 8:
        model.add(Conv2D(64, (5, 5), activation='relu', input_shape=(48, 48, 1)))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2,2)))
        model.add(Conv2D(64, (5, 5), activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(Conv2D(128, (4, 4), activation='relu'))
        model.add(Flatten())
        model.add(Dense(3072, activation='relu'))
        model.add(Dense(7, activation='softmax'))

    if model_id == 9:
        model1 = Sequential()
        # 48x48
        model1.add(Conv2D(16, (3, 3), activation='relu', input_shape=(48, 48, 1)))
        # 46x46
        model1.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model1.add(Dropout(0.5))
        # 23x23
        model1.add(Conv2D(32, (3, 3), activation='relu'))
        # 21x21
        model1.add(MaxPooling2D(pool_size=(2, 2)))
        model1.add(Dropout(0.5))
        # 10x10
        model1.add(Conv2D(64, (3, 3), activation='relu'))
        # 8x8
        model1.add(MaxPooling2D(pool_size=(2, 2)))
        model1.add(Dropout(0.5))
        # 4x4
        model1.add(Flatten())
        print(model1.summary())

        model2 = Sequential()
        # 48x48
        model2.add(Conv2D(16, (5, 5), activation='relu', input_shape=(48, 48, 1)))
        # 44x44
        model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model2.add(Dropout(0.5))
        # 22x22
        model2.add(Conv2D(32, (5, 5), activation='relu'))
        # 18x18
        model2.add(MaxPooling2D(pool_size=(2, 2)))
        model2.add(Dropout(0.5))
        # 9x9
        model2.add(Conv2D(64, (5, 5), activation='relu'))
        # 5x5
        model2.add(MaxPooling2D(pool_size=(2, 2)))
        model2.add(Dropout(0.5))
        # 2x2
        model2.add(Flatten())
        print(model2.summary())

        model3 = Sequential()
        # 48x48
        model3.add(Conv2D(16, (7, 7), activation='relu', input_shape=(48, 48, 1)))
        # 42x42
        model3.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model3.add(Dropout(0.5))
        # 21x21
        model3.add(Conv2D(32, (7, 7), activation='relu'))
        # 15x15
        model3.add(MaxPooling2D(pool_size=(2, 2)))
        model3.add(Dropout(0.5))
        # 7x7
        model3.add(Conv2D(64, (7, 7), activation='relu'))
        model3.add(Dropout(0.5))
        # 1x1
        model3.add(Flatten())
        print(model3.summary())

        model4 = Sequential()
        # 48x48
        model4.add(Conv2D(16, (9, 9), activation='relu', input_shape=(48, 48, 1)))
        # 40x40
        model4.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model4.add(Dropout(0.5))
        # 20x20
        model4.add(Conv2D(32, (9, 9), activation='relu'))
        # 12x12
        model4.add(MaxPooling2D(pool_size=(2, 2)))
        model4.add(Dropout(0.5))
        # 6x6
        model4.add(Flatten())
        print(model4.summary())

        model.add(Merge([model1, model2, model3, model4], mode ='concat'))
        #model.Concatenate([model1, model2, model3, model4])
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(7, activation='softmax'))

    if model_id == 10:
        model.add(Conv2D(64, (5, 5), activation='relu', input_shape=(48, 48, 1)))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(Dropout(0.5))
        model.add(Conv2D(64, (5, 5), activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(Dropout(0.5))
        model.add(Conv2D(128, (4, 4), activation='relu'))
        model.add(Flatten())
        model.add(Dense(3072, activation='relu'))
        model.add(Dense(7, activation='softmax'))

    # print model's summary
    print (model.summary())

    if loadModel:
        with open(model_file_name, 'r') as fin:
            model = model_from_json(fin.read())
        model.load_weights(model_file_name + ".h5")
        print("Loaded model from disk")

     # 8. Compile model
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # 9. Fit model on training data
    if model_id == 9:
        history = model.fit([X_train, X_train, X_train, X_train], y_train, batch_size=32, epochs=5, verbose=1)
    else:
        history = model.fit(X_train, y_train, batch_size = 32, epochs = 40, verbose = 1)

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.show()

    print(model.output_shape)
    model_json = model.to_json()
    with open(model_file_name, "w") as fout:
        fout.write(model_json)
    model.save_weights(model_file_name + ".h5")
    print ("Saved model to disk")
else:
    with open(model_file_name, 'r') as fin:
        model = model_from_json(fin.read())
    model.load_weights(model_file_name + ".h5")
    print("Loaded model from disk")
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    if model_id == 9:
        score = model.evaluate([X_train, X_train, X_train, X_train], y_train, verbose=1)
    else:
        score = model.evaluate(X_train, y_train, verbose=1)
    print (score)

# 10. Evaluate model on test data
print (X_test.shape)
if model_id == 9:
    score = model.evaluate([X_test, X_test, X_test, X_test], y_test, verbose=1)
else:
    score = model.evaluate(X_test, y_test, verbose=1)
print (score)

# for i in range(X_test.shape[0]):
#     plt.imshow(X_test[i,:,:,0], cmap='gray')
#     plt.show()
