import cv2
import os
import re
import bz2
from matplotlib import pyplot as plt
import face_recognition
import pickle
import time
import numpy as np
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
import imutils

emo_model_number = 9
use_cnn = True
cnn_model_id = 7
enable_augmentation = False

# results


if enable_augmentation:
    cnn_model_file = 'cnn_aug' + str(cnn_model_id) + '.model'
else:
    cnn_model_file = 'cnn_' + str(cnn_model_id) + '.model'
if use_cnn:
    with open(cnn_model_file, 'r') as fin:
        cnn_model = model_from_json(fin.read())
    cnn_model.load_weights(cnn_model_file + ".h5")
    print("Loaded model from disk from %s" %(cnn_model_file))
    cnn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# Original model
emotion_model_file = 'modelFNNSadHappy.dat'
num_layers = 5
act_funcs = ['relu', 'relu', 'relu', 'relu', 'sigmoid']
# ** Not good ** because model is trained only on three outputs - sad, happy and neutral
# totalFaces = 185,, errorCount = 95, error = 51.351351351351354

#model1
if emo_model_number == 1:
    emotion_model_file = 'modelFNNSadHappy_1.dat'
    num_layers = 3
    act_funcs = ['relu', 'relu', 'sigmoid']

if emo_model_number == 9:
    emotion_model_file = "Fnn" + str(emo_model_number) + ".model"
    num_layers = 3
    num_neurons = [32, 64, 7]
    act_funcs = ['relu', 'relu', 'sigmoid']
# totalFaces = 185,, errorCount = 110, error = 59.45945945945946

# functions
def sigmoid(x):
    return 1.0/(1+np.exp(-x))

def relu(x):
    return np.maximum(x,0)

def predict_emo(model, X):
    # normalize the input
    X = X/255.0
    a = {}
    z = {}
    for layer in range(num_layers):
        for q in range(num_layers):
            if q == 0:
                z[str(q)] = np.dot(model['w' + str(q)], X) + model['b' + str(q)]
            else:
                z[str(q)] = np.dot(model['w' + str(q)], a[str(q-1)]) + model['b' + str(q)]
            if act_funcs[q] == 'relu':
                a[str(q)] = relu(z[str(q)])
            elif act_funcs[q] == 'sigmoid':
                a[str(q)] = sigmoid(z[str(q)])
            else:
                print("I do not understand this function, " + act_funcs[q] + ".")
        y_hat = a[str(num_layers - 1)]
        max_y_hat = np.max(y_hat, axis=0, keepdims=True)
        p = y_hat == max_y_hat
        return(p)

# Load trained models
if use_cnn:
    all_emo = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
else:
    with open(emotion_model_file, 'rb') as f:
        emo_model, all_emo = pickle.load(f)

emo_to_id = {}
emo_to_id['Angry'] = 0
emo_to_id['Disgust'] = 1
emo_to_id['Fear'] = 2
emo_to_id['Happy'] = 3
emo_to_id['Sad'] = 4
emo_to_id['Surprise'] = 5
emo_to_id['Neutral'] = 6


confusion_matrix = np.zeros((7,7))


path = 'test_emo'

goldenData = {}
name = 'img_1024x1024.jpg'
goldenData[name] = []
goldenData[name].append(('Briti','Happy'))
goldenData[name].append(('Tiya','Happy'))
goldenData[name].append(('Kirti','Neutral'))
goldenData[name].append(('Manish','Neutral'))

goldenData['img_1920x1080.jpg'] = []
goldenData['img_1920x1080.jpg'].append(('Briti','Happy'))
goldenData['img_1920x1080.jpg'].append(('Manish','Neutral'))
goldenData['img_1920x1080.jpg'].append(('Kirti','Neutral'))
goldenData['img_1920x1080.jpg'].append(('Tiya','Happy'))

goldenData['img_256x256.jpg'] = []
goldenData['img_256x256.jpg'].append(('Tiya','Happy'))
goldenData['img_256x256.jpg'].append(('Manish','Neutral'))

goldenData['img_512x512.jpg'] = []
goldenData['img_512x512.jpg'].append(('Briti','Happy'))
goldenData['img_512x512.jpg'].append(('Manish','Neutral'))
goldenData['img_512x512.jpg'].append(('Tiya','Happy'))
goldenData['img_512x512.jpg'].append(('Kirti','Neutral'))

name = 'pics (1).jpg'
goldenData[name] = []
goldenData[name].append(('Stranger','Happy'))
goldenData[name].append(('Stranger','Happy'))
goldenData[name].append(('Stranger','Happy'))
goldenData[name].append(('Stranger','Happy'))
goldenData[name].append(('Stranger','Happy'))
goldenData[name].append(('Scarlet','Happy'))
goldenData[name].append(('Stranger','Happy'))
goldenData[name].append(('Stranger','Happy'))

name = 'pics (10).jpg'
goldenData[name] = []
goldenData[name].append(('Hasini','Happy'))
goldenData[name].append(('Manish','Happy'))
goldenData[name].append(('Hari','Neutral'))
goldenData[name].append(('Tiya','Neutral'))
goldenData[name].append(('Briti','Happy'))

name = 'pics (11).jpg'
goldenData[name] = []
goldenData[name].append(('Hasini','Happy'))
goldenData[name].append(('Tiya','Happy'))
goldenData[name].append(('Hari','Happy'))
goldenData[name].append(('Manish','Happy'))
goldenData[name].append(('Briti','Happy'))

name = 'pics (12).jpg'
goldenData[name] = []
goldenData[name].append(('Tiya','Happy'))
goldenData[name].append(('Hasini','Happy'))
goldenData[name].append(('Briti','Happy'))
goldenData[name].append(('Manish','Happy'))
goldenData[name].append(('Hari','Happy'))

name = 'pics (13).jpg'
goldenData[name] = []
goldenData[name].append(('Hm','Happy'))
goldenData[name].append(('Kirti','Happy'))
goldenData[name].append(('Tiya','Happy'))
goldenData[name].append(('Hari','Happy'))
goldenData[name].append(('Stranger','Neutral'))
goldenData[name].append(('Hasini','Happy'))
goldenData[name].append(('Briti','Happy'))

name = 'pics (14).jpg'
goldenData[name] = []
goldenData[name].append(('Tiya','Happy'))
goldenData[name].append(('Hari','Happy'))
goldenData[name].append(('Kirti','Happy'))
goldenData[name].append(('Hasini','Happy'))
goldenData[name].append(('Hm','Happy'))
goldenData[name].append(('Manish','Happy'))
goldenData[name].append(('Stranger','Neutral'))
goldenData[name].append(('Briti','Neutral'))

name = 'pics (15).jpg'
goldenData[name] = []
goldenData[name].append(('Kirti','Angry'))
goldenData[name].append(('Manish','Neutral'))
goldenData[name].append(('Tiya','Sad'))

name = 'pics (16).jpg'
goldenData[name] = []
goldenData[name].append(('Manish','Neutral'))
goldenData[name].append(('Tiya', 'Sad'))

name = 'pics (17).jpg'
goldenData[name] = []
goldenData[name].append(('Manish','Neutral'))
goldenData[name].append(('Tiya','Neutral'))

name = 'pics (18).jpg'
goldenData[name] = []
goldenData[name].append(('Hasini','Happy'))
goldenData[name].append(('Manish','Happy'))
goldenData[name].append(('Hari','Happy'))
goldenData[name].append(('Tiya','Happy'))
goldenData[name].append(('Briti','Happy'))

name = 'pics (19).jpg'
goldenData[name] = []
goldenData[name].append(('Hasini','Happy'))
goldenData[name].append(('Tiya','Happy'))
goldenData[name].append(('Hari','Happy'))
goldenData[name].append(('Manish','Happy'))
goldenData[name].append(('Briti','Happy'))

name = 'pics (2).jpg'
goldenData[name] = []
goldenData[name].append(('Stranger','Happy'))
goldenData[name].append(('Stranger','Happy'))
goldenData[name].append(('Stranger','Happy'))
goldenData[name].append(('Scarlet','Happy'))
goldenData[name].append(('Stranger','Happy'))
goldenData[name].append(('0','0'))

name = 'pics (20).jpg'
goldenData[name] = []
goldenData[name].append(('Hm','Happy'))
goldenData[name].append(('Kirti','Happy'))
goldenData[name].append(('Tiya','Happy'))
goldenData[name].append(('Hari','Happy'))
goldenData[name].append(('Stranger','Neutral'))
goldenData[name].append(('Hasini','Happy'))
goldenData[name].append(('Briti','Happy'))

name = 'pics (21).jpg'
goldenData[name] = []
goldenData[name].append(('Tiya','Happy'))

name = 'pics (22).jpg'
goldenData[name] = []
goldenData[name].append(('Pranavi','Happy'))
goldenData[name].append(('Tiya','Happy'))
goldenData[name].append(('Tanishq','Happy'))

name = 'pics (23).jpg'
goldenData[name] = []
goldenData[name].append(('Tiya','Happy'))
goldenData[name].append(('Pranavi','Neutral'))
goldenData[name].append(('Tanishq','Neutral'))

name = 'pics (24).jpg'
goldenData[name] = []
goldenData[name].append(('Trump','Angry'))

name = 'pics (25).jpg'
goldenData[name] = []
goldenData[name].append(('Kirti','Neutral'))
goldenData[name].append(('Manish','Happy'))

name = 'pics (26).jpg'
goldenData[name] = []
goldenData[name].append(('Tiya','Neutral'))
goldenData[name].append(('Manish','Angry'))

name = 'pics (27).jpg'
goldenData[name] = []
goldenData[name].append(('Tanishq','Happy'))
goldenData[name].append(('Stranger','Neutral'))

name = 'pics (28).jpg'
goldenData[name] = []
goldenData[name].append(('Tanishq','Sad'))
goldenData[name].append(('Stranger','Happy'))

name = 'pics (29).jpg'
goldenData[name] = []
goldenData[name].append(('Tiya','Happy'))

name = 'pics (3).jpg'
goldenData[name] = []
goldenData[name].append(('Manish','Neutral'))
goldenData[name].append(('Tiya','Neutral'))
goldenData[name].append(('Briti','Sad'))

name = 'pics (30).jpg'
goldenData[name] = []
goldenData[name].append(('Tiya','Happy'))

name = 'pics (31).jpg'
goldenData[name] = []
goldenData[name].append(('Tiya','Sad'))

name = 'pics (32).jpg'
goldenData[name] = []
goldenData[name].append(('Tiya','Neutral'))

name = 'pics (33).jpg'
goldenData[name] = []
goldenData[name].append(('Tiya','Happy'))

name = 'pics (34).jpg'
goldenData[name] = []
goldenData[name].append(('Tiya','Sad'))

name = 'pics (35).jpg'
goldenData[name] = []
goldenData[name].append(('Tiya','Happy'))

name = 'pics (36).jpg'
goldenData[name] = []
goldenData[name].append(('Tiya','Happy'))

name = 'pics (37).jpg'
goldenData[name] = []
goldenData[name].append(('Tiya','Neutral'))

name = 'pics (38).jpg'
goldenData[name] = []
goldenData[name].append(('Tiya','Happy'))
goldenData[name].append(('Manish','Happy'))

name = 'pics (4).jpg'
goldenData[name] = []
goldenData[name].append(('Stranger','Happy'))
goldenData[name].append(('Angelina','Happy'))
goldenData[name].append(('Stranger','Angry'))

name = 'pics (40).jpg'
goldenData[name] = []
goldenData[name].append(('Arya','Neutral'))
goldenData[name].append(('Tiya','Happy'))
goldenData[name].append(('Pranavi','Surprise'))
goldenData[name].append(('Tanishq','Fear'))

name = 'pics (41).jpg'
goldenData[name] = []
goldenData[name].append(('Arya','Neutral'))
goldenData[name].append(('Pranavi','Surprise'))

name = 'pics (42).jpg'
goldenData[name] = []
goldenData[name].append(('Tanishq','Neutral'))
goldenData[name].append(('Saanvi','Neutral'))

name = 'pics (43).jpg'
goldenData[name] = []
goldenData[name].append(('Sarvin','Neutral'))
goldenData[name].append(('Tanishq','Neutral'))
goldenData[name].append(('Arya','Neutral'))
goldenData[name].append(('Saanvi','Neutral'))

name = 'pics (44).jpg'
goldenData[name] = []
goldenData[name].append(('Sarvin','Angry'))
goldenData[name].append(('Tiya','Happy'))
goldenData[name].append(('Arya','Happy'))
goldenData[name].append(('Stranger','Neutral'))
goldenData[name].append(('Tanishq','Angry'))
goldenData[name].append(('Pranvi','Sad'))

name = 'pics (45).jpg'
goldenData[name] = []
goldenData[name].append(('Pranavi','Sad'))
goldenData[name].append(('Tanishq','Fear'))

name = 'pics (5).jpg'
goldenData[name] = []
goldenData[name].append(('Manish','Happy'))
goldenData[name].append(('Tiya','Neutral'))
goldenData[name].append(('Briti','Neutral'))

name = 'pics (57).jpg'
goldenData[name] = []
goldenData[name].append(('Obama','Sad'))
goldenData[name].append(('Bush','Neutral'))
goldenData[name].append(('Stranger','Happy'))
goldenData[name].append(('Stranger','Happy'))
goldenData[name].append(('Stranger','Happy'))

name = 'pics (58).jpg'
goldenData[name] = []
goldenData[name].append(('Bush','Happy'))
goldenData[name].append(('Obama','Happy'))
goldenData[name].append(('Stranger','Happy'))
goldenData[name].append(('Stranger','Happy'))

name = 'pics (59).jpg'
goldenData[name] = []
goldenData[name].append(('Trump','Happy'))
goldenData[name].append(('Obama','Happy'))

name = 'pics (6).jpg'
goldenData[name] = []
goldenData[name].append(('Tiya','Neutral'))
goldenData[name].append(('Briti','Happy'))
goldenData[name].append(('Manish','Neutral'))

name = 'pics (60).jpg'
goldenData[name] = []
goldenData[name].append(('Stranger','Happy'))
goldenData[name].append(('Obama','Happy'))
goldenData[name].append(('Trump','Happy'))

name = 'pics (61).jpg'
goldenData[name] = []
goldenData[name].append(('Stranger','Neutral'))
goldenData[name].append(('Strnger','Neutral'))
goldenData[name].append(('Stranger','Neutral'))
goldenData[name].append(('Stranger','Neutral'))
goldenData[name].append(('Dwayne','Neutral'))
goldenData[name].append(('Stranger','Neutral'))

name = 'pics (62).jpg'
goldenData[name] = []
goldenData[name].append(('Robert','Neutral'))
goldenData[name].append(('Stranger','Neutral'))

name = 'pics (63).jpg'
goldenData[name] = []
goldenData[name].append(('Scarlett','Happy'))
goldenData[name].append(('Stranger','Happy'))

name = 'pics (64).jpg'
goldenData[name] = []
goldenData[name].append(('Angelina','Neutral'))

name = 'pics (65).jpg'
goldenData[name] = []
goldenData[name].append(('Robert','Happy'))
goldenData[name].append(('Stranger','Happy'))

name = 'pics (66).jpg'
goldenData[name] = []
goldenData[name].append(('Stranger','Happy'))
goldenData[name].append(('Stranger','Happy'))
goldenData[name].append(('Stranger','Happy'))
goldenData[name].append(('Robert','Happy'))

name = 'pics (67).jpg'
goldenData[name] = []
goldenData[name].append(('Dwayne','Happy'))

name = 'pics (68).jpg'
goldenData[name] = []
goldenData[name].append(('Dwayne','Neutral'))

name = 'pics (69).jpg'
goldenData[name] = []
goldenData[name].append(('Angelina','Happy'))

name = 'pics (7).jpg'
goldenData[name] = []
goldenData[name].append(('Kirti','Neutral'))
goldenData[name].append(('Manish','Happy'))
goldenData[name].append(('Tiya','Happy'))
goldenData[name].append(('Briti','Happy'))

name = 'pics (70).jpg'
goldenData[name] = []
goldenData[name].append(('Steve','Happy'))
goldenData[name].append(('Stranger','Happy'))
goldenData[name].append(('Stranger','Happy'))
goldenData[name].append(('Stranger','Happy'))
goldenData[name].append(('Stranger','Neutral'))
goldenData[name].append(('Stranger','Happy'))

name = 'pics (71).jpg'
goldenData[name] = []
goldenData[name].append(('Steve','Happy'))

name = 'pics (72).jpg'
goldenData[name] = []
goldenData[name].append(('Stranger','Neutral'))
goldenData[name].append(('Scarlet','Happy'))
goldenData[name].append(('Stranger','Happy'))
goldenData[name].append(('Stranger','Neutral'))
goldenData[name].append(('Stranger','Disgust'))

name = 'pics (73).jpg'
goldenData[name] = []
goldenData[name].append(('Stranger','Neutral'))
goldenData[name].append(('Stranger','Neutral'))
goldenData[name].append(('Stranger','Neutral'))
goldenData[name].append(('Scarlet','Neutral'))


name = 'pics (76).jpg'
goldenData[name] = []
goldenData[name].append(('Tiya','Happy'))
goldenData[name].append(('Manish','Happy'))
goldenData[name].append(('Briti','Happy'))
goldenData[name].append(('Hasini','Happy'))
goldenData[name].append(('Hari','Happy'))
goldenData[name].append(('Hm','Happy'))
goldenData[name].append(('Kirti','Happy'))
goldenData[name].append(('Hm','Happy'))
goldenData[name].append(('Kirti','Happy'))

name = 'pics (77).jpg'
goldenData[name] = []
goldenData[name].append(('Tiya','Happy'))
goldenData[name].append(('Manish','Happy'))
goldenData[name].append(('Briti','Happy'))
goldenData[name].append(('Hasini','Happy'))
goldenData[name].append(('Hari','Happy'))
goldenData[name].append(('Hm','Happy'))
goldenData[name].append(('Kirti','Happy'))
goldenData[name].append(('Hm','Happy'))
goldenData[name].append(('Kirti','Happy'))
goldenData[name].append(('Kirti','Happy'))

name = 'pics (78).jpg'
goldenData[name] = []
goldenData[name].append(('Scarlet','Happy'))
goldenData[name].append(('Stranger','Neutral'))
goldenData[name].append(('Robert','Neutral'))

name = 'pics (9).jpg'
goldenData[name] = []
goldenData[name].append(('Hasini','Neutral'))
goldenData[name].append(('Tiya','Happy'))
goldenData[name].append(('Manish','Happy'))
goldenData[name].append(('Briti','Neutral'))


dataCnt = 0
errorCount = 0
totalFaces = 0
for (dirname, dirpath, files) in os.walk(path):
    print (dirname, dirpath, len(files))
    for file in files:
        dirname = dirname.replace('\\', '/')
        fname = dirname + '/' + file
        print ("Reading %s file" %(fname))
        image = cv2.imread(fname)
        face_locations = face_recognition.face_locations(image)

        # predict emotions using model
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        firstData = True
        h,w = img.shape
        margin = 10
        if len(face_locations) == 0:
            continue
        for (top, right, bottom, left) in face_locations:
            if emo_model_number == 0:
                curr_x = cv2.resize(img[top:bottom, left:right], (64, 64))
            else:
                curr_x = cv2.resize(img[max(0,top-margin):min(h,bottom+margin), max(0,left-margin):min(w,right+margin)], (48, 48))
            f = np.transpose(curr_x.reshape(1, -1))
            if firstData:
                X = f
                firstData = False
            else:
                X = np.hstack((X, f))
            # add 5 more pictures for just added picture
            # -20, +20, flip, -20, +20
            if enable_augmentation:
                image_pic = np.zeros((48, 48, 3))
                image_pic[:, :, 0] = curr_x
                image_pic[:, :, 1] = curr_x
                image_pic[:, :, 2] = curr_x
                for angle in (-20, 20):
                    image_r = imutils.rotate(image_pic, angle)
                    f = np.transpose(image_r[:,:,0].reshape(1, -1))
                    print (f.shape)
                    X = np.hstack((X, f))
                curr_x = cv2.flip(curr_x, 1)
                f = np.transpose(curr_x.reshape(1, -1))
                print(f.shape)
                X = np.hstack((X, f))
                image_pic[:, :, 0] = curr_x
                image_pic[:, :, 1] = curr_x
                image_pic[:, :, 2] = curr_x
                for angle in (-20, 20):
                    image_r = imutils.rotate(image_pic, angle)
                    f = np.transpose(image_r[:,:,0].reshape(1, -1))
                    print(f.shape)
                    X = np.hstack((X, f))

        # set the type as float 32, and normalize
        print (X.shape)
        X = X.astype('float32')
        X /= 255

        time0 = time.time()
        print('Calling predict emotions')
        if use_cnn:
            X = np.transpose(X)
            print ('Manish' + str(X.shape))
            X = X.reshape(X.shape[0], 48, 48, 1)
            if False:
                for i in range(X.shape[0]):
                    plt.imshow(X[i,:,:,0], cmap='gray')
                    plt.show()
                print (X.shape)
            p = cnn_model.predict(X)
            max_p = np.max(p, axis=1, keepdims=True)
            #print (p)
            #print (max_p)
            A = p == max_p
            A = np.transpose(A)
            # find majority of 6 columns
            if enable_augmentation:
                m1,n1 = A.shape
                A_new = np.zeros((m1, int(n1/6)))
                for i in range(int(n1/6)):
                    A_new[:,] = np.sum(A[:,i*6:i*6+6],axis=1,keepdims=True)
                max_p = np.max(A_new, axis=0, keepdims=True)
                A = A_new == max_p

            #print(A)
            #exit()
        else:
            A = predict_emo(emo_model, X)
        print('Returning predict emotions')
        time1 = time.time()
        print("The time taken in finding emotions is %s" % (time1 - time0))
        emo_names = []

        rows, columns = A.shape
        for col in range(columns):
            for row in range(rows):
                if A[row, col]:
                    emotion = all_emo[min(len(all_emo) - 1, row)]
                    emo_names.append(emotion)
                    break


        face_id = 0
        for (top, right, bottom, left), predicted_emo in zip(face_locations, emo_names):
            cv2.rectangle(image, (left, top), (right, bottom), (0,0,255), 2)
            cv2.rectangle(image, (left, bottom), (right, bottom+40), (0,0,255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            if dirname == 'test_emo/angry' or \
                dirname == 'test_emo/disgust' or \
                dirname == 'test_emo/happy' or \
                dirname == 'test_emo/neutral' or \
                dirname == 'test_emo/sad' or \
                dirname == 'test_emo/fear' or \
                dirname == 'test_emo/surprise':
                tmp_var = dirname.split('/')[1]
                name, emo = ('Stranger', tmp_var.title())
            elif face_id < len(goldenData[file]):
                name, emo = goldenData[file][face_id]
            else:
                name, emo = ('Stranger', 'Neutral')
                print ("default: face_id %s, len=%s" %(face_id, len(goldenData[file])))
            #predicted_emo = 'Neutral'
            if emo != predicted_emo:
                errorCount = errorCount + 1
            confusion_matrix[emo_to_id[emo],emo_to_id[predicted_emo]] += 1
            totalFaces = totalFaces + 1
            face_id = face_id+1
            cv2.putText(image, name, (left + 6, bottom + 20), font, 0.8, (255, 255, 255), 1)
            cv2.putText(image, emo, (left + 6, bottom + 40), font, 0.8, (255, 255, 255), 1)
            cv2.putText(image, predicted_emo, (left+6, bottom + 60), font, 0.8, (255, 255, 255), 1)
        if False:
            #cv2.imshow(fname, image)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
            plt.imshow(image)
            plt.show()
        dataCnt = dataCnt + 1

# calclualte error
acc = (1.0 - errorCount*1/totalFaces)*100
print ('totalFaces = %s,, errorCount = %s, acc = %s %%' %(totalFaces, errorCount, acc))
print (confusion_matrix)
print (np.sum(confusion_matrix, axis=1))