import numpy as np
import face_recognition
import dlib
from skimage import io
import cv2
import pickle
import time
from aiModels.Fnn import Fnn
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils


# This should not be hard coded
all_emo = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
cnn_model_id = 5

cnn_model_file = 'cnn_aug' + str(cnn_model_id) + '.model'
with open(cnn_model_file, 'r') as fin:
    cnn_model = model_from_json(fin.read())
cnn_model.load_weights(cnn_model_file + ".h5")
print("Loaded model from disk from %s" %(cnn_model_file))
cnn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Load image
image = cv2.imread("../session3/pic10.jpg")
#image = cv2.resize(image, (512,768))
img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# List of face locations in the picture
time0 = time.time()
#face_locations = face_recognition.face_locations(image, model = "cnn")
face_locations = face_recognition.face_locations(image)
print (face_locations)
time1 = time.time()
print("The time taken to do face locations is %s" %(time1 - time0))
if len(face_locations) == 0:
    print("Please try taking another picture, no face was found.")
    exit()

# finding the emotion for each face
# create the input vector
firstData = True
h,w = img.shape
margin = 10
for (top,right,bottom,left) in face_locations:
    curr_x = cv2.resize(img[max(0,top-margin):min(h,bottom+margin), max(0,left-margin):min(w,right+margin)], (48, 48))
    f = np.transpose(curr_x.reshape(1,-1))
    if firstData:
        X = f
        firstData = False
    else:
        X = np.hstack((X, f))

# set the type as float 32, and normalize
X = X.astype('float32')
X /= 255
X = np.transpose(X)
X = X.reshape(X.shape[0], 48, 48, 1)
# predict
p = cnn_model.predict(X)
max_p = np.max(p, axis=1, keepdims=True)
A = p == max_p
A = np.transpose(A)

# Add name of emotion based on result
emo_names = []
rows, columns = A.shape
for col in range(columns):
    for row in range(rows):
        if A[row, col]:
            emotion = all_emo[min(len(all_emo)-1, row)]
            emo_names.append(emotion)
            break

## display the information
# display the results
for (top, right, bottom, left), emo in zip(face_locations, emo_names):
    cv2.rectangle(image, (left, top), (right, bottom), (0,0,255), 2)
    cv2.rectangle(image, (left, bottom), (right, bottom+40), (0,0,255), cv2.FILLED)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(image, emo, (left + 6, bottom+40), font, 0.8, (255, 255, 255), 1)

cv2.imwrite('output_image.jpg', image)
# Open a window on the desktop showing the image
win = dlib.image_window()
image = cv2.resize(image,(1920,1080))
win.set_image(image)

# Wait until the user hits <enter> to close the window
dlib.hit_enter_to_continue()