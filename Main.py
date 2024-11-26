import numpy as np
import face_recognition
import dlib
from skimage import io
import cv2
import pickle
import time

emo_model_number = 1

## parameters
# original model
face_model_name = 'modelLinearRegression_MKTB.dat'
# Original model
emotion_model_file = 'modelFNNSadHappy.dat'
num_layers = 5
act_funcs = ['relu', 'relu', 'relu', 'relu', 'sigmoid']

#model1
if emo_model_number == 1:
    emotion_model_file = 'modelFNNSadHappy_1.dat'
    num_layers = 3
    act_funcs = ['relu', 'relu', 'sigmoid']

# True and False
firstData = True

# functions
# Predict using model
def predict(model, X):
    w = model['weight']
    b = model['bias']

    z = np.dot(w, X) + b
    y_hat = 1.0 / (1 + np.exp(-z))
    max_y_hat = np.max(y_hat, axis=0, keepdims=True)
    p = y_hat == max_y_hat
    return(p)

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
with open(face_model_name, 'rb') as fin:
    model, known_people = pickle.load(fin)

with open(emotion_model_file, 'rb') as f:
    emo_model, all_emo = pickle.load(f)


image = cv2.imread("test_pics/pics (66).jpg")
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

# Get face embedding for each face location
face_embedings = face_recognition.face_encodings(image, face_locations)

# Create input X from the face embeddings
X = np.transpose(np.array(face_embedings))

# main code
time0 = time.time()
print ('Calling predict person')
P = predict(model, X)
print ('Returning predict person')
time1 = time.time()
print("The time taken in finding person is %s" %(time1 - time0))
face_names = []
# 7. Now iterate over P to create information to display
rows, columns = P.shape
for col in range(columns):
    for row in range(rows):
        if P[row, col]:
            person = known_people[min(len(known_people)-1,row)]
            face_names.append(person)
            break

# finding the emotion for each face
for (top,right,bottom,left) in face_locations:
    curr_x = cv2.resize(img[top:bottom, left:right], (64,64))
    f = np.transpose(curr_x.reshape(1,-1))
    if firstData:
        X = f
        firstData = False
    else:
        X = np.hstack((X, f))

time0 = time.time()
print ('Calling predict emotions')
A = predict_emo(emo_model, X)
print ('Returning predict emotions')
time1 = time.time()
print("The time taken in finding emotions is %s" %(time1 - time0))
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
for (top, right, bottom, left), name, emo in zip(face_locations, face_names, emo_names):
    cv2.rectangle(image, (left, top), (right, bottom), (0,0,255), 2)
    cv2.rectangle(image, (left, bottom), (right, bottom+40), (0,0,255), cv2.FILLED)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(image, name, (left + 6, bottom+20), font, 0.8, (255, 255, 255), 1)
    cv2.putText(image, emo, (left + 6, bottom+40), font, 0.8, (255, 255, 255), 1)

cv2.imwrite('output_image.jpg', image)
# Open a window on the desktop showing the image
win = dlib.image_window()
win.set_image(image)

# Wait until the user hits <enter> to close the window
dlib.hit_enter_to_continue()
