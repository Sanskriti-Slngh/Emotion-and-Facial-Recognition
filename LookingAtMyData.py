import numpy as np
import cv2
from matplotlib import pyplot as plt
import pickle
import skimage.measure
from keras.utils import np_utils
import os
import face_recognition
import imutils

data_file_name = "emotion_48x48_aug_split90.dat"
enable_augmentation = True
X = []
Y = []
file = "D:/Datasets/data/fer2013/fer2013.csv"
emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
emo_name_to_number = {'Angry':0, 'Disgust':1, 'Fear':2, 'Happy':3, 'Sad':4, 'Surprise':5, 'Neutral':6}
#0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral

cnt = 0
with open(file, 'r') as f:
    for line in f:
        if cnt < 1:
            cnt = cnt + 1
            continue
        #print(line)
        emotion, p, _ = line.split(',')
        emotion = int(emotion)
        Y.append([emotion])
        emo = emotions[emotion]
        pixels = [int(x) for x in p.split(' ')]
        X.append(pixels)
        #image = np.reshape(np.array(pixels), (48,48))
        #plt.imshow(image, cmap='gray')
        #plt.title(emo)
        #plt.show()
        if enable_augmentation:
            image = np.reshape(np.array(pixels), (48, 48))
            image_pic = np.zeros((48,48,3))
            image_pic[:, :, 0] = image
            image_pic[:, :, 1] = image
            image_pic[:, :, 2] = image
            for angle in (-20,20):
                image_r = imutils.rotate(image_pic, angle)
                image_r = np.reshape(image_r[:,:,0], (-1))
                image_r.tolist()
                X.append(image_r)
                Y.append([emotion])
            flipped_image = cv2.flip(image,1)
            pixels = np.reshape(flipped_image,(-1)).tolist()
            X.append(pixels)
            Y.append([emotion])
            image_pic = np.zeros((48,48,3))
            image_pic[:, :, 0] = flipped_image
            image_pic[:, :, 1] = flipped_image
            image_pic[:, :, 2] = flipped_image
            for angle in (-20,20):
                image_r = imutils.rotate(image_pic, angle)
                image_r = np.reshape(image_r[:,:,0], (-1))
                image_r.tolist()
                X.append(image_r)
                Y.append([emotion])
                # image_to_show = np.reshape(np.array(image_r), (48, 48))
                # plt.imshow(image_to_show, cmap='gray')
                # plt.title(emo)
                # plt.show()
        cnt = cnt + 1

# our pictures
dir_path = 'data/HSN/'
for (dirname, dirpath, files) in os.walk(dir_path):
    print (dirname, dirpath, len(files))
    for file in files:
        dirname = dirname.replace('\\', '/')
        fname = dirname + '/' + file
        print ("Reading %s file" %(fname))
        image = cv2.imread(fname)
        face_locations = face_recognition.face_locations(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h,w = img.shape
        margin = 10
        for (top, right, bottom, left) in face_locations:
            curr_x = cv2.resize(img[max(0, top - margin):min(h, bottom + margin), max(0, left - margin):min(w, right + margin)], (48,48))
            pixels = np.reshape(curr_x,(-1)).tolist()
            X.append(pixels)
            emo = dirname.split('/')[-1]
            emotion = emo_name_to_number[emo]
            Y.append([emotion])
            image = np.reshape(np.array(pixels), (48,48))
            #plt.imshow(image, cmap='gray')
            #plt.title(emo)
            #plt.show()
            if enable_augmentation:
                image_pic = np.zeros((48, 48, 3))
                image_pic[:, :, 0] = curr_x
                image_pic[:, :, 1] = curr_x
                image_pic[:, :, 2] = curr_x
                for angle in (-20, 20):
                    image_r = imutils.rotate(image_pic, angle)
                    image_r = np.reshape(image_r[:, :, 0], (-1))
                    image_r.tolist()
                    X.append(image_r)
                    Y.append([emotion])
                curr_x = cv2.flip(curr_x,1)
                pixels = np.reshape(curr_x,(-1)).tolist()
                X.append(pixels)
                Y.append([emotion])
                image_pic = np.zeros((48, 48, 3))
                image_pic[:, :, 0] = curr_x
                image_pic[:, :, 1] = curr_x
                image_pic[:, :, 2] = curr_x
                for angle in (-20, 20):
                    image_r = imutils.rotate(image_pic, angle)
                    image_r = np.reshape(image_r[:, :, 0], (-1))
                    image_r.tolist()
                    X.append(image_r)
                    Y.append([emotion])
                    # image_to_show = np.reshape(np.array(image_r), (48, 48))
                    # plt.imshow(image_to_show, cmap='gray')
                    # plt.title(emo)
                    # plt.show()

X = np.array(X)
X = np.transpose(X)
print(X.shape)

Y = np.array(Y)
Y = np_utils.to_categorical(Y, num_classes=7)
Y = np.transpose(Y)

data = {}
data['input'] = X
data['target'] = Y

## Split data into train and test buckets
## train = 80%, test is 20%
train2TestRatio = 0.9
num_of_train_examples = int(data['input'].shape[1]*train2TestRatio)

shuffled_indices = np.arange(data['input'].shape[1])
np.random.shuffle(shuffled_indices)
#print shuffled_indices[0:9]

dataTrain = {}
dataTrain['input'] = data['input'][:, shuffled_indices[0:num_of_train_examples]]
dataTrain['target'] = data['target'][:, shuffled_indices[0:num_of_train_examples]]

dataTest = {}
dataTest['input'] = data['input'][:, shuffled_indices[num_of_train_examples:]]
dataTest['target'] = data['target'][:, shuffled_indices[num_of_train_examples:]]

print(dataTrain['input'].shape)
print(dataTrain['target'].shape)
print(dataTest['input'].shape)
print(dataTest['target'].shape)

with open(data_file_name, 'wb') as f:
    pickle.dump((dataTrain, dataTest, emotions), f)