import cv2
from matplotlib import pyplot as plt
import numpy as np
import pickle
from keras.utils import np_utils
import imutils

skipLines = 1
lineCount = 0

#0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

data_file_name = "fer2013_images_aug.dat"
data = {}
data['input'] = []
data['target'] = []

with open('data/fer2013/fer2013.csv', 'r') as fin:
    for line in fin:
        line = line.rstrip('\n')
        if lineCount < skipLines:
            lineCount += 1
            continue
        # emotion, image pixels, usage
        emotion, pixels, usage = line.split(',')
        image = [int(x) for x in pixels.split(' ')]
        #print (image)
        data['input'].append(image)
        data['target'].append(emotion)
        # flip, rotate (-20, -10, 10, 20)
        image = np.reshape(np.array(image), (48,48))
        image_f = cv2.flip(image,1)
        image_f = np.reshape(image, (-1))
        image_f.tolist()
        data['input'].append(image_f)
        data['target'].append(emotion)
        image_pic = np.zeros((48,48,3))
        image_pic[:, :, 0] = image
        image_pic[:, :, 1] = image
        image_pic[:, :, 2] = image
        #print (image.shape)
        #print (image_pic.shape)
        #for angle in np.arange(-30, 30, 10):
        #    image_r = imutils.rotate(image_pic, angle)
        #    image_r = np.reshape(image_r[:,:,0], (-1))
        #    image_r.tolist()
        #    data['input'].append(image_r)
        #    data['target'].append(emotion)
            #print (image_r.shape)
            #plt.imshow(image_r[:,:,0], cmap='gray')
            #plt.show()
        #exit()
        #print (image.shape)
        lineCount += 1
        if (lineCount%100 == 0):
            print (lineCount)

data['input'] = np.transpose(np.array(data['input']))
data['target'] = np.transpose(np_utils.to_categorical(data['target'],7))

print ("Total number of pictures %s" %(lineCount-1))
print (data['input'].shape)
print (data['target'].shape)

index=300
image = np.reshape(data['input'][:,index],(48,48))
plt.imshow(image, cmap='gray')
plt.title(str(data['target'][:,index]))
plt.show()

# split the data into train/test
train2TestRatio = 0.8
num_of_train_examples = int(data['input'].shape[1]*train2TestRatio)

shuffled_indices = np.arange(data['input'].shape[1])
np.random.shuffle(shuffled_indices)

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

