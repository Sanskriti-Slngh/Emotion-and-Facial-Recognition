import numpy as np
import cv2
import matplotlib.image as mpimg
from matplotlib import pyplot as plt
import pickle
import face_recognition
import os

data_file_name= 'sadHappyData.dat'
data = {}
ShowImages = False
firstData = True
data['fname'] = []
fnames = []

file_names = []
name_of_persons = ["Briti", "Manish", "Tiya", "Kirti", "Trump", "Obama", "Tim", "Jane", "Charlie", "Mr.Bean", "Johnny"]
emotion = ["Angry", "Contempt", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

path = 'D:/Datasets/data/emotions/'

for (dirname, dirpath, files) in os.walk(path):
    print (dirname, dirpath, len(files))
    emo = (dirname.split("/")[-1])
    emo = emo.title()
    for fi in files:
        file_names.append((dirname + "/" + fi, emo))

for person in name_of_persons:
    for emo in ["Happy", "Sad", "Neutral"]:
        for num in range(13):
            file_name = "data/HSN/" + person + "/" + emo + "/" + person + " (" + str(num + 1) + ").jpg"
            print (file_name)
            file_names.append((file_name,emo))

# data from Cohn-Kanade
image_dir = 'data/Cohn-Kanade/CKPlus/extended-cohn-kanade-images/cohn-kanade-images'
emo_dir = 'data/Cohn-Kanade/CKPlus/Emotion_labels/Emotion'
for _ in os.listdir(image_dir):
    for __ in os.listdir(image_dir + '/' + _):
        if __ == '.DS_Store':
            continue
        print (__)
        for (dirpath, dirnames, filenames) in os.walk(emo_dir + '/' + _  + '/' +__):
            if len(filenames) == 0:
                continue
            else:
                emo_fname = emo_dir + '/' + _ + '/' + __ + '/' + filenames[0]
                print (_, __, emo_fname)
                with open(emo_fname, 'r') as f:
                    x = f.readlines()
                    a = int(float(x[0].rstrip()))
                    if a > 7:
                        print ("Something went wrong, the number is " + str(a) + ".")
                        exit()
                    else:
                        emoPer = emotion[a-1]
                for (dirpath, dirnames, filenames) in os.walk(image_dir + '/' + _ + '/' + __):
                    num_files = len(filenames)
                    ## first two are neutral filese
                if num_files > 3:
                    file_name = image_dir + '/' + _ + '/' + __ + '/' + _ + '_' + __ + '_' + '00000001' + '.png'
                    file_names.append((file_name, 'Neutral'))
                    file_name = image_dir + '/' + _ + '/' + __ + '/' + _ + '_' + __ + '_' + '00000002' + '.png'
                    file_names.append((file_name, 'Neutral'))
                    if num_files > 9:
                        file_name = image_dir + '/' + _ + '/' + __ + '/' + _ + '_' + __ + '_' + '000000' + str(num_files) + '.png'
                        file_names.append((file_name, emoPer))
                    else:
                        file_name = image_dir + '/' + _ + '/' + __ + '/' + _ + '_' + __ + '_' + '0000000' + str(num_files) + '.png'
                        file_names.append((file_name, emoPer))
                    if (num_files-1) > 9:
                        file_name = image_dir + '/' + _ + '/' + __ + '/' + _ + '_' + __ + '_' + '000000' + str(num_files-1) + '.png'
                        file_names.append((file_name, emoPer))
                    else:
                        file_name = image_dir + '/' + _ + '/' + __ + '/' + _ + '_' + __ + '_' + '0000000' + str(num_files-1) + '.png'
                        file_names.append((file_name, emoPer))

# getting my Y or output
# emotion = ["Angry", "Contempt", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

def get_y(emo):
    if emo == 'Angry':
        return np.array([[1,0,0,0,0,0,0,0]])
    if emo == 'Contempt':
        return np.array([[0,1,0,0,0,0,0,0]])
    if emo == 'Disgust':
        return np.array([[0,0,1,0,0,0,0,0]])
    if emo == 'Fear':
        return np.array([[0,0,0,1,0,0,0,0]])
    if emo == 'Happy':
        return np.array([[0,0,0,0,1,0,0,0]])
    if emo == 'Sad':
        return np.array([[0,0,0,0,0,1,0,0]])
    if emo == 'Surprise':
        return np.array([[0,0,0,0,0,0,1,0]])
    if emo == 'Neutral':
        return np.array([[0,0,0,0,0,0,0,1]])
    print ("I do not understand this emotion, " + emo + ".")


pic_num = 0
for file,emo in file_names:
    print (file)
    color_image = cv2.imread(file)
    img = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    print (img.shape)
    face_locations = face_recognition.face_locations(img)
    for (top,right,bottom,left) in face_locations:
        #cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)
        #cv2.rectangle(img, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        #font = cv2.FONT_HERSHEY_DUPLEX
        #cv2.putText(img, emo, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        curr_x = cv2.resize(img[top:bottom, left:right], (64,64))
        pic_num += 1
        if ShowImages == True and pic_num >= 1000 and pic_num < 1050:
            plt.imshow(curr_x, cmap='gray')
            plt.show()
        f = np.transpose(curr_x.reshape(1,-1))
        if firstData:
            data['input'] = f
            data['target'] = np.transpose(get_y(emo))

            firstData = False
        else:
            data['input'] = np.hstack((data['input'], f))
            data['target'] = np.hstack((data['target'], np.transpose(get_y(emo))))
        data['fname'].append((file,2, 0))

        # get more data by flipping
        rows, cols = curr_x.shape
        for angle in (-30,-20,-10,0,10,20,30):
            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
            curr_x_rotated = cv2.warpAffine(curr_x, M, (cols, rows))
            f = np.transpose(curr_x_rotated.reshape(1, -1))
            data['input'] = np.hstack((data['input'], f))
            data['target'] = np.hstack((data['target'], np.transpose(get_y(emo))))
            data['fname'].append((file, 0, angle))
            i=1
            curr_x_rotated = cv2.flip(curr_x_rotated, i)
            f = np.transpose(curr_x_rotated.reshape(1, -1))
            data['input'] = np.hstack((data['input'], f))
            data['target'] = np.hstack((data['target'], np.transpose(get_y(emo))))
            data['fname'].append((file,i, angle))


print (data['input'].shape)
print (data['target'].shape)
print (data['target'][:, 10])

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
dataTrain['fname'] = []
for i in range(num_of_train_examples):
#    dataTrain['fname'] = []
#    print (data['fname'][shuffled_indices[i]])
#    print (data['target'][:,shuffled_indices[i]])
    dataTrain['fname'].append(data['fname'][shuffled_indices[i]])

dataTest = {}
dataTest['input'] = data['input'][:, shuffled_indices[num_of_train_examples:]]
dataTest['target'] = data['target'][:, shuffled_indices[num_of_train_examples:]]
dataTest['fname'] = []
for i in range(data['input'].shape[1]-num_of_train_examples):
#    dataTest['fname'] = []
#    print (data['fname'][shuffled_indices[num_of_train_examples+i]])
#    print (data['target'][:,shuffled_indices[num_of_train_examples+i]])
    dataTest['fname'].append(data['fname'][shuffled_indices[num_of_train_examples+i]])

print(dataTrain['input'].shape)
print(dataTrain['target'].shape)
print(dataTest['input'].shape)
print(dataTest['target'].shape)

with open(data_file_name, 'wb') as f:
    pickle.dump((data, dataTrain, dataTest, emotion), f)
