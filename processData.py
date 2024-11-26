import numpy as np
import cv2
import matplotlib.image as mpimg
from matplotlib import pyplot as plt
import pickle
import face_recognition

persons = ["an2i", "at33", "boland", "bpm", "ch4f", "cheyer", "choon", \
           "danieln", "glickman", "karyadi", "kawamura", "kk49", "megak", \
           "mitchell", "night", "phoebe", "saavik", "steffi", "sz24", "tammo"]

known_peoples = [\
    {'name':'Manish', 'age':40, 'num_pics':15},
    {'name':'Kirti', 'age':35, 'num_pics':15},
    {'name':'Tiya', 'age': 11, 'num_pics': 30},
    {'name': 'Briti', 'age': 7, 'num_pics': 30},
    {'name': 'Hasini', 'age': 11, 'num_pics': 15},
    {'name': 'Hari', 'age': 38, 'num_pics': 15},
    {'name': 'Hm', 'age': 40, 'num_pics': 15},
    {'name': 'Tanishq', 'age': 11, 'num_pics': 15},
    {'name': 'Pranavi', 'age': 10, 'num_pics': 15},
    {'name': 'Saanvi', 'age': 5, 'num_pics': 15},
    {'name': 'Sarvin', 'age': 6, 'num_pics': 15},
    {'name': 'Shobit', 'age': 40, 'num_pics': 15},
    {'name': 'Dimple', 'age': 39, 'num_pics': 15},
    {'name': 'Vineet', 'age': 47, 'num_pics': 15},
    {'name': 'Parul', 'age': 40, 'num_pics': 15},
    {'name': 'Obama', 'age':53, 'num_pics': 15},
    {'name': 'Trump', 'age': 71, 'num_pics': 15},
    {'name': 'Arya', 'age': 13, 'num_pics': 15},
    {'name': 'Robert', 'age': 40, 'num_pics': 15},
    {'name': 'Angelina', 'age': 40, 'num_pics': 15},
    {'name': 'Dwayne', 'age': 40, 'num_pics': 15},
    {'name': 'Scarlett', 'age': 40, 'num_pics': 15},
    {'name':'Steve', 'age':56, 'num_pics':15},
    {'name': 'Bush', 'age': 70, 'num_pics': 12}]

showImages = False
firstData = True
data = {}
number_of_known_people = len(known_peoples)
all_known = False
data_file_name= 'processedData.dat'

for person_id, person in enumerate(persons):
    if showImages == True:
        fig = plt.figure()

    # create list of file names
    fnames = []

    for first in ("left", "right", "straight", "up"):
        for second in ("angry", "happy", "neutral", "sad"):
            for third in ("open", "sunglasses"):
                a = person + "_" + first + "_" + second + "_" + third + ".pgm"
                fnames.append(a)

    p = {}
    for i,fn in enumerate(fnames):
       # print i, fn
        if showImages == True:
            a = fig.add_subplot(4, 8, i+1)
        print ('data/faces/' + person + "/" + fn)
        p["img" + str(i)] = cv2.imread('data/faces/' + person + "/" + fn)
        if showImages == True:
            plt.imshow(p["img" + str(i)], cmap="gray")
        face_locations = face_recognition.face_locations(p["img" + str(i)])
        if len(face_locations):
            tmp = np.transpose(np.array([face_recognition.face_encodings(p["img" + str(i)])[0]]))
            print ("Generataing Emmbedding for", person + "/" + fn)
        else:
            continue

        if firstData:
            data['input'] = tmp
            data['fname'] = ['data/faces/' + person + "/" + fn]
            data['target'] = np.array([[number_of_known_people]])
            firstData = False
        else:
            data['fname'].append('data/faces/' + person + "/" + fn)
            data['input'] = np.hstack((data['input'], tmp))
            data['target'] = np.hstack((data['target'], np.array([[number_of_known_people]])))

if showImages == True:
    plt.show()

number_of_persons = number_of_known_people+1

# known peoples embedding
print("ending, and starting known faces")
for person_id, known_people in enumerate(known_peoples):
    known_person = known_people['name']
    num_pics = known_people['num_pics']
    for i in range(num_pics):
        f_img = 'data/faces/' + known_person.lower() + '/' + known_person + ' (' + str(i + 1) + ').jpg'
        img = face_recognition.load_image_file(f_img)
        face_locations = face_recognition.face_locations(img)
        if len(face_locations):
            print("found face in ", f_img)
            tmp = np.transpose(np.array([face_recognition.face_encodings(img)[0]]))
            data['input'] = np.hstack((data['input'], tmp))
            data['target'] = np.hstack((data['target'], np.array([[person_id]])))
            data['fname'].append('data/faces/' + known_person.lower() + '/' + known_person + ' (' + str(i + 1) + ').jpg')

print (data['input'].shape)
print (data['target'].shape)

print (number_of_persons)

b = np.zeros((data['target'].shape[1], number_of_persons))
b[np.arange(data['target'].shape[1]), data['target'][0,:]] = 1
data['target'] = np.transpose(b)

print(data['target'].shape)

## Split data into train and test buckets
## train = 70%, test is 30%
train2TestRatio = 0.7
num_of_train_examples = int(data['input'].shape[1]*train2TestRatio)

shuffled_indices = np.arange(data['input'].shape[1])
np.random.shuffle(shuffled_indices)
#print shuffled_indices[0:9]

dataTrain = {}
dataTrain['input'] = data['input'][:, shuffled_indices[0:num_of_train_examples]]
dataTrain['target'] = data['target'][:, shuffled_indices[0:num_of_train_examples]]
dataTrain['fname'] = []
for i in range(num_of_train_examples):
#    print (data['fname'][shuffled_indices[i]])
#    print (data['target'][:,shuffled_indices[i]])
    dataTrain['fname'].append(data['fname'][shuffled_indices[i]])

dataTest = {}
dataTest['input'] = data['input'][:, shuffled_indices[num_of_train_examples:]]
dataTest['target'] = data['target'][:, shuffled_indices[num_of_train_examples:]]
dataTest['fname'] = []
for i in range(data['input'].shape[1]-num_of_train_examples):
#    print (data['fname'][shuffled_indices[num_of_train_examples+i]])
#    print (data['target'][:,shuffled_indices[num_of_train_examples+i]])
    dataTest['fname'].append(data['fname'][shuffled_indices[num_of_train_examples+i]])

print(dataTrain['input'].shape)
print(dataTrain['target'].shape)
print(dataTest['input'].shape)
print(dataTest['target'].shape)

known_people_names = []
for known_people in known_peoples:
    known_people_names.append(known_people['name'])

with open(data_file_name, 'wb') as fout:
    pickle.dump((data, dataTrain, dataTest, persons, known_people_names), fout)