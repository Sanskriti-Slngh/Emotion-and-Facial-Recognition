import numpy as np
import cv2
import matplotlib.image as mpimg
from matplotlib import pyplot as plt
import pickle

#persons = ["an2i", "at33", "boland", "bpm", "ch4f", "cheyer", "choon", \
          # "danieln", "glickman", "karyadi", "kawamura", "kk49", "megak", \
           #"mitchell", "night", "phoebe", "saavik", "steffi", "sz24", "tammo"]


#persons = ["an2i", "at33", "boland", "bpm", "ch4f", "cheyer", "choon", \
#          "danieln", "glickman", "karyadi"]

persons = ["an2i", "at33", "boland", "bpm", "ch4f", "cheyer", "choon", "danieln", "glickman"]


showImages = False
firstData = True
X = {}

for person_id, person in enumerate(persons):
    if showImages == True:
        fig = plt.figure()

    # create list of file names
    fnames = []

    for first in ("left", "right", "straight", "up"):
        for second in ("angry", "happy", "neutral", "sad"):
            for third in ("open", "sunglasses"):
                a = person + "_" + first + "_" + second + "_" + third + ".pgm"
    #            print a
                fnames.append(a)
    p = {}
    for i,fn in enumerate(fnames):
       # print i, fn
        if showImages == True:
            a = fig.add_subplot(4, 8, i+1)
        print 'data/faces/' + person + "/" + fn
        p["img" + str(i)] = cv2.resize(cv2.imread('data/faces/' + person + "/" + fn, 0),(32,32))
        if showImages == True:
            plt.imshow(p["img" + str(i)], cmap="gray")
        img = p["img" + str(i)]
        tmp = img[:,:].reshape(1, -1).T
        if firstData:
            X['input'] = tmp
            X['target'] = np.array([[person_id]])
            firstData = False
        else:
            X['input'] = np.hstack((X['input'], tmp))
            X['target'] = np.hstack((X['target'], np.array([[person_id]])))

print X['input'].shape
print X['target'].shape

print X['target'][0,65]

if showImages == True:
    plt.show()

data = {}
n = X['input'].shape[0]
m = X['input'].shape[1]

data['input'] = np.zeros([n*2, m*(m+1)/2], dtype=np.int8)
data['target'] = np.zeros([1,m*(m+1)/2], dtype=np.int8)

index = 0
for i in range(m):
    for j in range(i,m):
        data['input'][0:n,index] = X['input'][:,i]
        data['input'][n:2*n,index] = X['input'][:,j]
        data['target'][0,index] = X['target'][0,i] == X['target'][0,j]
        index = index + 1

print data['input'].shape
print data['target'].shape

# 2img = cv2.imread("data/faces/tammo/tammo_up_sad_sunglasses.pgm")
# tmp = img[:,:,0].reshape(1,-1).T
# if data == None:
  #  data = tmp
# else:
   # data = np.hstack((data, tmp))

#print data.shape

# one hot encoding
#number_of_persons = 20
#b = np.zeros((data['target'].shape[1], number_of_persons))
#b[np.arange(data['target'].shape[1]), data['target'][0,:]] = 1
#data['target'] = np.transpose(b)

#print data['target'].shape

## Split data into train and test buckets
## train = 90%, test is 10%
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

print dataTrain['input'].shape
print dataTrain['target'].shape
print dataTest['input'].shape
print dataTest['target'].shape

with open('processedData_X2P.dat', 'w') as fout:
    pickle.dump((data, dataTrain, dataTest, persons), fout)