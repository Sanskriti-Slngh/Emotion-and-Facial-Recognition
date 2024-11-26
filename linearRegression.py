import numpy as np
import pickle
import cv2
import os
from matplotlib import pyplot as plt

# Parameters to control the program
doTraining = True
loadModel = False
normalize_input = False
data_file_name = "processedData.dat"
model_file_name = "modelLinearRegression_MKTB.dat"
#data_file_name = "processedData_data_faces.dat"
#model_file_name = "modelLinearRegression_data_faces.dat"


## lists of costs to plot
costs = []

# Get the data from the file
with open(data_file_name, 'rb') as fin:
    data, dataTrain, dataTest, persons, known_people_names = pickle.load(fin)

known_people_names.append('Stranger')
X_all = dataTrain['input']
Y_all = dataTrain['target']
print (X_all.shape)
print (Y_all.shape)

n = X_all.shape[0]
m = X_all.shape[1]

learning_rate = 0.02
num_iter = 100000
cost_samples = 100
mini_batch_size = 64
number_of_mini_batches = int(np.ceil(m * 1.0/mini_batch_size))

# initializing parameters
w = np.random.randn(Y_all.shape[0],n) * 0.01
b = np.zeros((Y_all.shape[0],1))
model = {}
model['weight'] = w
model['bias'] = b

if loadModel:
    with open(model_file_name, 'rb') as fin:
        model = pickle.load(fin)
        print ("Starting from loaded model from %s" %(model_file_name))

def train(model, data):
    w = model['weight']
    b = model['bias']

    Y_all = data['target']
    X_all = data['input']

    # normalize the input
    if normalize_input:
        X_all = X_all/np.max(X_all)

    for iter in range(num_iter):
        for bid in range(number_of_mini_batches):
            start_of_batch = bid*mini_batch_size
            if bid == (number_of_mini_batches-1):
                end_of_batch = m
            else:
                end_of_batch = start_of_batch+mini_batch_size
            X = X_all[:,start_of_batch:end_of_batch]
            Y = Y_all[:,start_of_batch:end_of_batch]
            m1 = X.shape[1]
            z = np.dot(w, X) + b
            y_hat = 1.0/(1+np.exp(-z))
        #    loss = -(Y*np.log(y_hat) + ((1-Y)*np.log(1 - y_hat)))
            loss = -Y * np.log(y_hat)
            cost = 1.0/m1 * np.sum(loss)
            if (bid == 0 and (iter%(num_iter/cost_samples)) == 0):
                costs.append(cost)
                print ("iteration = %s cost = %s" %(iter, cost))

            dz = y_hat - Y
            dw = 1.0/m1 * np.dot(dz, X.T)
            db = 1.0/m1 * np.sum(dz, axis=1, keepdims=True)

            # update parameters
            w = w - (learning_rate*dw)
            b = b - (learning_rate*db)

    # update the model
    model['weight'] = w
    model['bias'] = b
    fig = plt.figure()
    plt.plot(costs)
    plt.show()
    with open(model_file_name, 'wb') as fout:
        pickle.dump((model, known_people_names), fout)
        print ("Saving the model into %s" %(model_file_name))
    return model


def predict(model, X):
    w = model['weight']
    b = model['bias']

    # normalize the input
    if normalize_input:
        X = X/np.max(X)

    z = np.dot(w, X) + b   ;# [20,m]
    y_hat = 1.0 / (1 + np.exp(-z)) ;# [20,m]
    max_y_hat = np.max(y_hat, axis=0, keepdims=True)
    print (y_hat[:,10])
    p = y_hat == max_y_hat
#    p = y_hat > 0.5
    return(p)

def cal_total_prediction_error(Y,P):
    return np.sum(Y != P)/2

def cal_total_prediction_error_old(Y, P):
    print (Y.shape)
    Y1 = np.argmax(Y, axis=0); # one hot encoding to person's id
    print (Y1.shape)
    P1 = np.argmax(P, axis=0); # one hot encoding to person's id (randomly break the tie)
    #print "Y1 is %s" %(Y1[1:5])
    #print P1[1:5]
    return np.sum(Y1 != P1); # count the error


# lets train our model
if doTraining:
    train(model, dataTrain)
else:
    with open(model_file_name, 'rb') as fin:
        model = pickle.load(fin)
        print("Starting from loaded model from %s" % (model_file_name))

trainError = cal_total_prediction_error(dataTrain['target'], predict(model, dataTrain['input']))
testError = cal_total_prediction_error(dataTest['target'], predict(model, dataTest['input']))

print ("training error = %d out of %d (%f)" %(trainError, dataTrain['target'].shape[1], trainError*100.0/dataTrain['target'].shape[1]))
print ("test error = %d out of %d (%f)" %(testError, dataTest['target'].shape[1], testError*100.0/dataTest['target'].shape[1]))

## prediction for test
if True:
    p = predict(model, dataTest['input'])
    num_tests = p.shape[1]

    for i in range(num_tests):
        x = dataTest['input'][:,i]
        y = dataTest['target'][:,i]
        print (y)
        print (dataTest['fname'][i])
        print (p[:,i])
        rows = len(known_people_names)
        #for col in range(columns):
        for row in range(rows):
            if y[row]:
                person = known_people_names[row]
                break

        for row in range(rows):
            if p[row,i]:
                print ("Predicted person is %s, real one is %s" %(known_people_names[row], person))
                break

        if known_people_names[row] != person:
            print (dataTest['fname'][i])
            img = cv2.imread(dataTest['fname'][i])
            plt.imshow(img)
            plt.show()


print (dataTest['input'][:,10])
p = predict(model, dataTest['input'])
print (p[:,10])