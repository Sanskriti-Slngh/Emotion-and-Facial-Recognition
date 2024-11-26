import numpy as np
import pickle
import cv2
import os
from matplotlib import pyplot as plt

# Parameters to control the program
model_id = 23
doTraining = True
loadModel = False
normalize_input = True
data_file_name = "emotion_48x48.dat"
model_file_name = "Fnn" + str(model_id) + ".model"
momentum_factor = 0.9

# parameters
num_iter = 100
cost_samples = 100
mini_batch_size = 64
learning_rate = 0.02

## FNN parameters
#num_layers = 5
#num_neurons = [512, 256, 128, 64, 8]
#num_neurons = [4096, 1024, 256, 64, 8]
#act_funcs = ['relu', 'relu', 'relu', 'relu', 'sigmoid']

# model 1
if model_id == 1:
    num_layers = 3
    num_neurons = [4, 32, 7]
    act_funcs = ['relu', 'relu', 'sigmoid']
# training error = 20709 out of 32298(64.12 %)
# test error = 2439 out of 3589(67.96 %)
# training error = 18643 out of 32298(57.72 %)
# test error = 2315 out of 3589(64.50 %)

if model_id == 2:
    num_layers = 3
    num_neurons = [8, 32, 7]
    act_funcs = ['relu', 'relu', 'sigmoid']
# first number is 100 iterations and learning rate of 0.02
# second one is best tuned.
# training error = 20159 out of 32298(62.42 %)
# test error = 2321 out of 3589(64.67 %)
# training error = 17264 (53.45%)
# test error = 2300 (64.08 %)

if model_id == 3:
    num_layers = 3
    num_neurons = [16, 32, 7]
    act_funcs = ['relu', 'relu', 'sigmoid']
# training error = 19720 (61.06 %)
# test error = 2319 (64.61 %)
# training error = 16015 (49.59 %)
# test error = 2243 (62.50 %)

if model_id == 4:
    num_layers = 3
    num_neurons = [32, 32, 7]
    act_funcs = ['relu', 'relu', 'sigmoid']
# training error = 19327 (59.84 %)
# test error = 2317 (64.56 %)
# training error = 14423 (44.66 %)
# test error = 2271 (63.28 %)

if model_id == 5:
    num_layers = 3
    num_neurons = [64, 32, 7]
    act_funcs = ['relu', 'relu', 'sigmoid']
# training error = 18454 (57.14 %)
# test error = 2284 (63.64 %)
# training error = 12878 (39.87 %)
# test error = 2237 (62.33 %)

if model_id == 6:
    num_layers = 3
    num_neurons = [4, 64, 7]
    act_funcs = ['relu', 'relu', 'sigmoid']
# training error = 20204 (62.55 %)
# test error = 2298 (64.03 %)
# training error = 19507 out of 32298 (60.396929 %)
# test error = 2327 out of 3589 (64.837002 %)

if model_id == 7:
    num_layers = 3
    num_neurons = [8, 64, 7]
    act_funcs = ['relu', 'relu', 'sigmoid']
# training error = 19662 (60.88 %)
# test error = 2290 (63.80 %)
#training error = 18107 out of 32298 (56.062295 %)
#test error = 2231 out of 3589 (62.162162 %)

if model_id == 8:
    num_layers = 3
    num_neurons = [16, 64, 7]
    act_funcs = ['relu', 'relu', 'sigmoid']
# training error = 18823 (58.28 %)
# test error = 2251 (62.72 %)
# training error = 16263 out of 32298 (50.352963 %)
# test error = 2258 out of 3589 (62.914461 %)

if model_id == 9:
    num_layers = 3
    num_neurons = [32, 64, 7]
    act_funcs = ['relu', 'relu', 'sigmoid']
# training error = 19784 (61.25 %)
# test error = 2333 (65.00%)
# training error = 13473 out of 32298 (41.714657 %)
# test error = 2176 out of 3589 (60.629702 %)

if model_id == 10:
    num_layers = 3
    num_neurons = [64, 64, 7]
    act_funcs = ['relu', 'relu', 'sigmoid']
# training error = 17333 (53.67 %)
# test error = 2192 (61.08 %)
# training error = 13499 out of 32298 (41.795158 %)
# test error = 2311 out of 3589 (64.391195 %)

# Lets try some deeper networks
# model 9 refinement
if model_id == 21:
    num_layers = 4
    num_neurons = [32, 64, 32, 7]
    act_funcs = ['relu', 'relu', 'relu', 'sigmoid']
# training error = 14372 out of 32298 (44.498111 %)
# test error = 2238 out of 3589 (62.357203 %)

if model_id == 22:
    num_layers = 5
    num_neurons = [32, 128, 64, 32, 7]
    act_funcs = ['relu', 'relu', 'relu', 'relu', 'sigmoid']
#training error = 13345 out of 32298 (41.318348 %)
#test error = 2262 out of 3589 (63.025913 %)

if model_id == 23:
    num_layers = 6
    num_neurons = [256, 128, 64, 32, 16, 7]
    act_funcs = ['relu', 'relu', 'relu', 'relu', 'relu', 'sigmoid']

## useful functions
def sigmoid(x):
    return 1.0/(1+np.exp(-x))

def dsigmoid(x):
    y = sigmoid(x)
    return y*(1-y)

def relu(x):
    return np.maximum(x,0)

def drelu(x):
    return x > 0

## lists of costs to plot
costs = []

# Get the data from the file
with open(data_file_name, 'rb') as fin:
    data, dataTrain, dataTest, emotion = pickle.load(fin)

X_all = dataTrain['input']
Y_all = dataTrain['target']
print (X_all.shape)
print (Y_all.shape)

n = X_all.shape[0]
m = X_all.shape[1]

number_of_mini_batches = int(np.ceil(m * 1.0/mini_batch_size))

# model
model = {}
for i in range(num_layers):
    if i == 0:
        model['w0'] = np.random.randn(num_neurons[0], X_all.shape[0])*0.01
        model['b0'] = np.zeros((num_neurons[0],1))
    else:
        model['w' + str(i)] = np.random.randn(num_neurons[i], num_neurons[i-1])*0.01
        model['b' + str(i)] = np.zeros((num_neurons[i], 1))

#for layer in range(num_layers):
#    fig = plt.figure()
 #   plt.subplot(121)
  #  plt.imshow(model['w' + str(layer)], cmap=plt.cm.gray)
   # fig.suptitle('Weights for layer = %s before training' %(layer))

    #plt.subplot(122)
#    plt.plot(model['b' + str(layer)][:,0])
 #   fig.suptitle('Bias for layer = %s before training' % (layer))

if loadModel:
    with open(model_file_name, 'rb') as fin:
        model, _ = pickle.load(fin)
        print ("Starting from loaded model from %s" %(model_file_name))

def train(model, data, learning_rate):

    Y_all = data['target']
    X_all = data['input']
    z = {}
    a = {}
    z_loss = {}
    a_loss = {}

    # start form 0
    mom_speed = {}
    for i in range(num_layers):
        mom_speed['w' + str(i)] = np.zeros(model['w' + str(i)].shape)
        mom_speed['b' + str(i)] = np.zeros(model['b' + str(i)].shape)

    # normalize the input
    if normalize_input:
        X_all = X_all/255.0

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

            # Layer calculation
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
                    print ("I do not understand this function, " + act_funcs[q] + ".")
            #Finding y_hat
            y_hat = a[str(num_layers-1)]

            # Loss and cost
            if (bid == 0 and (iter%(num_iter/cost_samples)) == 0):
                # Layer calculation
                for q in range(num_layers):
                    if q == 0:
                        z_loss[str(q)] = np.dot(model['w' + str(q)], X_all) + model['b' + str(q)]
                    else:
                        z_loss[str(q)] = np.dot(model['w' + str(q)], a_loss[str(q - 1)]) + model['b' + str(q)]
                    if act_funcs[q] == 'relu':
                        a_loss[str(q)] = relu(z_loss[str(q)])
                    elif act_funcs[q] == 'sigmoid':
                        a_loss[str(q)] = sigmoid(z_loss[str(q)])
                    else:
                        print("I do not understand this function, " + act_funcs[q] + ".")
                y_hat_loss = a_loss[str(num_layers - 1)]
                loss = -Y_all * np.log(y_hat_loss)
                cost = 1.0 / m * np.sum(loss)
                costs.append(cost)
                print ("iteration = %s cost = %s" %(iter, cost))

            # Backward path
            dz = {}
            da = {}
            dw = {}
            db = {}
            for i in range(num_layers):
                layer = num_layers - i - 1
                if layer == (num_layers - 1):
                    dz[str(layer)] = y_hat - Y
                    dw[str(layer)] = 1.0/m1 * np.dot(dz[str(layer)], (a[str(layer-1)]).T)
                    db[str(layer)] = 1.0/m1 * np.sum(dz[str(layer)], axis=1, keepdims=True)
                else:
                    da[str(layer)] = np.dot(model['w' + str(layer+1)].T, dz[str(layer+1)])
                    dz[str(layer)] = da[str(layer)] * drelu(z[str(layer)])
                    if layer == 0:
                        dw[str(layer)] = 1.0/m1 * np.dot(dz[str(layer)], (X.T))
                    else:
                        dw[str(layer)] = 1.0 / m1 * np.dot(dz[str(layer)], (a[str(layer - 1)].T))
                    db[str(layer)] = 1.0/m1 * np.sum(dz[str(layer)], axis = 1, keepdims = True)
            # Ending my backward path and starting Updating parameters

            # update parameters
            if iter == 500:
                learning_rate = learning_rate/10

            for layer in range(num_layers):
                mom_speed['w'+str(layer)] = momentum_factor * mom_speed['w'+str(layer)] + dw[str(layer)]
                mom_speed['b' + str(layer)] = momentum_factor * mom_speed['b' + str(layer)] + db[str(layer)]
                model['w' + str(layer)] = model['w' + str(layer)] - (learning_rate*mom_speed['w'+str(layer)])
                model['b' + str(layer)] = model['b' + str(layer)] - (learning_rate*mom_speed['b'+str(layer)])

    # update the model
    fig = plt.figure()
    plt.plot(costs)
    plt.show()
    with open(model_file_name, 'wb') as fout:
        pickle.dump((model, emotion), fout)
        print ("Saving the model into %s" %(model_file_name))
    return model


def predict(model, X):
    # normalize the input
    if normalize_input:
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

def cal_total_prediction_error(Y,P):
    return np.sum(Y != P)/2

# lets train our model
if doTraining:
    trainError = cal_total_prediction_error(dataTrain['target'], predict(model, dataTrain['input']))
    testError = cal_total_prediction_error(dataTest['target'], predict(model, dataTest['input']))
    print("training error before model is trained = %d out of %d (%f %%)" % (
    trainError, dataTrain['target'].shape[1], trainError * 100.0 / dataTrain['target'].shape[1]))
    print("test error before model is trained = %d out of %d (%f %%)" % (
    testError, dataTest['target'].shape[1], testError * 100.0 / dataTest['target'].shape[1]))

    train(model, dataTrain, learning_rate)
else:
    with open(model_file_name, 'rb') as fin:
        model, _ = pickle.load(fin)
        print("Starting from loaded model from %s" % (model_file_name))

if False:
    for layer in range(num_layers):
        fig = plt.figure()
        plt.subplot(121)
        plt.imshow(model['w' + str(layer)], cmap=plt.cm.gray)
        fig.suptitle('Weights for layer = %s after training, layers 5, first - 512' %(layer))

        plt.subplot(122)
        plt.plot(model['b' + str(layer)][:,0])
        fig.suptitle('Bias for layer = %s after training, 5 layers, first - 512' % (layer))

    plt.show()

trainError = cal_total_prediction_error(dataTrain['target'], predict(model, dataTrain['input']))
testError = cal_total_prediction_error(dataTest['target'], predict(model, dataTest['input']))

print ("training error = %d out of %d (%f %%)" %(trainError, dataTrain['target'].shape[1], trainError*100.0/dataTrain['target'].shape[1]))
print ("test error = %d out of %d (%f %%)" %(testError, dataTest['target'].shape[1], testError*100.0/dataTest['target'].shape[1]))

## prediction for test
if False:
    p = predict(model, dataTest['input'])
    num_tests = p.shape[1]

    for i in range(num_tests):
        x = dataTest['input'][:,i]
        y = dataTest['target'][:,i]
        print (y)
        print (dataTest['fname'][i])
        print (p[:,i])
        rows = len(emotion)
        #for col in range(columns):
        for row in range(rows):
            if y[row]:
                emo = emotion[row]
                break

        for row in range(rows):
            if p[row,i]:
                break

        if emotion[row] != emo:
            print("FAIL: Predicted emotion is %s, real one is %s" % (emotion[row], emo))
            fname, flip_id, angle = dataTest['fname'][i]
            img = cv2.imread(fname)
            if (flip_id != 2):
                img = cv2.flip(img,flip_id)
            plt.imshow(img)
            plt.show()