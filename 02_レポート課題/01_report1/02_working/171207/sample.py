import numpy as np
import pandas as pd
import regression as reg

N = 10000 
nfeat = 53

train_data = pd.read_csv("train.csv",header=None)
test_data = pd.read_csv("test.csv",header=None)

X = np.matrix(train_data)[:,:nfeat]
y = np.matrix(train_data)[:,nfeat].T #This is the target

##NOTE: unlike in previous notebooks, we are using row-wise sample matrix.

XTrain = X[:N,:] #use the first N samples for training
yTrain = y[:,:N]
XVal = X[N:,:] #use the rests for validation
yVal = y[:,N:]

w = reg.ridgeRegres(XTrain,yTrain,lam=0.1) #linear regression

yHatTrain = np.dot(XTrain,w)
yHatVal = np.dot(XVal,w)

print("Training error ", np.mean(np.abs(yTrain - yHatTrain.T)))
print("Validation error ", np.mean(np.abs(yVal - yHatVal.T)))

yHatTest = np.dot(np.array(test_data),w)
np.savetxt('result.txt', yHatTest)
