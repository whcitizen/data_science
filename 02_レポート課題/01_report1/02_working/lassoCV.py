import numpy as np
import pandas as pd
from sklearn import cross_validation, linear_model

N = 10000
nfeat = 53

train_data = pd.read_csv("train.csv", header=None)
test_data = pd.read_csv("test.csv", header=None)

X_train = train_data.iloc[:, 0:53].values
Y_train = train_data.iloc[:, 53].values.T

clf_lassoCV = linear_model.LassoCV()
clf_lassoCV.fit(X_train, Y_train)

print(clf_lassoCV.alpha_)

"""
w = clf_lassoCV.coef_

yHatTrain = np.dot(X_train, w)

print("Training error ", np.mean(np.abs(Y_train - yHatTrain.T)))

yHatTest = np.dot(np.array(test_data),w)
for i in range(0,len(yHatTest)):
    if yHatTest[i] < 0:
        yHatTest[i] = 0

np.savetxt('result_lasso_cv.txt', yHatTest)

"""