import numpy as np
import pandas as pd
from sklearn import cross_validation, linear_model

N = 10000
nfeat = 53

train_data = pd.read_csv("train.csv",header=None)
test_data = pd.read_csv("test.csv",header=None)

X = train_data.iloc[:, 0:53].values
Y = train_data.iloc[:, :53].values.T

X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y, test_size=0.2, random_state=0)

clf_lasso = linear_model.lasso(alpha=1.0)
clf_lasso.fit(X_train, Y_train)

w = clf_lasso.coef_

yHatTrain = np.dot(X_train,w)

yHatVal = np.dot(X_test,w)

print("Training error ", np.mean(np.abs(Y_train - yHatTrain.T)))
print("Validation error ", np.mean(np.abs(Y_test - yHatVal.T)))

yHatTest = np.dot(np.array(test_data),w)
for i in range(0,len(yHatTest)):
    if yHatTest[i] < 0:
        yHatTest[i] = 0

np.savetxt('result_lasso.txt', yHatTest)