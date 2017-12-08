import numpy as np
import pandas as pd
from sklearn import cross_validation, linear_model

N = 10000
nfeat = 53

train_data = pd.read_csv("train.csv", header=None)
test_data = pd.read_csv("test.csv", header=None)

X_train = train_data.iloc[:, 0:53].values
Y_train = train_data.iloc[:, 53].values.T

clf_Elastic_CV = linear_model.ElasticNetCV(fit_intercept=True)
clf_Elastic_CV.fit(X_train, Y_train)

w = clf_Elastic_CV.coef_

yHatTrain = np.dot(X_train, w)

print("Training error ", np.mean(np.abs(Y_train - yHatTrain.T)))

yHatTest = np.dot(np.array(test_data),w)
for i in range(0,len(yHatTest)):
    if yHatTest[i] < 0:
        yHatTest[i] = 0

np.savetxt('result_ElasticNet_cv.txt', yHatTest)