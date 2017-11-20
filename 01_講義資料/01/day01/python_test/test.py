#!/usr/bin/python

import numpy as np
import logRegres

dataArr,labelMat=logRegres.loadDataSet()
weights=logRegres.stocGradAscent1(np.array(dataArr),labelMat)
logRegres.plotBestFit(weights)

