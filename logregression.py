# -*- coding: utf-8 -*-
"""
Created on Thu May 26 16:35:39 2022

@author: m211991
"""

import logfunction as lf
import numpy as np
import pandas as pd
from patsy import dmatrices
from sklearn.linear_model import LogisticRegression

### create data set
pts = 1000

# creating two circle distributions with radius r1 and r2
# with n number of random pts and an error term for radius, e
# circle_data(n,r1,r2,e)

X,y = lf.circle_data(1000,1,2,1)

### visualize data
lf.data_plot(X[:,0],X[:,1],y)

### regression

# fit1
fit1 = LogisticRegression(random_state=0,C =1e10).fit(X, y)
odd_ratio1 = np.exp(fit1.coef_)
y_fit1 = fit1.predict(X)

lf.data_plot(X[:,0],X[:,1],y_fit1)

#fit2
X2 = np.column_stack([X,np.square(X)])
fit2 = LogisticRegression(random_state=0,C =1e10).fit(X2, y)
odd_ratio2 = np.exp(fit2.coef_)
y_fit2 = fit2.predict(X2)
lf.data_plot(X2[:,0],X2[:,1],y_fit2)


