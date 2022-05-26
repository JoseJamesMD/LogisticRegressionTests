import logfunction as lf
import numpy as np
import pandas as pd
from patsy import dmatrices
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings("ignore")

### create data set
# creating two circle distributions with radius r1 and r2
# with n number of random pts and an error term for radius, e
# circle_data(n,r1,r2,e)

n = 1000
r1 = 1
r2 = 2
e = 1

X,y = lf.circle_data(n,r1,r2,e)

### visualize data
lf.data_plot(X[:,0],X[:,1],y)

### regression

# sklearn automatically applies a variable C which is the inverse a regularization term λ. Increasing C minimizes the regularization.

# fit
fit1 = LogisticRegression(random_state=0,C =1e10).fit(X, y)

# odds ratio
odd_ratio1 = np.exp(fit1.coef_)
print('odds ratio1 \n x1: ' + str(round(odd_ratio1[0][0],2)) +\
      '\n x2: ' +str(round(odd_ratio1[0][1],2)))

# predict and plot
y_fit1 = fit1.predict(X)
lf.data_plot(X[:,0],X[:,1],y_fit1)

#fit2

# non linear parameterization --> add squared parameters X2 = [x1 x2 x1^2 x2^2]
X2 = np.column_stack([X,np.square(X)])

#fit
fit2 = LogisticRegression(random_state=0,C =1e10).fit(X2, y)

# odds ratio
odd_ratio2 = np.exp(fit2.coef_)
print('\nodds ratio2 \n x1: ' + str(round(odd_ratio2[0][0],2)) +\
      '\n x2: ' +str(round(odd_ratio2[0][1],2)) +\
          '\n x1^2: ' +str(round(odd_ratio2[0][2],2)) +\
              '\n x2^2: ' +str(round(odd_ratio2[0][3],2)))

# predict and plot
y_fit2 = fit2.predict(X2)
lf.data_plot(X2[:,0],X2[:,1],y_fit2)
