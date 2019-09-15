#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 18:06:15 2019

@author: anilchivukula
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
import math
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
from matplotlib import style
dataset = pd.read_csv("AAPL.csv")
dfreg = dataset.loc[:,['Adj Close','Volume']]
dfreg['HiLo_PCT'] = (dataset['High'] - dataset['Low'])/dataset['Close'] * 100.0
dfreg['PCT_change'] = (dataset['Close'] - dataset['Open'])/dataset['Open'] * 100.0

#drop missing values
dfreg.fillna(value=-99999, inplace=True)

#separate 1 % of data to forecast
forecast_out = int(math.ceil(0.01 * len(dfreg)))

#sepearting the label. predict the adj_close
forecast_col = 'Adj Close'
dfreg['label'] = dfreg[forecast_col].shift(-forecast_out)
X = np.array(dfreg.drop(['label'],1))

#Scale X so everyone has same distribution for linear regression
X = preprocessing.scale(X)

# Finally We want to find Data Series of late X and early X (train) for model generation and evaluation
X_lately = X[-forecast_out:]
X = X[:-forecast_out]
# Separate label and identify it as y
y = np.array(dfreg['label'])
y = y[:-forecast_out]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, 
                                                    random_state = 0)

#Linear Regression
clfreg = LinearRegression(n_jobs=-1)
clfreg.fit(X_train,y_train)

#quadratic Regression ridge 
clfpoly2 = make_pipeline(PolynomialFeatures(2),Ridge())
clfpoly2.fit(X_train,y_train)

#Regression using lasso
clflasso = Lasso(alpha=0.1,fit_intercept=True, max_iter=1000,
   normalize=False, positive=False, precompute=False, random_state=None,
   selection='cyclic', tol=0.0001, warm_start=False)

clflasso.fit(X_train,y_train)


y_test_pred_linearReg = clfreg.predict(X_test)
y_test_pred_poly2 = clfpoly2.predict(X_test)
y_test_pred_lasso = clflasso.predict(X_test)

confidenceLinearReg = clfreg.score(X_test,y_test_pred_linearReg)
confidencePoly2 = clfpoly2.score(X_test,y_test_pred_poly2)
confidenceLasso = clflasso.score(X_test,y_test_pred_lasso)

print("confidence score of prediction for linear regression: ",confidenceLinearReg)
print("confidence score of prediction for Quadratic polynomial regression: ",confidencePoly2)
print("confidence score of prediction for Lasso regression: ",confidenceLasso)
print("\n\n")

from datetime import datetime,timedelta
from matplotlib.dates import (YEARLY, DateFormatter,
                              rrulewrapper, RRuleLocator, drange)
import datetime
import pylab
formatter = DateFormatter('%m/%d/%y')
date1 = datetime.date(2018,9,14)
date2 = datetime.date(2019 , 9, 13)
delta = datetime.timedelta(days=15)
dates = drange(date1, date2, delta)
plt.plot_date(dates,y_test_pred_linearReg,label='Linear Regression')
plt.plot_date(dates,y_test_pred_poly2,label='Quadratic Polynomial')
plt.plot_date(dates,y_test_pred_lasso,label='Lasso')
pylab.legend(loc='upper left')

#last_date = dataset.iloc[-1]
#last_unix = last_date
#next_unix = last_unix + timedelta(days=1)

#for i in y_test_pred_linearReg:
    #next_date = next_unix
    #next_unix += datetime.timedelta(days=1)
    #dfreg.loc[next_date] = [np.nan for _ in range(len(dfreg.columns)-1)]+[i]

#dataset['Date'].plot()
#y_test_pred_linearReg.tail(500).plot()
#plt.legend(loc=4)
#plt.xlabel('Date')
#plt.ylabel('Price')
plt.title('Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Precicted Stock Price')
plt.show()

    
    

# Adjusting the size of matplotlib
#import matplotlib as mpl
#mpl.rc('figure', figsize=(8, 7))
#mpl.__version__

# Adjusting the style of matplotlib
#style.use('ggplot')


#plt.legend('Linear_Regression','Quadratic','Lasso')






