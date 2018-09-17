#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 22:38:22 2018

@author: apple
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#reading csv data file
DataSet = pd.read_csv('Position_Salaries.csv')
X = DataSet.iloc[:, 1:2].values
y = DataSet.iloc[:, 2:].values




'''#splitting data set into traing and test data set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
'''


#Scaling the values into same range 
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)


#Fittining regressor model to data set
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X, y)

# Predicting a new result with Polynomial Regression
def MakePrediction():
    y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))
    print(y_pred)

#plotting regression result
def plot():
    X_grid = np.arange(min(X), max(X), 0.1)
    X_grid = X_grid.reshape((len(X_grid), 1))
    plt.scatter(X, y, color = 'red')
    plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
    plt.xlabel("X Axis")
    plt.ylabel("Y Axis")
    plt.title("Regression Model")
    plt.show()


