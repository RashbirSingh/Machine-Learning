#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 08:34:34 2018

@author: apple
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#reading csv data file
DataSet = pd.read_csv('50_Startups.csv')
X = DataSet.iloc[:, :-1].values
y = DataSet.iloc[:, 4].values

'''
#using mean to input mising value
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

'''
#encoding changing string lable of country and purchased to numeric
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_X = LabelEncoder()
X[:, 3] = labelEncoder_X.fit_transform(X[:, 3])
oneHotEncoder = OneHotEncoder(categorical_features = [3])
X = oneHotEncoder.fit_transform(X).toarray()

#avoiding the dummy variable trap
X = X[:, 1:]


#splitting data set into traing and test data set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

'''
#Scaling the values into same range 
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
'''

#fitting multiple regression tohe training set
from sklearn.linear_model import LinearRegression
Regressor = LinearRegression()
Regressor.fit(X_train, y_train)

#Predicting the test results
y_pred = Regressor.predict(X_test)


#building a optimal model using backward elimination
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((50,1)).astype(int) , values = X, axis = 1)
X_opt = X[:, [0, 3, 5]]
Regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
print(Regressor_OLS.summary())
