#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 17:28:23 2018

@author: apple
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#reading csv data file
DataSet = pd.read_csv('Position_Salaries.csv')
X = DataSet.iloc[:, 1:2].values
y = DataSet.iloc[:, 2].values

'''
#using mean to input mising value
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

#encoding changing string lable of country and purchased to numeric
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_X = LabelEncoder()
X[:, 0] = labelEncoder_X.fit_transform(X[:, 0])
oneHotEncoder = OneHotEncoder(categorical_features = [0])
X = oneHotEncoder.fit_transform(X).toarray()
labelEncoder_Y = LabelEncoder()
Y = labelEncoder_Y.fit_transform(Y)
'''

'''#splitting data set into traing and test data set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
'''

'''
#Scaling the values into same range 
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
'''
# Visualising the Linear Regression results
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Visualising the Polynomial Regression results
from sklearn.preprocessing import PolynomialFeatures
Poly_reg = PolynomialFeatures(degree = 6)
X_poly = Poly_reg.fit_transform(X)
Poly_reg.fit(X_poly, y)
Lin_reg_2 = LinearRegression()
Lin_reg_2.fit(X_poly, y)


#plotting linear
plt.scatter(X, y)
plt.plot(X, lin_reg.predict(X))
plt.show()

#plotting poly nomial
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y)
plt.plot(X_grid, Lin_reg_2.predict(Poly_reg.fit_transform(X_grid)), color = 'red')
plt.show()

# Predicting a new result with Polynomial Regression
print(Lin_reg_2.predict(Poly_reg.fit_transform(6.5)))
