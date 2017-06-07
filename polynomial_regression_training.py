# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 17:18:09 2017

@author: Asus
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
np.set_printoptions(threshold=np.inf)


# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values
                


# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

#Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
regressorLin = LinearRegression()
regressorLin.fit(X,y)                                                   
                                                   
#Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
regressorPoly = PolynomialFeatures(degree = 4)
X_poly = regressorPoly.fit_transform(X)
regressorLin_2 = LinearRegression()
regressorLin_2.fit(X_poly,y)

# Visualizing the Linear Regression results
plt.scatter(X,y, color = 'Red' )
plt.plot(X,regressorLin.predict(X), color = 'Blue')
plt.title('Truth or Bluff (Linear Regression) ')
plt.xlabel('Position Label')
plt.ylabel('Salary')
plt.show()


# Visualizing the Polynomial Regression results
X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape(len(X_grid),1 )
plt.scatter(X,y, color = 'Red' )
plt.plot(X_grid,regressorLin_2.predict(regressorPoly.fit_transform(X_grid)), color = 'Blue')
plt.title('Truth or Bluff (Polynomial Regression) ')
plt.xlabel('Position Label')
plt.ylabel('Salary')
plt.show()

#Predicting a new result with Linear Regression
regressorLin.predict(6.5)

#Predicting a new result with Polynomial Regression
regressorLin_2.predict(regressorPoly.fit_transform(6.5))


