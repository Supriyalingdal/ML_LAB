import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
dataset = pd.read_csv("Desktop/data9.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
dataset.head()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)
y_pred = regressor.predict(X_test)
pd.DataFrame(data={'Actuals': y_test, 'Predictions': y_pred})
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
plt.figure(figsize=(4,3))
plt.scatter(y_test,y_pred)
plt.plot([0,50],[0,50],'--k')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.show()
