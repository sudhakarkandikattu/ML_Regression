#simple leniar regression


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values  # X is indepenent variable
y = dataset.iloc[:, 1].values    # y is dependent variable

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

#fitting training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

#testing set
y_pred = regressor.predict(X_test)

#visualising training set
#screenshot 5
plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('salary vs experience(training set)')
plt.xlabel('years of exp')
plt.ylabel('salary')
plt.show()


#visualising testing set
#screenshot 6
plt.scatter(X_test,y_test,color='red')
plt.plot(X_test,y_pred,color='blue')
plt.title('salary vs experience(training set)')
plt.xlabel('years of exp')
plt.ylabel('salary')
plt.show()
