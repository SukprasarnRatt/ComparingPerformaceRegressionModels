import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv('Student_Performance.csv')
X = dataset.iloc[:, 0:-1].values
y = dataset.iloc[:, -1].values

# Handling missing data Independent Variables
from sklearn.impute import SimpleImputer
imputer_num = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer_num = imputer_num.fit(X[:, [0, 1, 3, 4]])
X[:, [0, 1, 3, 4]] = imputer_num.transform(X[:, [0, 1, 3, 4]])

imputer_cat = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imputer_cat = imputer_cat.fit(X[:, 2:3])
X[:, 2:3] = imputer_cat.transform(X[:, 2:3])

# Handling missing data Dependent Variable by frequency
imputer_num2 = SimpleImputer(missing_values=np.nan, strategy='mean')
y = y.reshape(-1, 1)
imputer_num2 = imputer_num2.fit(y)
y = imputer_num2.transform(y)

# Encoding independent variables with label encoding
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
X[:, 2] = labelencoder.fit_transform(X[:, 2])

# Reshape y to 2D array
y = y.reshape(len(y), 1)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X_train = sc_X.fit_transform(X_train)
y_train = sc_y.fit_transform(y_train)

# Training the SVR model on the Training set
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(X_test)).reshape(-1,1))
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# Evaluating the Model Performance
from sklearn.metrics import r2_score
print(r2_score(y_test, y_pred))