# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


# Load the Wine Quality Red dataset
dataset= pd.read_csv("winequality.csv")
x= dataset.iloc[:, :-1].values
y= dataset.iloc[:, -1].values
print(x)
print(y,"\n")

#Separate features and target
features= ["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"]
target= ["quality"]

#handling missing values
imputer= SimpleImputer(missing_values=np.nan, strategy= "mean")
x[:, 0:]= imputer.fit_transform(x[:, 0:])


# Split the dataset into an 80-20 training-test set
x_train, x_test, y_train, y_test= train_test_split(x,y, test_size=0.2)
print(x_train,"\n", x_test,"\n", y_train,"\n", y_test,"\n till here")

# Create an instance of the StandardScaler class
sc= StandardScaler()

x_train[:, :]= sc.fit_transform(x_train[:, :])
x_test[:, :]= sc.transform(x_test[:, :])

print(x_train,"\n",x_test)
print(y_train,"\n",y_test)