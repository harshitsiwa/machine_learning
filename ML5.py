import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer

dataset= pd.read_csv("Data.csv")
x= dataset.iloc[:, :-1].values
y= dataset.iloc[:, -1].values

imputer= SimpleImputer(missing_values=np.nan, strategy= "mean")
x[:, 1:3]= imputer.fit_transform(x[:, 1:3])
print(x)

#categorical_features= ['Country','Purchased']
ct= ColumnTransformer(transformers=[("encoder",OneHotEncoder(),[0])], remainder="passthrough")
x=np.array(ct.fit_transform(x))

le= LabelEncoder()
y=le.fit_transform(y)
print(x,"\n")
print(y)

x_train,x_test,y_train,y_test= train_test_split(x,y, test_size=0.2)
print(x_train,"\n")
print(x_test)

#Feature Scaling
sc= StandardScaler()
x_train[:, 3:]= sc.fit_transform(x_train[:, 3:])
x_test[:, 3:]= sc.fit_transform(x_test[:, 3:])
print(x_train,"\n")
print(x_test)