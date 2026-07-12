import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

dataset= pd.read_csv("titanic.csv")
x=dataset.iloc[:, :]
y=dataset.iloc[:, -1].values

# Identify the categorical data
categorical_features= ['Sex','Embarked','Pclass']  

#OneHotEncoder
ct= ColumnTransformer(transformers=[("encoder",OneHotEncoder(),categorical_features)], remainder="passthrough")
x=np.array(ct.fit_transform(x))    #convert to np array for future ease of use

#LabelEncoder
le= LabelEncoder()
y= le.fit_transform(y)

print(x,"\n")
print(y)