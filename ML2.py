import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

dataset= pd.read_csv("pima-indians-diabetes.csv")
x= dataset.iloc[:, :-1].values
y= dataset.iloc[:, -1].values

missing_values= dataset.isna()
print("Missing values are:\n",missing_values)

missing_per_column= dataset.isna().sum()
print("No. of missing values per column are:\n",missing_per_column, '\n')

total_missing_values= dataset.isna().sum().sum()
print("Total no. of missing values in dataset are:", total_missing_values, '\n')

imputer= SimpleImputer(missing_values=np.nan, strategy="mean",)

#differnt startegies in simpleimputer are:
#imputer= SimpleImputer(missing_values=np.nan, strategy="median")
#imputer= SimpleImputer(missing_values=np.nan, startegy="most_frequent")
#imputer1= SimpleImputer(missing_values=np.nan, strategy="constant", fill_value=33.0)
#imputer2= SimpleImputer(missing_values=np.nan, strategy="constant", fill_value=60000.0)

#x[:, 1:2]= imputer1.transform(x[:, 1:2])
#x[:, 2:3]= imputer2.transform(x[:, 2:3])

#or we can combine fit and transform modules into 1 fit_transform module

#x[:, 1:2]= imputer1.fit_transform(x[:, 1:2])
#x[:, 2:3]= imputer2.fit_transform(x[:, 2:3])

x[:, :8]= imputer.fit_transform(x[:, :8])

print(x)

