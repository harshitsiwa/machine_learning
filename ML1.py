import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset= pd.read_csv("Data.csv")
x= dataset.iloc[:, :-1].value
y= dataset.iloc[:, -1].values
print(x)
print(y)

