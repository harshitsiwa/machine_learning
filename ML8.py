import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

dataset= pd.read_csv("Position_Salaries.csv")
x= dataset.iloc[:, 1:-1].values
y= dataset.iloc[:, -1].values
print(x,"\n")
print(y)

lin_reg= LinearRegression()
lin_reg.fit(x,y)

poly_reg= PolynomialFeatures(degree=4)
x_poly= poly_reg.fit_transform(x)
print(x_poly)
lin_reg2= LinearRegression()
lin_reg2.fit(x_poly, y)

#linear regression
plt.scatter(x, y, color="red")
plt.plot(x, lin_reg.predict(x), color="blue")
plt.title("Truth or Bluff")
plt.xlabel('position lvl')
plt.ylabel('salary')
plt.show()

#polynomial regression
plt.scatter(x, y, color='red')
plt.plot(x, lin_reg2.predict(x_poly), color='blue')
plt.title("poly reg.")
plt.xlabel('position lvl')
plt.ylabel('salary')
plt.show()

#smoother curve by taking decimal coordinates 
X_grid = np.arange(min(x), max(x), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(x, y, color = 'red')
plt.plot(X_grid, lin_reg2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#predicting salary using lin reg
print(lin_reg.predict([[6.5]]))

#using poly reg
print(lin_reg2.predict(poly_reg.fit_transform([[6.5]])))


