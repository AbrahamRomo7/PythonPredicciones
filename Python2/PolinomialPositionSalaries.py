import numpy as np #tratamiento de datos numericos
import matplotlib.pyplot as plt #Sublibreria para representar
import pandas as pd
from sklearn.linear_model import LinearRegression #Cargar y manipular datos
dataset = pd.read_csv("Position_Salaries.csv")

#Selecciono la variable independiente
X = dataset.iloc[:,1:2].values
#Selecciono la variable dependiente
y = dataset.iloc[:,2].values

lin_reg = LinearRegression()
lin_reg.fit(X, y)

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=3)
X_poly = poly_reg.fit_transform(X)

lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

plt.scatter(X, y, color='red')
plt.plot(X, lin_reg_2.predict(X_poly), color='blue')
plt.title('Regresión Polinómica (Grado 3)')
plt.xlabel('Nivel')
plt.ylabel('Salario')
plt.show()

