import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Cargar el dataset
dataset = pd.read_csv("Num.csv")

# Separar las variables independientes (X) y dependiente (y)
X = dataset.iloc[:, 0:2].values
y = dataset.iloc[:, 3].values

# Transformar las características en un polinomio de grado 3 para ambas variables
poly_reg = PolynomialFeatures(degree=3)
X_poly = poly_reg.fit_transform(X)

# Ajustar el modelo de regresión polinómica
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)

# Generar una malla de puntos para hacer predicciones
# Aquí creamos todas las combinaciones posibles de valores de X[:, 0] y X[:, 1]
x0_values = np.linspace(min(X[:, 0]), max(X[:, 0]), 100)
x1_values = np.linspace(min(X[:, 1]), max(X[:, 1]), 100)
x0_mesh, x1_mesh = np.meshgrid(x0_values, x1_values)
X_mesh = np.column_stack((x0_mesh.ravel(), x1_mesh.ravel()))

# Transformar los datos de la malla en características polinómicas
X_poly_mesh = poly_reg.transform(X_mesh)

# Realizar predicciones para la malla de puntos
y_pred_mesh = lin_reg.predict(X_poly_mesh)

# Graficar los resultados de la regresión polinómica
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], y, color='red', label='Datos reales')
ax.plot_trisurf(X_mesh[:, 0], X_mesh[:, 1], y_pred_mesh, color='blue', alpha=0.5)
ax.set_title('Regresión Polinómica (Grado 3)')
ax.set_xlabel('Variable independiente 1')
ax.set_ylabel('Variable independiente 2')
ax.set_zlabel('Variable dependiente (y)')
plt.legend()
plt.show()
