import numpy as np #tratamiento de datos numericos
import matplotlib.pyplot as plt #Sublibreria para representar
import pandas as pd #Cargar y manipular datos
ds = pd.read_csv("Salary_Data.csv")
## x sería la variable independiente (Las 3 columnas y las 10 filas)
## Country, Age, Salary
x = ds.iloc[:,:-1].values
##y sería la variable dependiente
##y -> purchase Columna predictora, a la que vamos a predecir
y = ds.iloc[:,1].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state = 0)
#En regresión lineal no es necesario escalar datos
from sklearn.linear_model import LinearRegression

regression = LinearRegression()
regression.fit(X_train, y_train)

y_pred = regression.predict(X_test)
plt.scatter(X_train, y_train, color = "red")
plt.plot(X_train, regression.predict(X_train), color = "blue")
plt.title("Sueldo vs Años de Experiencia (conjunto de entrenamiento)")
plt.xlabel("Años de experiencia")
plt.ylabel("Sueldo ($)")


plt.scatter(X_test, y_test, color = "green")
plt.plot(X_test, regression.predict(X_test), color = "blue")
plt.title("Sueldo vs Años de Experiencia (conjunto de entrenamiento)")
plt.xlabel("Años de experiencia")
plt.ylabel("Sueldo ($)")
plt.show()