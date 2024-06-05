import numpy as np #tratamiento de datos numericos
import matplotlib.pyplot as plt #Sublibreria para representar
import pandas as pd
from sklearn.linear_model import LinearRegression #Cargar y manipular datos
dataset = pd.read_csv("Num.csv")
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,3].values



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
regression = LinearRegression()
regression.fit(X_train, y_train)
y_pred = regression.predict(X_test)

plt.scatter(X_train, y_train, color = "red")
plt.plot(X_train, regression.predict(X_train), color = "blue")
plt.title("(conjunto de entrenamiento)")
plt.xlabel(" Tarjetas")
plt.ylabel("Goles")
plt.show()