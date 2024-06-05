import numpy as np #tratamiento de datos numericos
import matplotlib.pyplot as plt #Sublibreria para representar
import pandas as pd
from sklearn.linear_model import LinearRegression #Cargar y manipular datos
dataset = pd.read_csv("50_Startups.csv")

#Selecciono la variable independiente
X = dataset.iloc[:,:-1].values
#Selecciono la variable dependiente
y = dataset.iloc[:,4].values

from sklearn import preprocessing
labelencoder_X = preprocessing.LabelEncoder()

X[:,3] = labelencoder_X.fit_transform(X[:,3])

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

onehotencoder = ColumnTransformer([('one_hot_encoder', OneHotEncoder(categories='auto'),[3])],remainder='passthrough')

X = np.array(onehotencoder.fit_transform(X),dtype=float)

X=X[:,1:]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

regression = LinearRegression()
regression.fit(X_train, y_train)
y_pred = regression.predict(X_test)

plt.scatter(X_train, y_train, color = "red")
plt.plot(X_train, regression.predict(X_train), color = "blue")
plt.title("Gastos*Ciudad vs Utilidad (conjunto de entrenamiento)")
plt.xlabel("Gastos*Ciudad")
plt.ylabel("Utilidad")
plt.show()







